import json
import os
import warnings
from typing import List, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False, use_gpu: bool = True):
    # model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    if use_gpu:
        model.cuda()
        if use_fp16:
            model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, use_gpu=True):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.use_gpu = use_gpu

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, use_gpu=use_gpu)
        self.model.eval()
        device = next(self.model.parameters()).device
        print(f"[Encoder] model loaded on device: {device}, use_gpu={use_gpu}", flush=True)

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        import time
        _t0 = time.time()
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        print(f"[Encoder] encode {len(query_list)} queries in {time.time()-_t0:.3f}s", flush=True)
        return query_emb


class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        import time as _time
        super().__init__(config)

        # Diagnostic 1: file check────────────────────────────────────────
        print(f"[DenseRetriever] index_path  : {self.index_path}", flush=True)
        print(f"[DenseRetriever] corpus_path : {self.corpus_path}", flush=True)
        print(f"[DenseRetriever] model_path  : {config.retrieval_model_path}", flush=True)
        for label, path in [("index", self.index_path),
                             ("corpus", self.corpus_path),
                             ("model_dir", config.retrieval_model_path)]:
            exists = os.path.exists(path)
            size_str = ""
            if exists and os.path.isfile(path):
                size_str = f"  size={os.path.getsize(path)/1e9:.2f}GB"
            print(f"[DenseRetriever] {label} exists={exists}{size_str}", flush=True)

        # Load FAISS index───────────────────────────────────────
        t0 = _time.time()
        print(f"[DenseRetriever] Loading FAISS index...", flush=True)
        self.index = faiss.read_index(self.index_path)
        print(f"[DenseRetriever] FAISS index loaded in {_time.time()-t0:.1f}s  "
              f"ntotal={self.index.ntotal}  d={self.index.d}  "
              f"type={type(self.index).__name__}", flush=True)

        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
            print(f"[DenseRetriever] FAISS index moved to GPU (float16, sharded)", flush=True)

        # Load corpus────────────────────────────────────────────
        t0 = _time.time()
        print(f"[DenseRetriever] Loading corpus from {self.corpus_path} ...", flush=True)
        self.corpus = load_corpus(self.corpus_path)
        n = len(self.corpus)
        print(f"[DenseRetriever] Corpus loaded in {_time.time()-t0:.1f}s  "
              f"n_docs={n}", flush=True)
        # Print first sample to verify field format
        if n > 0:
            sample = self.corpus[0]
            print(f"[DenseRetriever] corpus[0] keys={list(sample.keys())}  "
                  f"sample={str(sample)[:200]}", flush=True)

        # ── Encoder ──────────────────────────────────────────────────
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            use_gpu=config.retrieval_use_gpu,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        print(f"[DenseRetriever] Ready. topk={self.topk}  faiss_gpu={config.faiss_gpu}  "
              f"encoder_gpu={config.retrieval_use_gpu}", flush=True)

    def _search(self, query: str, num: int = None, return_score: bool = False):
        import time as _time
        import threading
        tid = threading.get_ident()
        if num is None:
            num = self.topk
        t_total = _time.time()

        # encode
        t0 = _time.time()
        query_emb = self.encoder.encode(query)
        t_encode = _time.time() - t0

        # FAISS search
        t0 = _time.time()
        scores, idxs = self.index.search(query_emb, k=num)
        t_faiss = _time.time() - t0
        idxs = idxs[0]
        scores = scores[0]

        # corpus lookup
        t0 = _time.time()
        results = load_docs(self.corpus, idxs)
        t_corpus = _time.time() - t0

        t_total = _time.time() - t_total
        print(f"[Search tid={tid}] encode={t_encode:.3f}s  faiss={t_faiss:.3f}s  "
              f"corpus={t_corpus:.3f}s  total={t_total:.3f}s  "
              f"query={query[:50]!r}  top_idx={idxs[:3].tolist()}", flush=True)

        # Sanity check on returned results
        if not results:
            print(f"[Search WARNING] Empty results for query={query[:80]!r}", flush=True)
        else:
            first = results[0]
            print(f"[Search] result[0] keys={list(first.keys()) if isinstance(first,dict) else type(first)}  "
                  f"snippet={str(first)[:100]}", flush=True)

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)

            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()
            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]
            results.extend(batch_results)
            scores.extend(batch_scores)
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        retrieval_use_gpu: bool = False,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_use_gpu = retrieval_use_gpu


class QueryRequest(BaseModel):
    query: str
    topk: Optional[int] = None
    return_scores: bool = False


class BatchQueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

# anyio concurrent token count (set in __main__ before uvicorn start)
# Default 4: 4 Python x 8 OMP = 32 CPU threads (32-core machine)
# Setting 8: 8 Python x 4 OMP = 32 CPU threads (when --num_threads 16)
_anyio_concurrent_tokens: int = 4


@app.on_event("startup")
async def startup_event():
    import anyio
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = _anyio_concurrent_tokens


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if not request.topk:
        request.topk = config.retrieval_topk

    if request.return_scores:
        results, scores = retriever.search(query=request.query, num=request.topk, return_score=True)
    else:
        results = retriever.search(query=request.query, num=request.topk, return_score=False)
        scores = None

    resp = []
    if request.return_scores and scores is not None:
        combined = []
        for doc, score in zip(results, scores):
            combined.append({"document": doc, "score": float(score)})
        resp.append(combined)
    else:
        resp.append(results)
    return {"result": resp}


@app.post("/batch_retrieve")
def batch_retrieve_endpoint(request: BatchQueryRequest):
    """
    Batch endpoint: accepts multiple queries at once via _batch_search (GPU-friendly batching).
    Reduces round-trips and amortizes encoder cost across queries.
    """
    if not request.topk:
        request.topk = config.retrieval_topk

    if request.return_scores:
        all_results, all_scores = retriever.batch_search(
            query_list=request.queries, num=request.topk, return_score=True
        )
    else:
        all_results = retriever.batch_search(
            query_list=request.queries, num=request.topk, return_score=False
        )
        all_scores = [None] * len(request.queries)

    resp = []
    for i, results in enumerate(all_results):
        if request.return_scores and all_scores[i] is not None:
            combined = [{"document": doc, "score": float(score)}
                        for doc, score in zip(results, all_scores[i])]
            resp.append([combined])
        else:
            resp.append([results])
    return {"results": resp}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for faiss index")
    parser.add_argument("--encoder_gpu", action="store_true", help="Use GPU for encoder model (default: CPU)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on.")
    parser.add_argument("--num_threads", type=int, default=16,
                        help="Number of CPU threads for torch encoder. Use more on machines with many cores.")
    parser.add_argument("--anyio_concurrent", type=int, default=4,
                        help="Max concurrent anyio Python threads for FAISS queries. "
                             "faiss_omp = num_threads // anyio_concurrent. "
                             "anyio_concurrent × faiss_omp should not exceed machine CPU cores. "
                             "e.g. --num_threads 32 --anyio_concurrent 4: 4×8=32 threads (default). "
                             "e.g. --num_threads 32 --anyio_concurrent 8: 8×4=32 threads (2x throughput).")

    args = parser.parse_args()

    # Set CPU thread count for torch encoder to utilize multi-core CPUs
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(max(1, args.num_threads // 2))

    # Set FAISS OMP thread count (independent of torch threads)
    # faiss_omp x anyio_concurrent <= CPU cores, to avoid over-subscribing OMP threads
    # e.g. --num_threads 32 --anyio_concurrent 4: 4x8=32 (32-core machine)
    # e.g. --num_threads 32 --anyio_concurrent 8: 8x4=32 (32-core machine, higher concurrency)
    faiss_omp = max(1, args.num_threads // args.anyio_concurrent)
    faiss.omp_set_num_threads(faiss_omp)

    # Store anyio_concurrent in a global for startup_event() to read
    global _anyio_concurrent_tokens
    _anyio_concurrent_tokens = args.anyio_concurrent

    print(f"[Server] torch_threads={args.num_threads}  faiss_omp_threads={faiss_omp}  "
          f"anyio_concurrent={args.anyio_concurrent}", flush=True)

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,  # this is unused in the current retrieval implementation, which only supports single query
        retrieval_use_gpu=args.encoder_gpu,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=args.port)
