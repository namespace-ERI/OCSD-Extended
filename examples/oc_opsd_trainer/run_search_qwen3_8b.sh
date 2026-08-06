set -x
ENGINE=${1:-vllm}

export WANDB_API_KEY=your_key_here
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

beta=0.5
eps_A=1e-6
env_name=search
K_future=2

experiment_name="oc_opsd_search_qwen3_8b_top20nll_beta${beta}_K${K_future}"

train_data_size=128
val_data_size=512
group_size=8

BASE=${WORKSPACE_ROOT:-/path/to/workspace}
export MODEL_PATH=${MODEL_DIR:-/path/to/pretrained_models}/Qwen3-8B
export WANDB_MODE=disabled

RUN_DATE=$(date +%Y%m%d_%H%M%S)

cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SKILLRL_ROOT="$(dirname "$(dirname "$cur_dir")")"

cd "$SKILLRL_ROOT" || exit 1

TRAIN_DATA="${BASE}/data/searchR1_processed_direct/train.parquet"
VAL_DATA="${BASE}/data/searchR1_processed_direct/test.parquet"

SEARCH_PORT=8000
CBMODEL=${WORKSPACE_ROOT:-/path/to/workspace}/cbmodel
INDEX_FILE="${CBMODEL}/data/agent/search/searchR1_index/e5_Flat.index"
CORPUS_FILE="${CBMODEL}/data/agent/search/searchR1_index/wiki-18.jsonl"
RETRIEVER_MODEL="${CBMODEL}/models/e5-base-v2"

echo "=== Starting retrieval server (port ${SEARCH_PORT}) ==="
pip install uvicorn fastapi ${PIP_MIRROR} --target "${EXTRA}" --quiet 2>/dev/null || true
python3 -c "import faiss" 2>/dev/null || \
    pip install faiss-cpu ${PIP_MIRROR} --target "${EXTRA}" --quiet 2>/dev/null || true

export HF_DATASETS_CACHE="${BASE}/data/hf_cache"
mkdir -p "${HF_DATASETS_CACHE}"
echo "[Search] HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"

python3 "${SKILLRL_ROOT}/examples/search/retriever/retrieval_server.py" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model "${RETRIEVER_MODEL}" \
    --num_threads 32 \
    --port ${SEARCH_PORT} > "${LOG_DIR}/retrieval_server.log" 2>&1 &
RETRIEVAL_PID=$!
echo "Retrieval server PID=${RETRIEVAL_PID}, waiting for startup..."

SERVER_UP=0
for i in $(seq 1 360); do
    sleep 5
    if ! kill -0 ${RETRIEVAL_PID} 2>/dev/null; then
        echo "[ERROR] Retrieval server died! Full log:"
        cat "${LOG_DIR}/retrieval_server.log"
        exit 1
    fi
    if curl -s --max-time 5 -X POST "http://127.0.0.1:${SEARCH_PORT}/retrieve" \
        -H "Content-Type: application/json" \
        -d '{"query":"test","topk":1,"return_scores":false}' > /dev/null 2>&1; then
        echo "Search server UP (${i}x5s elapsed)"
        SERVER_UP=1
        break
    fi
    [ $((i % 12)) -eq 0 ] && echo "[WAIT] ${i}x5=$(( i * 5 ))s elapsed, still loading corpus..."
done
if [ $SERVER_UP -eq 0 ]; then
    echo "[ERROR] Retrieval server did not respond within 1800s. Last log:"
    tail -20 "${LOG_DIR}/retrieval_server.log"
    exit 1
fi

python3 -m verl.trainer.main_oc_opsd \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +algorithm.oc_opsd.beta=$beta \
    +algorithm.oc_opsd.eps_A=$eps_A \
    +algorithm.oc_opsd.env_name=$env_name \
    +algorithm.oc_opsd.K_future=$K_future \
    +algorithm.oc_opsd.top_step_ratio=0.2 \
    +env.thinking_tag=reason \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.timeout=600 \
    env.search.search_url='http://127.0.0.1:${SEARCH_PORT}/retrieve' \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_search_OC_OPSD' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=140 \
    trainer.test_freq=150 \
    trainer.total_epochs=999 \
    trainer.total_training_steps=150 \
    trainer.default_local_dir=${BASE}/checkpoints/searchqa/${experiment_name}/${RUN_DATE} \
    trainer.val_before_train=False ${@:2}
