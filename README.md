# Agentic Reinforcement Learning with Observation-Calibrated Self-Distillation

## MiniHack and ScienceWorld extension

This repository extends [yiy1x/OCSD](https://github.com/yiy1x/OCSD) at
`7cfdf110d05d2a745166a9162d3c9222f85117d6`. It adds **MiniHack Room** and
**ScienceWorld** to the existing OCSD trainers and keeps the original three
benchmark implementations. Upstream attribution and Apache-2.0 licensing are
retained; environment adapter provenance is in [NOTICE-NEW-BENCHMARKS.txt](NOTICE-NEW-BENCHMARKS.txt).

Supported new-benchmark methods are `oc_opsd` (OCSD / OC-OPSD), `grpo`, `opsd`,
`grpo_opsd`, `sdar`, `rlsd`, `skillsd`, `skill_grpo`, `gigpo`, `gigpo_dynamic`,
`rloo`, `dapo`, `gspo`, and `ppo`. The launchers reuse the
original trainer entry points and losses. Skill-based teachers receive the
frozen bank through an exact native task-type route. OCSD continues to use
future observations rather than a skill bank. Skill-GRPO adds skills to acting
prompts, retaining its upstream with/without-skills validation behavior.

### Installation for the added environments

First install the original environment below. In a dedicated environment:

```bash
python -m pip install -r requirements-new-benchmarks.txt
python -m pip install --no-deps -r requirements-minihack-no-deps.txt
python -m pip install -e . --no-deps
```

Use Linux, Python3.12 and Java17 (`JAVA_HOME` / `PATH` must expose `java`).
NLE/MiniHack may need CMake and a C++ build toolchain if wheels are unavailable.
NLE1.3 declares gymnasium1.2, while upstream OCSD pins0.29.1. The explicit
`--no-deps` installation preserves the original environments; the MiniHack
reset/step/win path was tested with gymnasium0.29.1. Do not let a resolver
silently upgrade the original environment. The added packages and their native
simulator assets are installed dependencies, not Git-tracked binaries.

For many ScienceWorld workers, bound Java resources, for example:

```bash
export JAVA_TOOL_OPTIONS='-Xmx512m -XX:ActiveProcessorCount=1 -XX:+UseSerialGC'
```

### Included data and external downloads

**Both new skill libraries and all training/evaluation task inputs are committed
to this public repository.** No private shared-storage path or extraction API
is needed to use them. A regular clone downloads them without Git LFS:

```bash
git clone https://github.com/namespace-ERI/OCSD-Extended.git
cd OCSD-Extended
python scripts/new_benchmarks/verify_release.py
```

| Included in Git | Size | Purpose |
|---|---:|---|
| [skills/minihack_room](skills/minihack_room) | 1.29 MiB | Frozen Markdown/JSON prompt bank and all1,000 source strategy records |
| [skills/scienceworld](skills/scienceworld) | 5.36 MiB | Frozen prompt bank and all3,194 non-dev source strategy records |
| [data/new_benchmarks](data/new_benchmarks) | About200 KiB of task manifests, plus release metadata | Exact seeds, task/variation pairs, dev holdout and fixed test selection |
| [data/prepared](data/prepared) | 78.4 KiB total,7 parquets | Training, dev and test inputs; also ScienceWorld `test_full` |

These are **online interactive RL benchmarks**, not static question/answer or
offline trajectory datasets. MiniHack creates a level from each seed;
ScienceWorld loads the named task/variation in its simulator. The parquets
already contain every selected task identity. Goals and observations arrive
from the simulator at reset. The original Qwen rollout chats, model weights
and native simulator binaries are not bundled in Git. Raw rollout chats have
no public download URL in this release and are not required for RL training
or using the frozen banks. The published `evidence.jsonl` files contain the
extracted strategies, source identities and hashes, not the original chats.

Install/download simulator assets from these official package releases:

| External asset | Official download | Installed content |
|---|---|---|
| ScienceWorld1.2.3 | [PyPI release](https://pypi.org/project/scienceworld/1.2.3/) | Python interface and bundled simulator JAR; Java17 required by this setup |
| MiniHack1.0.2 | [PyPI release](https://pypi.org/project/minihack/1.0.2/) | Task generator and level definitions |
| NLE1.3.0 | [PyPI release](https://pypi.org/project/nle/1.3.0/) | Native NetHack engine; use the appropriate Linux/Python wheel |
| Qwen3-4B | [Official model](https://huggingface.co/Qwen/Qwen3-4B) | Initial model/tokenizer, downloaded separately |

The installation commands above fetch the simulator packages; no separate
large benchmark corpus download is needed. [sources.json](data/new_benchmarks/sources.json)
records the benchmark sources, pinned package URLs and SHA256 hashes of the
distributions used for validation. Its NLE wheel is specifically Linux x86_64,
CPython3.12; another platform needs its own matching distribution.

For an offline GPU node, build a wheel bundle on an internet-connected Linux
machine with the same Python/platform and an existing compatible OCSD
environment, then copy `downloads/new-benchmarks` to the training machine:

```bash
python -m pip wheel --wheel-dir downloads/new-benchmarks \
  -r requirements-new-benchmarks.txt
python -m pip wheel --no-deps --wheel-dir downloads/new-benchmarks \
  -r requirements-minihack-no-deps.txt

# On the training machine, after installing the original OCSD environment:
python -m pip install --no-index --find-links downloads/new-benchmarks \
  -r requirements-new-benchmarks.txt
python -m pip install --no-index --find-links downloads/new-benchmarks \
  --no-deps -r requirements-minihack-no-deps.txt
```

Download model weights once on the connected machine, then use the resulting
local directory as `MODEL_PATH` on every training node:

```bash
export HF_HOME="$PWD/downloads/huggingface-cache"
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B", local_dir="models/Qwen3-4B")
PY
export MODEL_PATH="$PWD/models/Qwen3-4B"
```

The release checksum checker uses only Python's standard library. Run it on
the original checkout before optional parquet regeneration; different PyArrow
versions can produce different parquet bytes for identical task identities.
After regeneration, use `audit_counts.py --prepared data/prepared` to check
identity equality instead. The release manifest covers the two new banks and
their data inputs; the original three benchmarks retain their own setup below.

### Data, splits, and training budget

Checked-in JSON manifests specify actual task identities, and the corresponding
parquets are already included. Optional regeneration is offline and does not
download the unrelated Geometry3k placeholder dataset used by the old preparation
script:

```bash
python scripts/new_benchmarks/prepare_data.py
```

| Added environment | Training pool | Development | Final test | Horizon | Prompt / response tokens |
|---|---:|---:|---:|---:|---:|
| MiniHack-Room-Ultimate-5x5-v0 | 1,000 fixed seeds | 128 fixed seeds | 128 fixed seeds (author preprocessing default; see limitation below) | 30 | 12,288 / 2,048 |
| ScienceWorld BEACON L0,29 task families | 3,194 task/variation pairs | 128 pairs | 128 fixed pairs selected from 1,661 | 30 | 15,000 / 512 |

MiniHack training seeds are the same1,000 used for strategy extraction. Dev
uses128 distinct seeds in `[2**30, 2**31-1)`, sampled with seed20261016.
Test uses the128 contiguous seeds `2**30 + i`, `i=0..127`, following the
author's preprocessing defaults; all three sets are disjoint. The initial
NLE engine seeds, including level-generator seeds and fixed moon phase,
are explicitly set.

The source ScienceWorld L0 protocol contains3,322 train and1,661 test pairs and
no dev split. This extension preserves **all1,661 pairs as the full test pool**
and uses a **fixed128-pair subset for the paper evaluation**, as specified in
CompeteRL Appendix B.1. Selection follows the author preprocessing algorithm:
`sha256_pair_rank_v1`, evaluation seed1. Exact identities are checked in as
[scienceworld_paper_eval_selection.json](data/new_benchmarks/scienceworld_paper_eval_selection.json),
generated with that algorithm and its default evaluation seed.
The larger pool is available separately as `test_full`; it is not the paper's
reported128-instance evaluation. We hold out128 from the original training
pool: up to4 per task family followed by a
deterministic hash fill (seed20261016). The remaining3,194 are used for RL and
skill construction. This128-pair dev split is our extension protocol, not an
official ScienceWorld dev set. It covers all29 L0 families. Train/dev/test
membership is checked for overlap; no dev/test trajectory enters the skill bank.

The main OCSD/GRPO-family default budget follows the upstream ALFWorld/WebShop
scripts. This table is not a claim that every upstream algorithm has identical
hyperparameters:

| Setting | Upstream ALFWorld | Upstream WebShop | Upstream Search-QA | New environments |
|---|---:|---:|---:|---:|
| Tasks per update | 16 | 16 | 128 | 16 |
| Rollouts per task | 8 | 8 | 8 | 8 |
| Outer updates | 150 | 150 | 150 | 150 |
| Periodic validation batch | 128 | 128 | 512 | 128 |

Defaults for the fixed-sampling methods collect **19,200 training episodes per method**
(`150 × 16 × 8`), at most576,000 environment steps. This matches the original
long-horizon benchmarks' episode budget; Search uses more questions per update
and only4 interaction steps, so its batch size is not copied here. MiniHack
has2,400 task draws across1,000 seeds; ScienceWorld has2,400 draws across3,194
pairs. Neither setting promises every training task will appear in one run.

Two upstream exceptions need explicit treatment:

* **PPO:** the original ALFWorld/WebShop scripts use128 independent tasks and
  one rollout each, rather than16 tasks repeated8 times. The new PPO launchers
  retain **128 × 1 × 150**, also19,200 episodes. PPO therefore has more task
  draws than the grouped algorithms, matching the upstream comparison design.
* **GSPO:** the original ALFWorld example runs200 epochs. The new-environment
  default deliberately caps GSPO at150 updates to match the other methods'
  episode budget; this is our budget normalization, not an exact reproduction
  of that200-epoch setting. Its loss and clipping remain unchanged.

Pool sizes are separate choices from update batch sizes. MiniHack's1,000
training seeds follow the requested extraction pool;128 dev follows the old
validation-batch scale. ScienceWorld's3,322/1,661 source pools come from the
BEACON L0 protocol, with the documented128-train-instance holdout.

**MiniHack reproduction limitation:** CompeteRL specifies fixed held-out seeds
but its PDF does not give their number or identities. The author launcher
defaults to64 validation instances, while the author preprocessing script
defaults to128 contiguous seeds starting at `2**30`. This release explicitly
chooses the latter; it does **not** claim to recover the exact test manifest
used for the paper's reported MiniHack scores. To regenerate either code
default, run the following and then prepare the parquets again:

```bash
python scripts/new_benchmarks/align_paper_test.py --minihack-test-count 128
python scripts/new_benchmarks/prepare_data.py
```

Use `--minihack-test-count 64` for the author launcher-sized alternative;
report that choice with results. ScienceWorld selection is unchanged by this
option. These operations preserve train/dev identities and skill libraries.

### CompeteRL training settings for reference

The test profile above follows CompeteRL Appendix B.1 (pp14–15) and evaluation
temperature in Table4 (p15). Its **training** settings differ from the OCSD
budget retained here. Appendix B.2 and Table4 report:

| CompeteRL setting | MiniHack Room | ScienceWorld |
|---|---:|---:|
| Training tasks | No finite seed count stated in PDF; author code samples seeds online from `[0, 2**30)` | 3,322 L0 task/variation pairs |
| Maximum optimizer updates | 200 | 300 |
| Task prompts / rollouts per task | 16 / 8 | 16 / 8 |
| Maximum environment steps per episode | 30 | 30 |
| Prompt / response tokens | 12,288 / 2,048 | 15,000 / 512 |
| PPO micro-batch per GPU | 8 | 1 |
| Evaluation temperature | 0 | 0 |

Shared paper settings are learning rate `1e-6`, KL coefficient `0.01`, PPO
mini-batch256, training temperature1/top-p1, BF16/FSDP with gradient
checkpointing, and4 GPUs with vLLM tensor parallelism4. Its foresight and
hindsight phases execute paired trajectories from the same task reset.
`200 × 16 × 8 = 25,600` and `300 × 16 × 8 = 38,400` are the nominal
sampling slots **per phase**, not the total executions across both phases.

This extension still uses the OCSD comparison budget of150 updates and the
1,000/3,194 training pools above. It has **not** changed training to the
CompeteRL200/300-update budgets or restored the128 held-out ScienceWorld dev
pairs to training. Source facts and these differences are also recorded in
[competerl_paper_settings.json](validation/competerl_paper_settings.json),
including the supplied PDF's SHA256. MiniHack online sampling is an observation
from author code, not a finite training-data count supplied by the PDF.

`trainer.total_training_steps=150` is explicit. The old16-row
placeholder parquet made150 epochs equal150 updates; using the full task pool
with only an epoch limit would accidentally multiply the budget. The new
parquets contain one real identity per task, with shuffled training batches;
each identity is repeated8 times for grouped rollout (once for PPO). Validation uses one
rollout per identity and a fixed dev set; final test visits all identities in
the selected test parquet, including the final partial batch. Training defaults reject selecting test as
the periodic evaluation split.

MiniHack reward is native binary success. ScienceWorld uses native
`score_delta / 100`, `simplification=easy`, success at score100, and preserves
negative-score terminal failures. It reports native score alongside success.
An admissible command is parseable, not guaranteed useful or safe. Invalid
format/admissibility becomes an explicit no-op; it never silently executes a
different action. All methods share the same new-benchmark rewards, splits,
step limits, sampler and prompts. Qwen3 native thinking is disabled, with
explicit `<reason>` tags for action reasoning, as in the upstream Qwen3 setup.

DAPO and dynamic GiGPO retain their upstream group filtering (up to10 sampling
attempts) and asymmetric clipping0.2/0.28. They can generate additional discarded
episodes; report that extra sampling cost rather than treating19,200 as their
total generated episode count. GSPO keeps0.0003/0.0004 clipping. PPO uses a
value-head critic initialized from the same backbone, with critic LR1e-5 and
128 independent tasks per update.

### Frozen skills and matching

The released libraries are in `skills/minihack_room` and `skills/scienceworld`:

* `claude_style_skills.json`: general, category skills and common mistakes.
* `general_skills.md`, category Markdown files, `skill_mapping.json`: the
  original OCSD `SkillProvider` interface.
* `evidence.jsonl`: every eligible extracted strategy, with source/prompt hashes
  and the fixed first-rollout `r00` identity; raw chats and private paths are omitted.

Strategies were generated using SkillRL-style extraction with
`glm53_512k_o64k_pd128_coding` from Qwen3-4B training rollouts. All1,000 MiniHack
strategy sections are present; ScienceWorld exports only the3,194 non-dev
strategies. The two outstanding optional refined-trajectory sections are not
needed by these fixed-bank baselines and are not used to invent missing evidence.

The compiled prompt banks contain21 positive workflow entries for MiniHack
(15 broad +6 navigation) and77 for ScienceWorld (15 broad +62 across12
nonempty categories). All29 ScienceWorld categories have explicit mapping
entries, including the17 without successful evidence. These workflow counts
are distinct from the1,000/3,194 source-strategy counts.

The default is `skill_all=false`. The training process caches the library once, then
each skill-based teacher prompt receives **all15 general entries plus the
current native category's entries**. MiniHack therefore supplies21 entries;
ScienceWorld supplies15–21 (15 for each of the17 categories without successful
specific evidence). Its62 specific entries are the total across the library,
not62 entries per task. There is no model-based retrieval or top-k selection
in this path. Setting the method's `skill_all=true` override explicitly changes
this to the whole bank,21/77 entries, and is not the comparison default.

The15/6 compilation budget follows WebShop's bank scale. Original ALFWorld has
12 general plus5/6/10 per matched category, WebShop15 plus5/6, and Search10
plus5. These are bank sizes, not a guarantee of identical prompt token counts.
The legacy teacher routing can send `data_source=text` through the Search
lookup before keyword matching and return only general skills for original
ALFWorld/WebShop inputs. The new environments explicitly prioritize native
`task_type`; this extension does not alter that legacy routing behavior.

This release uses **deterministic extractive compilation**, not an additional
LLM semantic synthesis: concise, supported and diverse existing workflow
strings are selected verbatim, retaining their evidence IDs and applicability.
The prompt budget matches the original WebShop library: at most15 general
entries and6 per category; common mistakes retain5 entries in the JSON, as
auxiliary data. The OCSD Markdown teacher interface consumes general/category
skills, as upstream does. Some broad-bank workflows retain a task-family scope
and must not be read as universal scientific procedures. Complete strategy
evidence remains available for future synthesis/review. No new API call is
required to use this frozen release.

The selected first-rollout strategies contain587 MiniHack successes and247
ScienceWorld training successes. ScienceWorld has **17/29 categories without
successful evidence**. Their category files explicitly say so and the teacher
falls back to the broad bank; no successful procedure is fabricated. Sparse
skill evidence is a limitation of this release, not a claimed baseline result.

Matching is deterministic and uses environment namespace plus native type:
`room_navigation` for MiniHack, and ScienceWorld's exact `task_name` for
ScienceWorld. Seeds and variation IDs identify simulator instances, not
retrieval keys. The category propagates through the per-turn DataProto before
teacher construction, taking priority over old Search `data_source` routing.
This prevents all ScienceWorld tasks from falling into an old first-category
fallback. Skill-GRPO uses the same route in its acting prompt.

To rebuild from a local extraction run with `inputs.jsonl` and cached strategy
sections, using the checked-in split manifests:

```bash
python scripts/new_benchmarks/export_skills.py --source /path/to/extraction-run
```

### Training and final evaluation

Launch any method on one8-GPU allocation. `MODEL_PATH` may be a local Qwen3
1.7B/4B/8B checkpoint; all comparisons should use the same initial checkpoint.

```bash
export MODEL_PATH=/path/to/Qwen3-4B
export OUTPUT_DIR=/path/to/outputs/scienceworld/oc_opsd
bash examples/oc_opsd_trainer/run_scienceworld_qwen3_4b.sh

# Equivalent common launcher; replace --method with any supported method.
python scripts/new_benchmarks/train.py --env minihack_room --method sdar \
  --model "$MODEL_PATH" --output /path/to/outputs/minihack/sdar

# Inspect all overrides without allocating GPU resources.
python scripts/new_benchmarks/train.py --env scienceworld --method rlsd --dry-run
```

Both new environments have wrappers under each original method directory.
The following are the **actual extension launcher defaults**, distinct from
the CompeteRL paper settings in the reference table above. See
[train.py](scripts/new_benchmarks/train.py) for the complete arguments.

| Shared launcher parameter | Default |
|---|---|
| GPU allocation / rollout tensor parallelism | One node,8 GPUs / TP2 |
| Optimizer updates | 150, explicitly capped by `trainer.total_training_steps` |
| Task batch / rollouts per task | 16 /8; PPO128 /1 |
| Actor learning rate | `1e-6` |
| PPO mini-batch / actor micro-batch per GPU | 64 /8 for both new environments |
| Actor KL | Enabled, `low_var_kl`, coefficient0.01; no KL term added to reward |
| Training temperature / top-p | 1 /1 |
| Dev / final-test temperature | 0.4 sampled /0 greedy |
| Qwen3 thinking | `enable_thinking=false`; explicit `<reason>` action reasoning |
| Environment horizon / history | 30 interactions /2 recent observation-action pairs |
| Prompt / response limits | MiniHack12,288 /2,048; ScienceWorld15,000 /512 |
| vLLM model length / token batch cap | 16,384 /16,384 |
| vLLM GPU memory utilization | 0.6 |
| Model memory options | Gradient checkpointing and remove-padding enabled; actor parameter/optimizer offload disabled; reference parameter offload enabled |
| Invalid action penalty | Enabled, coefficient0.1, applied by the trainer in addition to native task reward |
| Seed | 0 for data shuffle, rollout and environment; `--seed` overrides these, not the fixed test identities |
| Ray CPU reservation per environment worker | 0.1 logical CPU; each ScienceWorld slot also starts a JVM, so provision physical CPU/RAM accordingly |

| Method | Method-specific defaults |
|---|---|
| `oc_opsd` | beta0.5, epsilon`1e-6`, K_future2, top-step ratio0.2; future observations, no skill bank |
| `opsd` | SDAR coefficient0.01, gate5, PG loss coefficient0 |
| `grpo_opsd` | SDAR coefficient0.01, gate0 |
| `sdar` | SDAR coefficient0.01, gate5 |
| `rlsd` | lambda0.5, clip0.2, warmdown disabled (`-1`) |
| `skillsd` | SDL lambda0.001, warmdown disabled (`-1`) |
| `skill_grpo` | Category-matched skills in acting prompts; upstream with/without-skills validation |
| `grpo` / `rloo` | Corresponding upstream advantage estimator |
| `gigpo` | Step advantage weight1, `mean_std_norm` mode |
| `gigpo_dynamic` | Same GiGPO settings; group filtering, up to10 sampling attempts, clipping0.2/0.28 |
| `dapo` | GRPO advantage; group filtering, up to10 sampling attempts, clipping0.2/0.28 |
| `gspo` | GSPO loss, clipping0.0003/0.0004 |
| `ppo` | GAE and value-head critic; critic LR`1e-5`, critic micro-batch/GPU4;128 independent tasks |

The skill-using methods above default to `skill_all=false`. JSON common
mistakes are auxiliary data; the Markdown teacher path consumes general and
category workflow entries. `--dry-run` prints every final Hydra override and
does not allocate GPUs. Additional overrides follow `--`, for example:

```bash
python scripts/new_benchmarks/train.py --env scienceworld --method opsd \
  --model "$MODEL_PATH" --seed 42 --dry-run -- \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7
```

Checkpoint retention keeps the latest checkpoint, saves every30 updates, and
validates every5 updates. Run at least3 independent training seeds when
reporting comparisons; no multi-seed results are claimed here.

Evaluate a converted/exported model checkpoint on the fixed128-instance test
profile with:

```bash
python scripts/new_benchmarks/train.py --env scienceworld --method grpo \
  --model /path/to/exported-hf-checkpoint --output /path/to/test-results \
  --eval-split test --val-only
```

Test evaluation uses **temperature0 and `do_sample=False`**. ScienceWorld
success requires the final native score to equal100; lower scores are failures.
For an additional ScienceWorld evaluation on all1,661 held-out pairs, use
`--eval-split test_full --val-only` instead. Report that larger evaluation
separately from the paper's128-instance metric. Both test profiles require
`--val-only`; periodic development evaluation remains temperature0.4 with
sampling enabled.

Use the same skill-free evaluator for distilled methods; Skill-GRPO's own
entry point additionally reports with/without-skills validation. Set the
training seed through `--seed`; explicit additional Hydra overrides follow `--`.

### Validation status

The shipped inputs have49 SHA256-checked files (about6.9 MiB), including both
complete frozen banks and all7 prepared parquets. See
[the data release audit](validation/data_release.json). Native task/category
and prepared identity checks passed; this does not constitute a GPU training
result or a fresh installation test of the original OCSD environment.

Six CPU integration tests cover all28 training and28 final-test
method/environment Hydra configurations, real task
parquet generation, train/dev/test isolation, exact task-to-skill routing,
verbatim skill provenance, action parsing, and OCSD action schemas. Native
MiniHack reset/no-op/real win/terminal replay and native ScienceWorld reset,
step, task type and score-delta reward were exercised. See
`validation/new_benchmarks.json` for the recorded checks. An additional CPU
pipeline check uses the real Qwen3 tokenizer, two grouped native simulator
slots and a scripted policy to exercise the real rollout collector,0/1/2-step
future evidence, and teacher batching with unchanged response tokens. It
checks skill prefixes survive the configured prompt budget; it does not
perform a model forward pass or optimizer step.

[The quantity and matching audit](validation/counts_and_matching.json) also
checks the generated parquets, zero split overlap, and every ScienceWorld
category against the native simulator's task-name list:

```bash
PYTHONPATH=. python scripts/new_benchmarks/audit_counts.py \
  --native --prepared data/prepared --output /path/to/counts-audit.json
```

```bash
python -m unittest discover -s tests/new_benchmarks -v
PYTHONPATH=. python scripts/new_benchmarks/smoke_native.py
PYTHONPATH=. python scripts/new_benchmarks/smoke_ray.py --output /path/to/ray-check.json
PYTHONPATH=. python scripts/new_benchmarks/smoke_pipeline.py \
  --tokenizer /path/to/Qwen3-4B --output /path/to/pipeline-check.json
```

This is an implementation release. **A full8-GPU optimizer run and150-update
learning curves on the new environments have not been validated or reported.**
The original installation and benchmark instructions follow.

<p align="center">
  <img src="fig/methodology.png" alt="OCSD" width="100%">
</p>

## Requirements

- Linux + CUDA, Python 3.12, 8× A100/H100
- Java 17 for WebShop

## Installation

```bash
conda env create -f environment.yml
conda activate verl-agent
python -m pip install -e . --no-deps
```

Benchmark envs:

```bash
# ALFWorld
python -m pip install alfworld
alfworld-download --data-dir ./data/alfworld

# WebShop
git clone https://github.com/princeton-nlp/webshop.git && cd webshop
./setup.sh -d all

# Search-QA — follow Search-R1 (https://github.com/PeterGriffinJin/Search-R1)
# then launch the retriever
bash examples/search/retriever/retrieval_launch.sh
```

## Quick start

Every experiment is a single self-contained script under
`examples/<algo>_trainer/`. Edit the two placeholder paths at the top
(`MODEL_DIR`, `BASE`), then run:

```bash
bash examples/oc_opsd_trainer/run_alfworld_qwen3_4b.sh
```

Three tasks (ALFWorld / WebShop / Search) × three sizes (Qwen3-1.7B / 4B / 8B)
per method — 9 scripts each.

| Directory                     | Method    |
| ----------------------------- | --------- |
| `examples/oc_opsd_trainer/`   | OC-OPSD   |
| `examples/opsd_trainer/`      | OPSD      |
| `examples/grpo_opsd_trainer/` | GRPO+OPSD |
| `examples/sdar_trainer/`      | SDAR      |
| `examples/rlsd_trainer/`      | RLSD      |
| `examples/grpo_trainer/`      | GRPO      |

## OCSD hyperparameters

Passed via `+algorithm.oc_opsd.*`:

| Key              | Description                                          | Default |
| ---------------- | ---------------------------------------------------- | ------- |
| `beta`           | Calibration strength β on the advantage              | `0.5`   |
| `eps_A`          | Numerical floor on \|A_seq\| in the sign gate         | `1e-6`  |
| `top_step_ratio` | Fraction of highest-NLL steps kept in m_oc           | `0.2`   |
| `env_name`       | `alfworld` / `webshop` / `search`                    | –       |

## Acknowledgements

Built on top of [verl](https://github.com/volcengine/verl) (ByteDance) and
[SDAR](https://github.com/ZJU-REAL/SDAR) (ZJU-REAL). Benchmarks:
[ALFWorld](https://github.com/alfworld/alfworld),
[WebShop](https://github.com/princeton-nlp/webshop),
[Search-R1](https://github.com/PeterGriffinJin/Search-R1).

## License

Apache License 2.0.
