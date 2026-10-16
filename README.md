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

### Data, splits, and training budget

Checked-in JSON manifests specify actual task identities. Preparation is offline
and does not download the unrelated Geometry3k placeholder dataset used by the
old preparation script:

```bash
python scripts/new_benchmarks/prepare_data.py
```

| Added environment | Training pool | Development | Final test | Horizon | Prompt / response tokens |
|---|---:|---:|---:|---:|---:|
| MiniHack-Room-Ultimate-5x5-v0 | 1,000 fixed seeds | 128 fixed seeds | 500 fixed seeds | 30 | 12,288 / 2,048 |
| ScienceWorld BEACON L0,29 task families | 3,194 task/variation pairs | 128 pairs | 1,661 pairs | 30 | 15,000 / 512 |

MiniHack training seeds are the same1,000 used for strategy extraction. Dev/test
use distinct seeds in `[2**30, 2**31-1)`, deterministically sampled with
seed20261016; none overlaps the training pool. The initial NLE engine seeds,
including level-generator seeds and fixed moon phase, are explicitly set.

The source ScienceWorld L0 protocol contains3,322 train and1,661 test pairs and
no dev split. This extension leaves **all1,661 test pairs unchanged** and holds
out128 from the original training pool: up to4 per task family followed by a
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
validation-batch scale, and500 test is an extension choice following the scale
of our prior full WebShop evaluation. The128/500 split is not claimed to be
an official MiniHack benchmark split. ScienceWorld's3,322/1,661 source pools
come from the BEACON L0 protocol, with the documented128-train-instance holdout.

Importantly, `trainer.total_training_steps=150` is explicit. The old16-row
placeholder parquet made150 epochs equal150 updates; using the full task pool
with only an epoch limit would accidentally multiply the budget. The new
parquets contain one real identity per task, with shuffled training batches;
each identity is repeated8 times for grouped rollout (once for PPO). Validation uses one
rollout per identity and a fixed dev set; final test uses the full test parquet,
including the final partial batch. Training defaults reject selecting test as
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
Method defaults follow the corresponding upstream scripts: OCSD beta0.5,
K_future2 and top-step ratio0.2; SDAR coefficient0.01/gate5; GRPO+OPSD gate0;
OPSD disables the PG term; RLSD lambda0.5/clip0.2; SkillSD lambda0.001.
Checkpoint retention keeps the latest checkpoint, saves every30 updates, and
validates every5 updates. Run at least3 independent training seeds when
reporting comparisons; no multi-seed results are claimed here.

Evaluate a converted/exported model checkpoint on the entire test set with:

```bash
python scripts/new_benchmarks/train.py --env scienceworld --method grpo \
  --model /path/to/exported-hf-checkpoint --output /path/to/test-results \
  --eval-split test --val-only
```

Use the same skill-free evaluator for distilled methods; Skill-GRPO's own
entry point additionally reports with/without-skills validation. Evaluation
temperature defaults to0.4 and sampling is enabled, matching upstream. Set
training seed through `--seed`; explicit additional Hydra overrides follow `--`.

### Validation status

CPU validation covers all28 method/environment Hydra configurations, real task
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
