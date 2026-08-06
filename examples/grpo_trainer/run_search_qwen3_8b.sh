set -x

GPU_NUM=8
ENGINE=vllm

train_data_size=128
val_data_size=512
group_size=8           # must be <= GPU_NUM

# Training config
total_epochs=999
total_training_steps=150
save_freq=140
test_freq=150

experiment_name="grpo_qwen3_8b_search_baseline"

BASE=${MODEL_DIR:-/path/to/models}/sdar
OUTPUT_BASE=${HOME}/models/sdar
CHECKPOINT_DIR="${OUTPUT_BASE}/checkpoints"
export MODEL_PATH=${MODEL_DIR:-/path/to/pretrained_models}/Qwen3-8B
export WANDB_API_KEY=your_key_here
export WANDB_MODE=disabled
export HIGHLIGHT_CONFIGS='<search>:0,0,255;</search>:0,0,255;<information>:255,0,0;</information>:255,0,0'

RUN_DATE=$(date +%Y%m%d_%H%M%S)

echo "Vanilla GRPO Baseline - 8-GPU Workbench (Search-QA)"

cur_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SKILLRL_ROOT="$(dirname "$(dirname "$cur_dir")")"

cd "$SKILLRL_ROOT" || exit 1

TRAIN_DATA="${BASE}/data/searchR1_processed_direct/train.parquet"
VAL_DATA="${BASE}/data/searchR1_processed_direct/test.parquet"

SEARCH_URLS_FILE="${SEARCH_URLS_FILE:-$SKILLRL_ROOT/logs/retrieval/search_urls_spare1.txt}"
SEARCH_URLS_WAIT_TIMEOUT="${SEARCH_URLS_WAIT_TIMEOUT:-300}"
if [[ -n "${SEARCH_URLS:-}" ]]; then
    echo "[SearchURLs] Using explicitly provided SEARCH_URLS env, skip file wait."
else
    echo "[SearchURLs] SEARCH_URLS not set, waiting for ${SEARCH_URLS_FILE} (timeout ${SEARCH_URLS_WAIT_TIMEOUT}s)..."
    WAITED=0
    while [[ ! -f "${SEARCH_URLS_FILE}" ]]; do
        if (( WAITED >= SEARCH_URLS_WAIT_TIMEOUT )); then
            echo "[SearchURLs] ERROR: ${SEARCH_URLS_FILE} not found after ${SEARCH_URLS_WAIT_TIMEOUT}s."
            echo "[SearchURLs] Please submit retrieval job first:"
            exit 1
        fi
        sleep 1
        WAITED=$((WAITED + 1))
    done
    SEARCH_URLS="$(cat "${SEARCH_URLS_FILE}")"
    echo "[SearchURLs] Loaded from ${SEARCH_URLS_FILE} (waited ~${WAITED}s): ${SEARCH_URLS}"
fi

# Convert to OmegaConf list syntax [url1,url2] for verl
# envs.py rotates URLs by env index (round-robin)
SEARCH_URLS_YAML="[$( echo "${SEARCH_URLS}" | tr ',' ' ' | tr -s ' ' | sed 's/^ *//;s/ *$//' | tr ' ' ',' )]"
echo "[SearchURLs] Rendered YAML list: ${SEARCH_URLS_YAML}"


python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    +env.thinking_tag=reason \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    env.search.timeout=60 \
    env.search.search_url=$SEARCH_URLS_YAML \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_agent_search_GRPO' \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${GPU_NUM} \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir=${CHECKPOINT_DIR}/searchqa/${experiment_name}/${RUN_DATE} \
    trainer.val_before_train=False
