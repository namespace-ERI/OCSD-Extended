"""Shared launcher for the upstream algorithms; --dry-run prints the exact argv."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
METHODS = {'grpo': 'main_ppo', 'oc_opsd': 'main_oc_opsd', 'opsd': 'main_sdar',
           'grpo_opsd': 'main_sdar', 'sdar': 'main_sdar', 'rlsd': 'main_rlsd',
           'skillsd': 'main_skillsd', 'skill_grpo': 'main_skill_grpo',
           'gigpo': 'main_ppo', 'gigpo_dynamic': 'main_ppo', 'rloo': 'main_ppo',
           'dapo': 'main_ppo', 'gspo': 'main_ppo', 'ppo': 'main_ppo'}


def command(args):
    kind = args.env
    prepared = Path(args.prepared).resolve() / kind
    common = {
        'algorithm.adv_estimator': 'grpo', 'data.train_files': str(prepared / 'train.parquet'),
        'data.val_files': str(prepared / f'{args.eval_split}.parquet'),
        'data.train_batch_size': 16, 'data.val_batch_size': 128,
        'data.max_prompt_length': 12288 if kind == 'minihack_room' else 15000,
        'data.max_response_length': 2048 if kind == 'minihack_room' else 512,
        'data.filter_overlong_prompts': True, 'data.truncation': 'error', 'data.return_raw_chat': True,
        '++data.seed': args.seed,
        '++data.apply_chat_template_kwargs.enable_thinking': False,
        'actor_rollout_ref.model.path': args.model, 'actor_rollout_ref.actor.optim.lr': 1e-6,
        'actor_rollout_ref.model.use_remove_padding': True,
        'actor_rollout_ref.actor.ppo_mini_batch_size': 64,
        'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu': 8,
        'actor_rollout_ref.actor.use_kl_loss': True, 'actor_rollout_ref.actor.kl_loss_coef': .01,
        'actor_rollout_ref.actor.kl_loss_type': 'low_var_kl',
        'actor_rollout_ref.model.enable_gradient_checkpointing': True,
        'actor_rollout_ref.actor.fsdp_config.param_offload': False,
        'actor_rollout_ref.actor.fsdp_config.optimizer_offload': False,
        'actor_rollout_ref.rollout.name': args.engine,
        'actor_rollout_ref.rollout.tensor_model_parallel_size': 2,
        'actor_rollout_ref.rollout.gpu_memory_utilization': .6,
        '++actor_rollout_ref.rollout.seed': args.seed,
        'actor_rollout_ref.rollout.max_model_len': 16384,
        'actor_rollout_ref.rollout.max_num_batched_tokens': 16384,
        'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu': 8,
        'actor_rollout_ref.rollout.enable_chunked_prefill': False,
        'actor_rollout_ref.rollout.enforce_eager': False, 'actor_rollout_ref.rollout.free_cache_engine': False,
        'actor_rollout_ref.rollout.temperature': 1., 'actor_rollout_ref.rollout.top_p': 1.,
        'actor_rollout_ref.rollout.val_kwargs.temperature': .4,
        'actor_rollout_ref.rollout.val_kwargs.do_sample': True,
        'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu': 8,
        'actor_rollout_ref.ref.fsdp_config.param_offload': True,
        'actor_rollout_ref.actor.use_invalid_action_penalty': True,
        'actor_rollout_ref.actor.invalid_action_penalty_coef': .1,
        'algorithm.use_kl_in_reward': False,
        '++env.thinking_tag': 'reason', 'env.env_name': kind, 'env.seed': args.seed,
        'env.max_steps': 30, 'env.history_length': 2, 'env.rollout.n': 8,
        'env.resources_per_worker.num_cpus': .1, 'env.resources_per_worker.num_gpus': 0,
        'trainer.critic_warmup': 0, 'trainer.logger': "['console','tensorboard']",
        'trainer.project_name': f'ocsd_{kind}', 'trainer.experiment_name': args.method,
        'trainer.n_gpus_per_node': 8, 'trainer.nnodes': 1,
        'trainer.ray_wait_register_center_timeout': 600,
        'trainer.save_freq': 30, 'trainer.test_freq': 5,
        # The upstream placeholder parquet has16 rows, so150 epochs=150 updates.
        # Real task parquet has many rows: cap STEPS explicitly to keep that budget.
        'trainer.total_epochs': 150, 'trainer.total_training_steps': 150,
        'trainer.val_before_train': True, 'trainer.val_only': args.val_only,
        'trainer.max_actor_ckpt_to_keep': 1, 'trainer.max_critic_ckpt_to_keep': 1,
        'trainer.default_local_dir': str(Path(args.output).resolve()),
    }
    if kind == 'scienceworld':
        common.update({'++env.scienceworld.variation_manifest': str(ROOT / 'data/new_benchmarks/scienceworld.json'),
                       '++env.scienceworld.eval_split': args.eval_split, '++env.scienceworld.simplification': 'easy',
                       '++env.scienceworld.reward_mode': 'score_delta'})
    if args.method == 'oc_opsd':
        common.update({'++algorithm.oc_opsd.beta': .5, '++algorithm.oc_opsd.eps_A': 1e-6,
                       '++algorithm.oc_opsd.K_future': 2, '++algorithm.oc_opsd.top_step_ratio': .2,
                       '++algorithm.oc_opsd.env_name': kind})
    elif args.method in {'opsd', 'grpo_opsd', 'sdar'}:
        common.update({'++algorithm.sdar.sdar_coef': .01,
                       '++algorithm.sdar.gate_beta': 0. if args.method == 'grpo_opsd' else 5.,
                       '++algorithm.sdar.skills_dir': str(ROOT / 'skills' / kind),
                       '++algorithm.sdar.skill_all': False})
        if args.method == 'opsd':
            common['++actor_rollout_ref.actor.pg_loss_coef'] = 0.
    elif args.method == 'rlsd':
        common.update({'++algorithm.rlsd.rlsd_lambda': .5, '++algorithm.rlsd.warmdown_steps': -1,
                       '++algorithm.rlsd.clip_eps': .2, '++algorithm.rlsd.skills_dir': str(ROOT / 'skills' / kind),
                       '++algorithm.rlsd.skill_all': False})
    elif args.method == 'skillsd':
        common.update({'++algorithm.skillsd.sdl_lambda': .001, '++algorithm.skillsd.warmdown_steps': -1,
                       '++algorithm.skillsd.skills_dir': str(ROOT / 'skills' / kind),
                       '++algorithm.skillsd.skill_all': False})
    elif args.method == 'skill_grpo':
        common.update({'++algorithm.skill_grpo.skills_dir': str(ROOT / 'skills' / kind),
                       '++algorithm.skill_grpo.skill_all': False})
    elif args.method in {'gigpo', 'gigpo_dynamic'}:
        common.update({'algorithm.adv_estimator': 'gigpo', 'algorithm.gigpo.step_advantage_w': 1.,
                       'algorithm.gigpo.mode': 'mean_std_norm'})
    elif args.method == 'rloo':
        common['algorithm.adv_estimator'] = 'rloo'
    elif args.method == 'gspo':
        common.update({'actor_rollout_ref.actor.policy_loss.loss_mode': 'gspo',
                       'actor_rollout_ref.actor.clip_ratio_low': .0003,
                       'actor_rollout_ref.actor.clip_ratio_high': .0004})
    elif args.method == 'ppo':
        common.update({'algorithm.adv_estimator': 'gae', 'critic.model.path': args.model,
                       'critic.optim.lr': 1e-5, 'critic.model.use_remove_padding': True,
                       'critic.ppo_micro_batch_size_per_gpu': 4,
                       'data.train_batch_size': 128, 'env.rollout.n': 1})
    if args.method in {'dapo', 'gigpo_dynamic'}:
        common.update({'algorithm.filter_groups.enable': True,
                       'algorithm.filter_groups.max_num_gen_batches': 10,
                       'actor_rollout_ref.actor.clip_ratio_low': .2,
                       'actor_rollout_ref.actor.clip_ratio_high': .28})
    def value(v):
        return str(v).lower() if isinstance(v, bool) else str(v)
    return [sys.executable, '-m', 'verl.trainer.' + METHODS[args.method]] + [f'{k}={value(v)}' for k, v in common.items()] + args.overrides


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=['minihack_room', 'scienceworld'], required=True)
    p.add_argument('--method', choices=list(METHODS), required=True)
    p.add_argument('--model', default=os.getenv('MODEL_PATH', 'Qwen/Qwen3-4B'))
    p.add_argument('--prepared', default=str(ROOT / 'data/prepared'))
    p.add_argument('--output', default='outputs/new_benchmarks')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--engine', default='vllm')
    p.add_argument('--eval-split', choices=['dev', 'test'], default='dev')
    p.add_argument('--val-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('overrides', nargs=argparse.REMAINDER)
    return p


if __name__ == '__main__':
    a = parser().parse_args()
    if a.overrides[:1] == ['--']:
        a.overrides = a.overrides[1:]
    if a.eval_split == 'test' and not a.val_only:
        raise SystemExit('Use dev for training; test is only available with --val-only.')
    cmd = command(a)
    if a.dry_run:
        print(shlex.join(cmd))
    else:
        for split in ['train', a.eval_split]:
            if not (Path(a.prepared) / a.env / f'{split}.parquet').is_file():
                raise SystemExit('Run python scripts/new_benchmarks/prepare_data.py first')
        os.environ.setdefault('WANDB_MODE', 'disabled')
        os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))
