"""Exercise the real rollout collector and teacher batching with a scripted policy.

Requires a local Qwen3 tokenizer and native simulators, but no model weights are
loaded and no optimizer update is performed. Two native slots share each task.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer
    from hydra import compose, initialize_config_dir
    from verl import DataProto
    from agent_system.multi_turn_rollout.rollout_loop import TrajectoryCollector
    from agent_system.environments.new_benchmarks.manager import NewBenchmarkManager
    from verl.trainer.ppo.rlsd_ray_trainer import build_teacher_batch
    from verl.trainer.ppo.rlsd_utils import SkillProvider
    from agent_system.environments.env_package.minihack_room.envs import MiniHackRoomWorker
    from agent_system.environments.env_package.minihack_room.projection import minihack_room_projection
    from agent_system.environments.env_package.scienceworld.envs import ScienceWorldWorker
    from agent_system.environments.env_package.scienceworld.projection import scienceworld_projection
    spec = importlib.util.spec_from_file_location('launch_extension', ROOT / 'scripts/new_benchmarks/train.py')
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    results = {}

    class ScriptedPolicy:
        world_size = 1
        def __init__(self, kind):
            self.kind, self.turn = kind, 0
        def generate_sequences(self, batch):
            action = '<answer>invalid</answer>' if self.turn < 2 else '<answer>northeast</answer>'
            if self.kind == 'scienceworld':
                action = '<action>look around</action>'
            text = '<reason>Scripted interface check.</reason>' + action
            ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
            response = torch.full((len(batch), 64), tokenizer.pad_token_id, dtype=torch.long)
            response[:, :len(ids)] = ids
            prompt = batch.batch['input_ids']
            response_mask = torch.zeros_like(response)
            response_mask[:, :len(ids)] = 1
            mask = torch.cat([batch.batch['attention_mask'], response_mask], -1)
            full = torch.cat([prompt, response], -1)
            self.turn += 1
            return DataProto.from_single_dict({'input_ids': full, 'responses': response,
                'attention_mask': mask, 'position_ids': (mask.cumsum(-1) - 1).clamp(min=0)})

    class LocalVector:
        def __init__(self, workers, kind):
            self.workers, self.kind = workers, kind
        def reset(self, seeds=None, variation_indices=None):
            identities = seeds if seeds is not None else [0] * len(variation_indices)
            values = [w.reset(i) for w, i in zip(self.workers, identities)]
            return list(map(list, zip(*values)))
        def step(self, actions):
            return list(map(list, zip(*(w.step(a) for w, a in zip(self.workers, actions)))))
        def close(self):
            for w in self.workers:
                w.close()

    for kind in ['minihack_room', 'scienceworld']:
        opts = launch.parser().parse_args(['--env', kind, '--method', 'sdar', '--model', args.tokenizer])
        with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
            cfg = compose(config_name='ppo_trainer', overrides=launch.command(opts)[3:])
        cfg.env.max_steps, cfg.env.rollout.n = 3, 2
        if kind == 'minihack_room':
            workers = [MiniHackRoomWorker('MiniHack-Room-Ultimate-5x5-v0', {'max_episode_steps': 30}) for _ in range(2)]
            identity, projection = {'seed': 1042419699}, minihack_room_projection
        else:
            pair = json.loads((ROOT / 'data/new_benchmarks/scienceworld.json').read_text())['train'][0]
            workers = [ScienceWorldWorker([pair], {'env_step_limit': 30, 'simplification': 'easy', 'reward_mode': 'score_delta'}) for _ in range(2)]
            identity, projection = {'task_index': pair[0], 'variation_id': pair[1]}, scienceworld_projection
        vector = LocalVector(workers, kind)
        manager = NewBenchmarkManager(vector, projection, cfg, kind)
        collector = TrajectoryCollector(cfg, tokenizer)
        # Collector expands one task to the two grouped native slots itself.
        batch = DataProto.from_single_dict({'input_ids': torch.ones((1, 1), dtype=torch.long),
            'attention_mask': torch.ones((1, 1), dtype=torch.long), 'position_ids': torch.zeros((1, 1), dtype=torch.long),
            'raw_prompt': np.array(['unused'], dtype=object),
            'data_source': np.array([kind], dtype=object),
            'env_kwargs': np.array([dict(identity)], dtype=object)})
        try:
            output = collector.multi_turn_loop(batch, ScriptedPolicy(kind), manager)
            assert len(output) == 6
            assert set(output.non_tensor_batch['benchmark']) == {kind}
            categories = set(output.non_tensor_batch['task_type'])
            assert len(categories) == 1
            assert set(output.non_tensor_batch['available_future_len']) == {0, 1, 2}
            provider = SkillProvider(str(ROOT / 'skills' / kind))
            teacher = build_teacher_batch(output, provider, tokenizer, cfg.data.max_prompt_length)
            assert torch.equal(teacher.batch['responses'], output.batch['responses'])
            decoded = tokenizer.decode(teacher.batch['input_ids'][0], skip_special_tokens=False)
            assert '[Privileged Skill Information]' in decoded
            assert provider.get_privileged_info_from_task_type(next(iter(categories))).splitlines()[0] in decoded
            results[kind] = {'native_slots': 2, 'turn_samples': len(output), 'task_type': list(categories)[0],
                             'future_lengths': [0, 1, 2], 'teacher_skill_prefix_retained': True,
                             'teacher_response_tokens_unchanged': True}
        finally:
            manager.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
