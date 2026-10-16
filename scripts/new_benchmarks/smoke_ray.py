"""Test make_envs and real Ray-native worker grouping without GPU/model inference."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import ray
    from hydra import compose, initialize_config_dir
    from agent_system.environments import make_envs
    spec = importlib.util.spec_from_file_location('launch_extension', ROOT / 'scripts/new_benchmarks/train.py')
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    temp = tempfile.mkdtemp(prefix='ocsd-ray-')
    reports = {}
    try:
        ray.init(num_cpus=2, include_dashboard=False, object_store_memory=128 * 1024**2, _temp_dir=temp)
        for kind in ['minihack_room', 'scienceworld']:
            parsed = launch.parser().parse_args(['--env', kind, '--method', 'grpo'])
            with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
                cfg = compose(config_name='ppo_trainer', overrides=launch.command(parsed)[3:])
            cfg.data.train_batch_size, cfg.data.val_batch_size, cfg.env.rollout.n = 1, 1, 2
            train, dev = make_envs(cfg)
            manifest = json.loads((ROOT / f'data/new_benchmarks/{kind}.json').read_text())
            def identity(split):
                x = manifest[split][0]
                return {'seed': x} if kind == 'minihack_room' else {'task_index': x[0], 'variation_id': x[1]}
            try:
                obs, infos = train.reset([identity('train')] * 2)
                assert len(obs['text']) == 2 and obs['anchor'][0] == obs['anchor'][1]
                action = '<reason>check</reason><answer>north</answer>' if kind == 'minihack_room' else '<reason>check</reason><action>look around</action>'
                _, rewards, done, infos = train.step([action] * 2)
                assert len(rewards) == 2 and all(i['is_action_valid'] for i in infos)
                if kind == 'minihack_room':
                    assert train.memory[0][-1]['action'] == 'north'
                dev_obs, _ = dev.reset([identity('dev')])
                assert len(dev_obs['text']) == 1
                reports[kind] = {'factory': True, 'grouped_native_workers': 2, 'identical_group_reset': True,
                                 'real_step': True, 'heldout_reset': True}
            finally:
                train.close()
                dev.close()
    finally:
        ray.shutdown()
        shutil.rmtree(temp, ignore_errors=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2) + '\n')
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
