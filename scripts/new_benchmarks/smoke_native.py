"""Real native reset/step checks, without model weights or a GPU allocation."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    from agent_system.environments.env_package.minihack_room.envs import MiniHackRoomWorker
    from agent_system.environments.env_package.scienceworld.envs import ScienceWorldWorker
    reports = {}
    mh = MiniHackRoomWorker('MiniHack-Room-Ultimate-5x5-v0', {'max_episode_steps': 30})
    try:
        obs, info = mh.reset(1042419699)
        again, _ = mh.reset(1042419699)
        assert obs == again
        unchanged, reward, done, _ = mh.step(-1)
        assert unchanged == again and reward == 0 and not done
        end, reward, done, info = mh.step(4)
        assert done and info['won'] and reward == 1
        _, reward2, done2, _ = mh.step(4)
        assert reward2 == 0 and done2
        reports['minihack_room'] = {'deterministic_reset': True, 'invalid_noop': True,
                                   'real_success': True, 'terminal_replay_no_duplicate_reward': True}
    finally:
        mh.close()
    data = json.loads((ROOT / 'data/new_benchmarks/scienceworld.json').read_text())
    sw = ScienceWorldWorker(data['train'], {'env_step_limit': 30, 'simplification': 'easy', 'reward_mode': 'score_delta'})
    try:
        obs, info = sw.reset(0)
        task = info['task_name']
        _, reward, done, _ = sw.step('')
        assert reward == 0 and not done
        _, reward, done, info = sw.step('look around')
        assert info['task_name'] == task and isinstance(info['score'], float)
        assert abs(reward - info['score_delta'] / 100) < 1e-8
        reports['scienceworld'] = {'real_reset_and_step': True, 'native_task_type': task,
                                   'invalid_noop': True, 'score_delta_reward': True}
    finally:
        sw.close()
    text = json.dumps(reports, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
