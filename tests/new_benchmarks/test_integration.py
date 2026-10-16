"""CPU integration checks for split identity, skills, schemas and all launch configs."""
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Integration(unittest.TestCase):
    def test_original_skill_routes(self):
        from verl.trainer.ppo.rlsd_utils import SkillProvider
        alf = SkillProvider(str(ROOT / 'skills/alfworld'))
        self.assertIn(alf.skill_contents['heat'], alf.get_privileged_info('pick_heat_then_place_in_recep/game.tw-pddl'))
        web = SkillProvider(str(ROOT / 'skills/webshop'))
        self.assertIn(web.skill_contents['apparel'], web.get_privileged_info_from_prompt('Find a cotton shirt'))

    def test_splits_and_rows(self):
        prepare = module('prepare_extension', 'scripts/new_benchmarks/prepare_data.py')
        for kind, sizes in [('minihack_room', [1000, 128, 500]), ('scienceworld', [3194, 128, 1661])]:
            d = json.loads((ROOT / f'data/new_benchmarks/{kind}.json').read_text())
            sets = [set(map(str, d[s])) for s in ['train', 'dev', 'test']]
            self.assertEqual(list(map(len, sets)), sizes)
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            for split in ['train', 'dev', 'test']:
                rows = prepare.rows(kind, split)
                self.assertEqual(len(rows), len(d[split]))
                self.assertIn('env_kwargs', rows[0])
                self.assertNotIn('ground_truth', rows[0])

    def test_skill_evidence_and_exact_routing(self):
        from verl.trainer.ppo.rlsd_utils import SkillProvider
        for kind in ['minihack_room', 'scienceworld']:
            folder = ROOT / 'skills' / kind
            bank = json.loads((folder / 'claude_style_skills.json').read_text())
            evidence = {x['episode_id']: x for x in map(json.loads, (folder / 'evidence.jsonl').read_text().splitlines())}
            self.assertEqual(len(evidence), 1000 if kind == 'minihack_room' else 3194)
            self.assertTrue(all(x.endswith('/r00') for x in evidence))
            allowed = json.loads((ROOT / f'data/new_benchmarks/{kind}.json').read_text())['train']
            if kind == 'scienceworld':
                self.assertEqual({tuple(map(int, x.split('/')[1][1:].split('_v'))) for x in evidence}, set(map(tuple, allowed)))
            self.assertLessEqual(len(bank['general_skills']), 15)
            provider = SkillProvider(str(folder))
            for category, skills in bank['task_specific_skills'].items():
                self.assertLessEqual(len(skills), 6)
                text = provider.get_privileged_info_from_task_type(category)
                for skill in skills:
                    self.assertIn(skill['principle'], text)
                    for identity in skill['evidence_episode_ids']:
                        self.assertTrue(evidence[identity]['success'])
                        self.assertEqual(evidence[identity]['category'], category)
                        self.assertEqual(evidence[identity]['strategic_guidelines']['planning_pattern'], skill['principle'])
            self.assertEqual(provider.get_privileged_info_from_task_type('unknown'), provider.skill_contents['general_skills'])
            self.assertNotEqual(provider.get_privileged_info_from_task_type('room_navigation' if kind == 'minihack_room' else 'measure-melting-point-known-substance'), '')

    def test_action_parsers_and_schema(self):
        from agent_system.environments.env_package.scienceworld.projection import scienceworld_projection
        from agent_system.environments.env_package.minihack_room.projection import minihack_room_projection
        from verl.trainer.ppo.oc_opsd_utils import extract_action_text, build_action_schema
        self.assertEqual(scienceworld_projection(['<reason>check</reason><action>look around</action>']), (['look around'], [1]))
        self.assertEqual(scienceworld_projection(['<reason>x</reason><action>a</action><action>b</action>'])[1], [0])
        self.assertEqual(minihack_room_projection(['<answer>north-east</answer>', 'bad']), ([4, -1], [1, 0]))
        self.assertEqual(build_action_schema(extract_action_text('<answer>north</answer>'), 'minihack_room'), 'move north')
        schema = build_action_schema('focus on blue box', 'scienceworld')
        self.assertEqual(schema, 'focus on a target')
        self.assertNotIn('blue', schema)

    def test_all_hydra_configs(self):
        from hydra import compose, initialize_config_dir
        train = module('launch_extension', 'scripts/new_benchmarks/train.py')
        for kind in ['minihack_room', 'scienceworld']:
            for method in train.METHODS:
                a = train.parser().parse_args(['--env', kind, '--method', method])
                argv = train.command(a)
                with initialize_config_dir(config_dir=str(ROOT / 'verl/trainer/config'), version_base=None):
                    cfg = compose(config_name='ppo_trainer', overrides=argv[3:])
                self.assertEqual(cfg.env.rollout.n, 1 if method == 'ppo' else 8)
                self.assertEqual(cfg.data.train_batch_size, 128 if method == 'ppo' else 16)
                self.assertEqual(cfg.data.train_batch_size * cfg.env.rollout.n, 128)
                self.assertEqual(cfg.trainer.total_training_steps, 150)
                self.assertEqual(cfg.env.max_steps, 30)
                self.assertFalse(cfg.data.apply_chat_template_kwargs.enable_thinking)
                if method == 'opsd':
                    self.assertEqual(cfg.actor_rollout_ref.actor.pg_loss_coef, 0)


if __name__ == '__main__':
    unittest.main()
