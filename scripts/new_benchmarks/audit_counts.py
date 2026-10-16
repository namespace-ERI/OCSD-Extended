"""Verify actual task counts, split isolation and full skill taxonomy coverage."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', action='store_true', help='Verify every ScienceWorld category against the native simulator')
    p.add_argument('--prepared', type=Path, help='Also verify generated parquet task identities')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = {}
    for kind in ['minihack_room', 'scienceworld']:
        data = json.loads((ROOT / f'data/new_benchmarks/{kind}.json').read_text())
        folder = ROOT / 'skills' / kind
        bank = json.loads((folder / 'claude_style_skills.json').read_text())
        mapping = json.loads((folder / 'skill_mapping.json').read_text())
        evidence = [json.loads(x) for x in (folder / 'evidence.jsonl').read_text().splitlines()]
        def canonical(v):
            return int(v) if kind == 'minihack_room' else tuple(v)
        pools = {s: set(map(canonical, data[s])) for s in ['train', 'dev', 'test']}
        assert all(len(data[s]) == len(pools[s]) for s in pools)
        assert not (pools['train'] & pools['dev'] or pools['train'] & pools['test'] or pools['dev'] & pools['test'])
        identities = []
        for row in evidence:
            assert row['episode_id'].endswith('/r00')
            identifier = row['episode_id'].split('/')[1]
            identity = int(identifier[4:]) if kind == 'minihack_room' else tuple(map(int, re.fullmatch(r't(\d+)_v(\d+)', identifier).groups()))
            identities.append(identity)
        assert len(identities) == len(set(identities)) and set(identities) == pools['train']
        categories = {e['category'] for e in evidence}
        assert categories == set(bank['task_specific_skills']) == set(mapping['task_to_skill'])
        assert all((folder / file).is_file() for file in mapping['skill_files'].values())
        info = {'task_counts': {s: len(pools[s]) for s in pools}, 'duplicates': 0, 'split_overlap': 0,
                'strategy_count': len(evidence), 'all_first_rollout': True,
                'general_skills': len(bank['general_skills']),
                'specific_skills': sum(map(len, bank['task_specific_skills'].values())),
                'mapped_categories': len(categories),
                'categories_without_success_skills': sum(not v for v in bank['task_specific_skills'].values())}
        if 'test_full' in data:
            full = set(map(canonical, data['test_full']))
            assert len(full) == len(data['test_full']) == 1661
            assert pools['test'] <= full and not (full & pools['train'] or full & pools['dev'])
            info['full_test_pool_size'] = len(full)
            info['selected_test_subset_in_full_pool'] = True
        info['paper_eval'] = data['metadata'].get('paper_eval')
        if args.prepared:
            import pyarrow.parquet as pq
            prepared_pools = dict(pools)
            if 'test_full' in data:
                prepared_pools['test_full'] = full
            for split, expected in prepared_pools.items():
                table = pq.read_table(args.prepared / kind / f'{split}.parquet')
                values = table['env_kwargs'].to_pylist()
                actual = [int(x['seed']) if kind == 'minihack_room' else (int(x['task_index']), int(x['variation_id'])) for x in values]
                assert len(actual) == len(expected) and set(actual) == expected
            info['parquet_identities_exact'] = True
        if args.native and kind == 'scienceworld':
            from scienceworld import ScienceWorldEnv
            native = ScienceWorldEnv(taskName=None, envStepLimit=30)
            try:
                names = list(native.get_task_names())
                assert {names[pair[0]] for pair in pools['train']} == categories
                for row, (task_index, _) in zip(evidence, identities):
                    assert names[task_index] == row['category']
                info['all_29_categories_match_native_task_names'] = True
            finally:
                native.close()
        report[kind] = info
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
