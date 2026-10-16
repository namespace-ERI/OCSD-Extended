"""Align test profiles with CompeteRL Appendix B; preserve the training protocol."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def scienceworld_selection(pool, size=128, seed=1):
    """Author preprocessing sha256_pair_rank_v1 (fixed identities, not a batch sample)."""
    ranked = sorted((hashlib.sha256(f'sha256_pair_rank_v1:{seed}:{int(t)}:{int(v)}'.encode()).digest(),
                     int(t), int(v)) for t, v in pool)
    return [[t, v] for _, t, v in ranked[:size]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--minihack-test-count', type=int, choices=[64, 128], default=128)
    args = p.parse_args()
    folder = ROOT / 'data/new_benchmarks'
    mh_path = folder / 'minihack_room.json'
    mh = json.loads(mh_path.read_text())
    mh['test'] = list(range(2**30, 2**30 + args.minihack_test_count))
    assert not (set(mh['test']) & set(mh['dev']) or set(mh['test']) & set(mh['train']))
    mh['metadata']['protocol'] = f'fixed_train1000_dev128_test{args.minihack_test_count}_author_code'
    mh['metadata']['paper_eval'] = {
        'source': 'CompeteRL Appendix B.1 p14 and B.2 Table4 p15',
        'paper_stated': {'environment': 'MiniHack-Room-Ultimate-5x5-v0', 'fixed_heldout_seeds': True,
                         'max_steps': 30, 'eval_temperature': 0.0},
        'seed_start': 2**30, 'test_count': args.minihack_test_count,
        'count_basis': 'author preprocessing default128; author training launcher default64',
        'paper_exact_seed_manifest_verified': False,
        'limitation': 'PDF does not state seed count or list; this is an explicit author-code-compatible choice.'}
    mh_path.write_text(json.dumps(mh, indent=2) + '\n')
    sw_path = folder / 'scienceworld.json'
    sw = json.loads(sw_path.read_text())
    pool = sw.get('test_full', sw['test'])
    assert len(pool) == 1661
    sw['test_full'] = pool
    sw['test'] = scienceworld_selection(pool)
    gold = json.loads((folder / 'scienceworld_paper_eval_selection.json').read_text())
    assert sw['test'] == [[x['task_index'], x['variation_id']] for x in gold['instances']]
    sw['metadata']['paper_eval'] = {
        'source': 'CompeteRL Appendix B.1 pp14-15 and B.2 Table4 p15',
        'test_pool_size': 1661, 'selected_test_size': 128, 'eval_seed': 1,
        'selection_algorithm': 'sha256_pair_rank_v1', 'max_steps': 30,
        'eval_temperature': 0.0, 'success': 'final native score exactly100',
        'selection_matches_author_preprocessing': True,
        'selection_manifest': 'scienceworld_paper_eval_selection.json'}
    sw['metadata']['native_test_unchanged'] = False
    sw['metadata']['native_test_pool_unchanged'] = True
    sw_path.write_text(json.dumps(sw, indent=2) + '\n')
    print('Test profiles updated; training/dev identities and training budget unchanged.')


if __name__ == '__main__':
    main()
