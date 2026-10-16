"""Create real task-identity parquet rows without downloading placeholder datasets."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(kind, split):
    manifest = json.loads((ROOT / 'data/new_benchmarks' / f'{kind}.json').read_text())
    result = []
    for i, identity in enumerate(manifest[split]):
        kwargs = {'seed': identity} if kind == 'minihack_room' else dict(task_index=identity[0], variation_id=identity[1])
        # Actual goal and native category are populated by reset, never supplied by a judge.
        result.append({'data_source': kind, 'prompt': [{'role': 'user', 'content': 'Interact with the environment.'}],
                       'ability': 'agent', 'env_kwargs': kwargs,
                       'extra_info': {'split': split, 'index': i}})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT / 'data/prepared')
    args = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    for kind in ['minihack_room', 'scienceworld']:
        folder = args.output / kind
        folder.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((ROOT / 'data/new_benchmarks' / f'{kind}.json').read_text())
        for split in ['train', 'dev', 'test'] + (['test_full'] if 'test_full' in manifest else []):
            entries = rows(kind, split)
            pq.write_table(pa.Table.from_pylist(entries), folder / f'{split}.parquet')
            print(kind, split, len(entries))


if __name__ == '__main__':
    main()
