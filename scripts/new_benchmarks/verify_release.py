"""Check the shipped data/skill bytes using only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'data/new_benchmarks/release_manifest.json').read_text())
    failures = []
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(f"Missing or invalid path: {item['path']}")
        elif path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            failures.append(f"Content differs from release: {item['path']}")
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f"Verified {len(manifest['files'])} data/skill files ({manifest['total_bytes']:,} bytes).")
    print('This checks release integrity; it does not run a simulator or train a model.')


if __name__ == '__main__':
    main()
