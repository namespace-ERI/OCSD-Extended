#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/new_benchmarks/train.py" --env scienceworld --method ppo --model "${MODEL_PATH:-Qwen/Qwen3-4B}" --output "${OUTPUT_DIR:-$ROOT/outputs/scienceworld/ppo}" "$@"
