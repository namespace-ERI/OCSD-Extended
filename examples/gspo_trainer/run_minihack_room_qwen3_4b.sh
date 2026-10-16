#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$ROOT/scripts/new_benchmarks/train.py" --env minihack_room --method gspo --model "${MODEL_PATH:-Qwen/Qwen3-4B}" --output "${OUTPUT_DIR:-$ROOT/outputs/minihack_room/gspo}" "$@"
