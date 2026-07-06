#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/llama8b_ruler4k_stage1.yaml}
python -m primettt.stage1_oracles \
  --data "$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["data"])' "$CONFIG")" \
  --model "$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["model"])' "$CONFIG")" \
  --out "$(python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["out"])' "$CONFIG")"

