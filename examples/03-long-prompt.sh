#!/usr/bin/env bash
# 03-long-prompt.sh - generate with a long context, exercising the ratio-4 and
# ratio-128 compressors across many block boundaries.
#
#   examples/03-long-prompt.sh "Summarize the following document: ..."
#
# The context is raised to 8192; compression still keeps the memory plan
# bounded because the compressed history is a fraction of the window size.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT_DIR}/bin/dsv4"
MODEL="${DSV4_MODEL_DIR:-${ROOT_DIR}/model/DeepSeek-V4-Flash-0731}"

if [ ! -x "$BIN" ]; then echo "build bin/dsv4 first (make -j)" >&2; exit 1; fi

exec "$BIN" --model "$MODEL" --prompt "${1:-Summarize the following document:}" \
     --max-tokens 64 --context 8192 --memory-gib 6
