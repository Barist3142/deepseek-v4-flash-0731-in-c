#!/usr/bin/env bash
# 02-memory-budgets.sh - show that the same prompt produces the same tokens at
# two very different cache budgets (0.5 GiB vs 4 GiB expert cache), proving the
# engine is not silently dropping experts when memory is tight.
#
#   examples/02-memory-budgets.sh "科技的边界在哪里？"
#
# The generated token IDs must match; wall time differs, cache counters differ,
# output must not.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${ROOT_DIR}/bin/dsv4"
MODEL="${DSV4_MODEL_DIR:-${ROOT_DIR}/model/DeepSeek-V4-Flash-0731}"
PROMPT="${1:-科技的边界在哪里？}"

if [ ! -x "$BIN" ]; then echo "build bin/dsv4 first (make -j)" >&2; exit 1; fi

echo "== 0.5 GiB expert cache =="
"$BIN" --model "$MODEL" --prompt "$PROMPT" --max-tokens 16 --cache-gib 0.5 \
    2>/tmp/dsv4_ids_small.txt | tee /tmp/dsv4_out_small.txt
echo
echo "== 4 GiB expert cache =="
"$BIN" --model "$MODEL" --prompt "$PROMPT" --max-tokens 16 --cache-gib 4 \
    2>/tmp/dsv4_ids_big.txt | tee /tmp/dsv4_out_big.txt

echo
small=$(grep 'generated ids:' /tmp/dsv4_ids_small.txt)
big=$(grep 'generated ids:' /tmp/dsv4_ids_big.txt)
if [ "$small" = "$big" ]; then
    echo "OK: identical token IDs at both budgets"
else
    echo "FAIL: token IDs differ between budgets" >&2
    echo "  small: $small" >&2
    echo "  big:   $big" >&2
    exit 1
fi
