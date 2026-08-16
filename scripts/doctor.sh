#!/usr/bin/env bash
# doctor.sh - verify a DeepSeek-V4-Flash-0731 checkpoint directory is complete
# and self-consistent before running the engine.
#
#   scripts/doctor.sh MODEL_DIR
#
# Checks, in order:
#   1. the required metadata files exist (config.json, tokenizer.json,
#      tokenizer_config.json and generation_config.json);
#   2. all 48 shards are present with the exact expected byte size and SHA-256
#      (from scripts/dsv4_shard_sizes.txt);
#   3. if bin/dsv4 has been built, opens all shard headers and required global
#      tensor bindings with --validate-only.
#
# Prints `READY FOR FULL-CHECKPOINT TEST` on success.
set -euo pipefail

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "[doctor] need sha256sum or shasum for checkpoint verification" >&2
        return 127
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/dsv4_shard_sizes.txt"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 MODEL_DIR" >&2
    exit 2
fi
MODEL_DIR="$1"

fail() { echo "[doctor] FAIL: $*" >&2; exit 1; }

echo "[doctor] checking ${MODEL_DIR}"

# ------------------------------------------------------- metadata files ----
for f in config.json tokenizer.json tokenizer_config.json generation_config.json; do
    [ -f "${MODEL_DIR}/${f}" ] || fail "missing ${f}"
done
[ -s "${MODEL_DIR}/config.json" ] || fail "config.json is empty"
[ -s "${MODEL_DIR}/tokenizer.json" ] || fail "tokenizer.json is empty"
echo "[doctor] metadata files present"

# ------------------------------------------------------------ 48 shards ----
if [ ! -f "$MANIFEST" ]; then
    fail "manifest not found: $MANIFEST"
fi
NMANIFEST=$(grep -vc '^#' "$MANIFEST" || true)
[ "$NMANIFEST" -eq 48 ] || fail "manifest has ${NMANIFEST} entries, expected 48"

TOTAL=0
while read -r name expected sha; do
    [ -n "$name" ] || continue
    case "$name" in \#*) continue ;; esac
    TOTAL=$((TOTAL + expected))
    file="${MODEL_DIR}/${name}"
    [ -f "$file" ] || fail "missing shard ${name}"
    actual=$(wc -c < "$file")
    [ "$actual" -eq "$expected" ] || fail "shard ${name} size ${actual} != ${expected}"
done < "$MANIFEST"
[ "$TOTAL" -eq 166886535336 ] || fail "shard byte total ${TOTAL} != 166886535336"
echo "[doctor] 48 shards present with exact sizes (${TOTAL} bytes)"

# ---------------------------------------------------------- sha-256 -------
if [ "${DSV4_SHARDS_VERIFIED:-0}" = "1" ]; then
    echo "[doctor] SHA-256 already verified by the calling download process"
else
    echo "[doctor] verifying SHA-256 (this reads 166.9 GB; takes a few minutes)"
    FAIL=0
    while read -r name expected sha; do
        [ -n "$name" ] || continue
        case "$name" in \#*) continue ;; esac
        got=$(sha256_file "${MODEL_DIR}/${name}")
        if [ "$got" != "$sha" ]; then
            echo "[doctor] SHA-256 MISMATCH ${name}" >&2
            FAIL=1
        fi
    done < "$MANIFEST"
    [ "$FAIL" -eq 0 ] || fail "SHA-256 verification failed"
    echo "[doctor] all 48 shards SHA-256 verified"
fi

# ------------------------------------------------------ optional index ------
if [ -f "${MODEL_DIR}/model.safetensors.index.json" ]; then
    echo "[doctor] optional model.safetensors.index.json present"
else
    echo "[doctor] optional model.safetensors.index.json absent; shard headers are authoritative"
fi

# --------------------------------------------------------- validate-only ----
if [ -x "${SCRIPT_DIR}/../bin/dsv4" ]; then
    echo "[doctor] running bin/dsv4 --validate-only"
    "${SCRIPT_DIR}/../bin/dsv4" --model "$MODEL_DIR" --validate-only \
        || fail "dsv4 --validate-only failed"
fi

echo
echo "[doctor] READY FOR FULL-CHECKPOINT TEST"
