#!/usr/bin/env bash
# download-dsv4-one.sh - download a single shard of the fixed ModelScope revision
# with resumable, size-checked transfers. Used both by download-dsv4.sh workers
# and directly for re-fetching one damaged shard.
#
#   scripts/download-dsv4-one.sh MODEL_DIR model-00007-of-00048.safetensors
#
# Reads scripts/dsv4_shard_sizes.txt for the expected size/SHA. Resumes from the
# current on-disk length; every curl failure restarts a fresh `curl -C -`
# process; a file larger than expected is rejected outright.
set -euo pipefail

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "[dsv4-one] need sha256sum or shasum for checkpoint verification" >&2
        return 127
    fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REV=f981a343464c25f82b901e5882716b3b2fa514de
BASE="https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/${REV}"
MANIFEST="${SCRIPT_DIR}/dsv4_shard_sizes.txt"

if [ "$#" -lt 2 ]; then
    echo "usage: $0 MODEL_DIR SHARD_NAME" >&2
    exit 2
fi
MODEL_DIR="$1"
NAME="$2"
FILE="${MODEL_DIR}/${NAME}"

line=$(awk -v n="$NAME" '$1 == n {print $2, $3}' "$MANIFEST")
[ -n "$line" ] || { echo "[dsv4-one] $NAME not in manifest" >&2; exit 1; }
EXPECTED=${line%% *}
SHA=${line##* }
mkdir -p "$MODEL_DIR"

STALL=0
while :; do
    if [ -f "$FILE" ]; then
        CUR=$(wc -c < "$FILE")
    else
        CUR=0
    fi
    if [ "$CUR" -gt "$EXPECTED" ]; then
        echo "[dsv4-one] $NAME on-disk $CUR exceeds expected $EXPECTED; aborting" >&2
        exit 1
    fi
    if [ "$CUR" -eq "$EXPECTED" ]; then
        GOT=$(sha256_file "$FILE")
        if [ "$GOT" = "$SHA" ]; then
            echo "[dsv4-one] $NAME complete and verified"
            exit 0
        fi
        echo "[dsv4-one] $NAME complete size but SHA mismatch ($GOT != $SHA); re-downloading" >&2
        # Corrupt cache entry: reset it so the resume offset restarts from 0.
        : > "$FILE"
        continue
    fi
    PREV="$CUR"
    if ! curl -sfL --connect-timeout 30 --max-time 3600 -C - -o "$FILE" "${BASE}/${NAME}"; then
        echo "[dsv4-one] $NAME curl failed at $CUR bytes; retrying" >&2
        sleep 1
        continue
    fi
    CUR=$(wc -c < "$FILE")
    if [ "$CUR" -eq "$PREV" ] && [ "$CUR" -lt "$EXPECTED" ]; then
        STALL=$((STALL + 1))
        [ "$STALL" -ge 5 ] && { echo "[dsv4-one] $NAME stalled; giving up" >&2; exit 1; }
    else
        STALL=0
    fi
done
