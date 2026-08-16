#!/usr/bin/env bash
# download-dsv4.sh - fetch the fixed DeepSeek-V4-Flash-0731 revision from ModelScope.
#
#   scripts/download-dsv4.sh MODEL_DIR
#
# Fetches the fixed revision f981a343464c25f82b901e5882716b3b2fa514de, never a
# moving branch. Small JSON files are downloaded fresh every time; the 48 large
# shards are fetched with a per-file resume worker, DSV4_DOWNLOAD_JOBS at a time
# (default 4). A worker re-reads the on-disk length after every curl failure and
# starts a NEW `curl -C -` process from that offset; it never combines --retry
# with a stale resume offset. Files larger than their manifest size are rejected.
#
# Requirements: curl and either sha256sum (Linux) or shasum (macOS). Needs
# ~172 GB free on the destination filesystem.
set -euo pipefail

sha256_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        echo "[dsv4] need sha256sum or shasum for checkpoint verification" >&2
        return 127
    fi
}

REV=f981a343464c25f82b901e5882716b3b2fa514de
BASE="https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V4-Flash-0731/resolve/${REV}"
JOBS="${DSV4_DOWNLOAD_JOBS:-4}"
STALL_LIMIT_S="${DSV4_STALL_LIMIT_S:-120}"

# locate manifest next to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${SCRIPT_DIR}/dsv4_shard_sizes.txt"

if [ "$#" -lt 1 ]; then
    echo "usage: $0 MODEL_DIR" >&2
    exit 2
fi
MODEL_DIR="$1"
mkdir -p "$MODEL_DIR"

SMALL_FILES="config.json tokenizer.json tokenizer_config.json generation_config.json"

# ---------------------------------------------------------------- small files --
echo "[dsv4] downloading small metadata files"
for f in $SMALL_FILES; do
    curl -sfL --max-time 120 -o "${MODEL_DIR}/${f}" "${BASE}/${f}"
    echo "[dsv4]   ${f}: $(wc -c < "${MODEL_DIR}/${f}") bytes"
done

# ------------------------------------------------------------------ manifest --
if [ ! -f "$MANIFEST" ]; then
    echo "[dsv4] manifest not found: $MANIFEST" >&2
    exit 1
fi

TOTAL_EXPECTED=0
while read -r name expected sha; do
    TOTAL_EXPECTED=$((TOTAL_EXPECTED + expected))
done < "$MANIFEST"
NMANIFEST=$(wc -l < "$MANIFEST")
if [ "$NMANIFEST" -ne 48 ]; then
    echo "[dsv4] manifest has $NMANIFEST lines, expected 48" >&2
    exit 1
fi

# ------------------------------------------------------------ disk space check --
AVAIL=$(df -Pk "$MODEL_DIR" | awk 'NR==2 {print $4}')
# df reports 1 KiB blocks; we need remaining download + 5 GiB working headroom.
WORK_HEADROOM=$((5 * 1024 * 1024))
REMAINING_BYTES=0
while read -r name expected sha; do
    if [ -f "${MODEL_DIR}/${name}" ]; then
        cur=$(wc -c < "${MODEL_DIR}/${name}" 2>/dev/null || echo 0)
        if [ "$cur" -gt "$expected" ]; then
            echo "[dsv4] ${name} is LARGER than its manifest size ($cur > $expected); refusing to download" >&2
            exit 1
        fi
        REMAINING_BYTES=$((REMAINING_BYTES + (expected - cur)))
    else
        REMAINING_BYTES=$((REMAINING_BYTES + expected))
    fi
done < "$MANIFEST"

NEED_KB=$(( (REMAINING_BYTES + 1023) / 1024 + WORK_HEADROOM ))
if [ "$AVAIL" -lt "$NEED_KB" ]; then
    echo "[dsv4] only ${AVAIL} KiB free on $MODEL_DIR, need at least ${NEED_KB} KiB" >&2
    exit 1
fi

# ------------------------------------------------------------- shard download --
# One worker per shard: resume loop that re-reads the current size after every curl
# failure. No --retry; a new process is spawned per attempt so the resume offset is
# always fresh. Timeouts are read/write (connect happens fast to a CDN front).
download_one() {
    local name="$1" expected="$2"
    local file="${MODEL_DIR}/${name}"
    local last=-1 stall=0
    while :; do
        if [ -f "$file" ]; then
            cur=$(wc -c < "$file")
        else
            cur=0
        fi
        if [ "$cur" -gt "$expected" ]; then
            echo "[dsv4] ${name}: on-disk size $cur exceeds manifest $expected; aborting" >&2
            return 1
        fi
        if [ "$cur" -eq "$expected" ]; then
            echo "[dsv4] ${name}: complete (${expected} bytes)"
            return 0
        fi
        # no-progress watchdog: if two consecutive attempts make no forward progress
        # across a stall window, this worker gives up instead of looping forever.
        if [ "$cur" -eq "$last" ]; then
            stall=$((stall + 1))
            if [ "$stall" -ge 3 ]; then
                echo "[dsv4] ${name}: stalled at ${cur} bytes across attempts; giving up" >&2
                return 1
            fi
        else
            stall=0
        fi
        last=$cur

        if ! curl -sfL --connect-timeout 30 --max-time 3600 \
                 --speed-limit 1 --speed-time "$STALL_LIMIT_S" -C - \
                 -o "$file" "${BASE}/${name}"; then
            echo "[dsv4] ${name}: curl failed at ${cur} bytes; retrying"
            sleep 1
            continue
        fi
        # curl exited 0: file must now match the manifest size or we reject it.
        cur=$(wc -c < "$file")
        if [ "$cur" -ne "$expected" ]; then
            echo "[dsv4] ${name}: finished at ${cur} bytes, manifest says ${expected}; aborting" >&2
            return 1
        fi
        echo "[dsv4] ${name}: complete (${expected} bytes)"
        return 0
    done
}
export -f download_one
export MODEL_DIR BASE STALL_LIMIT_S

echo "[dsv4] downloading 48 shards with ${JOBS} concurrent worker(s), ${REMAINING_BYTES} bytes remaining"
# shellcheck disable=SC2016
awk '$0 !~ /^#/ && NF == 3 {print $1, $2}' "$MANIFEST" \
  | xargs -P "$JOBS" -n 2 bash -c 'download_one "$0" "$1"'

# ------------------------------------------------------------ checksum verify --
echo "[dsv4] verifying SHA-256 of all 48 shards"
FAIL=0
while read -r name expected sha; do
    if [ ! -f "${MODEL_DIR}/${name}" ]; then
        echo "[dsv4] MISSING ${name}" >&2
        FAIL=1
        continue
    fi
    got=$(sha256_file "${MODEL_DIR}/${name}")
    if [ "$got" != "$sha" ]; then
        echo "[dsv4] SHA-256 MISMATCH ${name}: got ${got} want ${sha}" >&2
        FAIL=1
    fi
done < "$MANIFEST"
if [ "$FAIL" -ne 0 ]; then
    echo "[dsv4] checksum verification FAILED" >&2
    exit 1
fi

echo "[dsv4] all 48 shards present and SHA-256 verified"
if [ -x "${SCRIPT_DIR}/doctor.sh" ]; then
    echo "[dsv4] running doctor"
    DSV4_SHARDS_VERIFIED=1 "${SCRIPT_DIR}/doctor.sh" "$MODEL_DIR" || exit 1
fi
echo "[dsv4] DONE: model directory ready at ${MODEL_DIR}"
