#!/usr/bin/env bash
# 01-hello.sh - the quickest possible smoke test: one token from a short prompt.
#
#   scripts/try-dsv4.sh "科技的边界在哪里？"
#
# Requires bin/dsv4 to be built (make -j) and the checkpoint downloaded
# (scripts/download-dsv4.sh). Produces a single token so it finishes quickly
# even on a modest machine.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../scripts/try-dsv4.sh" "科技的边界在哪里？"
