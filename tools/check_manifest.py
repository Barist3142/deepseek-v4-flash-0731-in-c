#!/usr/bin/env python3
"""check_manifest.py - validate scripts/dsv4_shard_sizes.txt: exactly 48 lines,
each `<name> <size> <sha256>`, 64-bit hashes, total 166,886,535,336 bytes.

Used by CI so a hand-edited manifest cannot silently drift.
"""
import re
import sys

MANIFEST = "scripts/dsv4_shard_sizes.txt"
EXPECTED_TOTAL = 166886535336
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def main():
    lines = []
    with open(MANIFEST) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                lines.append(ln)
    if len(lines) != 48:
        print(f"FAIL: manifest has {len(lines)} entries, expected 48", file=sys.stderr)
        return 1
    total = 0
    for i, ln in enumerate(lines, 1):
        parts = ln.split()
        if len(parts) != 3:
            print(f"FAIL: line {i} does not have 3 fields", file=sys.stderr)
            return 1
        name, size, sha = parts
        if not re.match(r"^model-\d{5}-of-00048\.safetensors$", name):
            print(f"FAIL: line {i} bad name {name!r}", file=sys.stderr)
            return 1
        if not size.isdigit() or int(size) <= 0:
            print(f"FAIL: line {i} bad size {size!r}", file=sys.stderr)
            return 1
        if not SHA_RE.match(sha):
            print(f"FAIL: line {i} bad sha256 {sha!r}", file=sys.stderr)
            return 1
        total += int(size)
    if total != EXPECTED_TOTAL:
        print(f"FAIL: total {total} != {EXPECTED_TOTAL}", file=sys.stderr)
        return 1
    print(f"manifest OK: 48 shards, total {total} bytes, 64-bit hashes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
