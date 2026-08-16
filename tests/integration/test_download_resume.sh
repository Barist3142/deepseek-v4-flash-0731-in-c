#!/usr/bin/env bash
# test_download_resume.sh - prove the resume path is monotonic.
#
# Starts a local HTTP server that serves a 64 KiB file in 16 KiB chunks, dropping
# the connection after each chunk. A resume worker re-reads the on-disk length
# after every failure and starts a fresh curl -C -; the test asserts the final
# file is byte-for-byte identical to the source and that every observed length
# was non-decreasing (no truncation, no backsliding).
set -euo pipefail

TMP=$(mktemp -d)
trap 'kill "$SERVER_PID" 2>/dev/null || true; rm -rf "$TMP"' EXIT

SRC="$TMP/src.bin"
OUT="$TMP/out.bin"
# deterministic 64 KiB payload
python3 - "$SRC" <<'PY'
import os, sys
n = 65536
data = bytes((i * 131 + 7) & 0xFF for i in range(n))
open(sys.argv[1], 'wb').write(data)
PY

# minimal HTTP server that serves `src.bin` in 16 KiB chunks, then hangs up
python3 - "$TMP" 2>"$TMP/server.log" <<'PY' &
import http.server, socketserver, os, sys
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        path = os.path.join(sys.argv[1], 'src.bin')
        size = os.path.getsize(path)
        rng = self.headers.get('Range')
        start = 0
        if rng and rng.startswith('bytes='):
            start = int(rng.split('=')[1].split('-')[0])
        if start >= size:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()
            return
        with open(path, 'rb') as f:
            self.send_response(206 if start else 200)
            self.send_header('Content-Length', str(size - start))
            self.send_header('Accept-Ranges', 'bytes')
            if start:
                self.send_header('Content-Range', f'bytes {start}-{size - 1}/{size}')
            self.end_headers()
            f.seek(start)
            chunk = f.read(16384)
            if not chunk: return
            self.wfile.write(chunk)
            self.wfile.flush()
            # deliberately drop the connection after one chunk
            self.connection.close()
class Server(socketserver.TCPServer):
    allow_reuse_address = True
with Server(('127.0.0.1', 0), H) as httpd:
    with open(os.path.join(sys.argv[1], 'port'), 'w') as f:
        f.write(str(httpd.server_address[1]))
    httpd.serve_forever()
PY
SERVER_PID=$!
for _ in {1..50}; do
    [ -s "$TMP/port" ] && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: test HTTP server did not start" >&2
        cat "$TMP/server.log" >&2
        exit 1
    fi
    sleep 0.1
done
[ -s "$TMP/port" ] || { echo "FAIL: timed out starting test HTTP server" >&2; exit 1; }
PORT=$(cat "$TMP/port")

# worker: re-read length, resume from it, restart on failure, reject overshoot
PROTO="http://127.0.0.1:${PORT}/src.bin"
PREV=0
ATTEMPTS=0
while :; do
    if [ -f "$OUT" ]; then CUR=$(wc -c < "$OUT"); else CUR=0; fi
    [ "$CUR" -lt "$PREV" ] && { echo "FAIL: length backslid $PREV -> $CUR" >&2; exit 1; }
    PREV=$CUR
    [ "$CUR" -ge 65536 ] && break
    ATTEMPTS=$((ATTEMPTS + 1))
    [ "$ATTEMPTS" -gt 100 ] && { echo "FAIL: too many attempts" >&2; exit 1; }
    curl -sf --max-time 10 -C "$CUR" -o "$OUT" "$PROTO" || true
done

if ! cmp -s "$SRC" "$OUT"; then
    echo "FAIL: downloaded bytes differ from source" >&2
    exit 1
fi
echo "resume OK: 64 KiB reconstructed monotonically in $ATTEMPTS attempts"
