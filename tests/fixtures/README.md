# DeepSeek-V4-Flash-0731 test fixtures

These small, deterministic files let `make test` validate the engine without
network access or the 166.9 GB checkpoint.

| path | purpose |
|---|---|
| `dsv4_config.json` | released configuration shape and planner parsing |
| `st/` | two synthetic safetensors shards covering ranks, offsets and dtypes |
| `tiny_dsv4/` | non-zero four-layer DeepSeek-V4-Flash-0731 graph plus scalar-oracle logits |

The tiny graph uses the same tensor names, FP8/FP4 block formats, expert layout,
Hyper-Connection flow and ratio-4/ratio-128 compression boundaries as the full
checkpoint at deliberately small dimensions. Its 130-position oracle crosses
both compression modes and ring-buffer wraparound; native and portable paths
must report `maxdiff=0.000000`.

Regenerate the tiny checkpoint and oracle with:

```bash
python3 tools/make_tiny_dsv4.py
python3 tools/reference_tiny_dsv4.py \
  tests/fixtures/tiny_dsv4 tests/fixtures/tiny_dsv4/ref_logits.json
```

The checked-in files remain authoritative for release tests. A regenerated
fixture should only replace them when its generator and independent reference
calculation have both been reviewed.
