# Security

The model directory is an input trust boundary. The engine must never execute
checkpoint-contained Python code, and it never does: `config.json`,
`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`,
`model.safetensors.index.json` and the shard headers are parsed as data only.

## Defensive checks

- Config lengths, shapes, offsets, integer conversions and allocation products
  are bounds-checked before use; a malformed config refuses to load rather than
  guessing defaults.
- The safetensors reader validates dtype, rank, shape, data offsets and EOF for
  every tensor and rejects size/shape mismatches. The engine maps only validated,
  read-only layer ranges; streamed expert reads remain explicitly bounded.
- The tokenizer loader checks numeric ids, added-token fields and merge entries;
  the JSON parser caps nesting depth to stop hostile recursion.

## Reporting

Report a vulnerability privately to the maintainers; do not open a public issue
with exploit detail. Please include the affected component and a minimal
reproducer where possible.
