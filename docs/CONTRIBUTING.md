# Contributing

Thanks for considering contributing. The gate for every change is `make test`:
the weightless suite must stay green, and the tiny-model graph test compares
against an independent Python scalar oracle, so numerical changes are caught
by construction.

## Workflow

1. Fork and branch.
2. Make one focused change.
3. Run `make test`, `make strict`, and `make asan` locally.
4. If you touch numerics, regenerate the oracle
   (`python3 tools/make_tiny_dsv4.py tests/fixtures/tiny_dsv4` then
   `python3 tools/reference_tiny_dsv4.py tests/fixtures/tiny_dsv4`) and confirm
   the graph test still passes.
5. Open a PR with a description of what changed and why.

## Style

- C99, no VLAs in new code, `-Wall -Wextra -Wpointer-arith -Wshadow -Wvla`
  clean (see GNUmakefile).
- Keep `-ffp-contract=off`; do not hand-tune accumulation order without passing
  the logits/token parity gate.
- Comments explain *why*, not what. Numerical contracts are restated at their
  point of use.

## Boundaries

- Do not commit weights, logs, credentials, or machine-local paths.
- Do not add runtime dependencies on Python; Python is for download, fixtures,
  and the independent oracle only.
