# Kernel benchmarks

`make bench-dsv4` builds `bin/bench_dsv4_kernels`, which measures the FP8/FP4
GEMV kernels at the exact shapes the expert layers run, with deterministic
inputs and a printed checksum per run.

```
FP4 2048x4096   gate/up:   2048 rows x 4096 cols
FP4 4096x2048   down:      4096 rows x 2048 cols
FP8 4096x4096   dense projection
```

The checksum pins the numerics: a before/after optimisation comparison is only
meaningful if the checksums match. Absolute milliseconds depend on the machine.

The FP8/FP4 activation path quantises the input once (per-128 E8M0), then the
weight loop is a 256-entry decode lookup; this is the same kernel the engine's
expert layers use, so the benchmark reflects real per-token cost rather than a
toy loop.
