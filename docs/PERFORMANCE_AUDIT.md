# Performance audit

This file records the performance work and how it is measured. Numbers are
machine-specific; the invariants are the checksums and the token IDs.

## Kernel microbenchmarks

`make bench-dsv4` measures the exact shapes the expert layers run:

| shape | purpose |
|---|---|
| FP4 2048×4096 | routed gate/up |
| FP4 4096×2048 | routed down |
| FP8 4096×4096 | dense projection |

The production kernels use lookup-table weight decoding and quantise each input
once per projection. The benchmark performs one warm-up and reports the best of
three measured calls. Absolute milliseconds differ by CPU; the printed
checksums must stay stable across optimisation levels.

## Expert I/O and compute overlap

Before each MoE block, the main thread reserves cache slots for all routed
expert misses. POSIX read workers fill those slots concurrently while the main
thread computes cache-hit routed experts and the shared expert. Each expert has
an independent output buffer; after every required read and projection finishes,
the outputs are summed in ascending expert-ID order and the shared result is
added last. This preserves the original floating-point accumulation order.

`DSV4_NO_EXPERT_OVERLAP=1` selects the synchronous path in the same binary. A
controlled 9 GiB, 12-thread A/B run produced the exact oracle and identical
cache counts and read volume in both modes:

| mode | wall | forward | MoE | expert wait | expert I/O span |
|---|---:|---:|---:|---:|---:|
| synchronous | 82.72 s | 82.493 s | 48.753 s | 37.496 s | 37.496 s |
| overlapped | 79.52 s | 79.262 s | 44.329 s | 31.534 s | 37.981 s |

The overlap hid about 6 seconds of expert-read latency and reduced wall time by
3.2 seconds (3.9%). I/O remains the dominant optimisation target; the measured
gain is intentionally reported instead of an extrapolated pipeline estimate.

## Dense projection cache and layer read-ahead

The total-memory planner first keeps decoded `wo_a` prefixes while reserving at
least 1 GiB for the expert LRU. On the WSL2 reference host, a 10 GiB plan holds
all 43 `wo_a` layers (2.69 GiB) and 487 expert slots (6.07 GiB).
Eliminating repeated reads and FP8-to-BF16 decoding of those projections reduced
the ext4 1-token smoke test from 251 s to approximately 124 s on that host.

Keeping packed `wo_b` for all 43 layers was tested and rejected. It consumed
1.34 GiB, reduced the expert cache to 379 slots, and raised 16-token expert
traffic from 31.85 GiB to 36.11 GiB. Wall time stayed flat (419 s versus 418 s),
so the extra persistent cache did not justify its expert-cache cost.

After loading layer L, the engine issues `POSIX_FADV_WILLNEED` for the
non-expert tensors in layer L+1, overlapping buffered reads with the current
layer's compute. Experts stay on the `O_DIRECT` path, and cached `wo_a` tensors
are skipped. `DSV4_NO_PREFETCH=1` disables only this read-ahead for same-binary
A/B measurements.

On the WSL2 reference host (8 threads, 10 GiB, 16 generated tokens), read-ahead
reduced wall time from 430.1 s to 420.1 s (2.3%). Both runs produced the exact
oracle token sequence, 2,602 expert hits / 2,558 misses, and 31.85 GiB of expert
reads. The read-ahead run advised 97.70 GiB and peaked at 8.98 GiB RSS.

## Next-layer load overlap

Once a layer's decoded `wo_a` is resident, loading the rest of that layer is
allocation and buffered I/O only. During incremental decode, a POSIX worker now
loads layer L+1 into an independent bundle while the main OpenMP team computes
layer L. Layers whose `wo_a` is not resident stay synchronous, preventing a
background `wo_a` decode from opening a competing OpenMP region.

`DSV4_NO_LAYER_OVERLAP=1` selects the synchronous path. A same-binary 9 GiB,
12-thread A/B produced the exact oracle, 2,343 hits / 2,817 misses and 35.07 GiB
of expert reads in both modes:

| mode | wall | forward | critical-path load | actual layer I/O | peak RSS |
|---|---:|---:|---:|---:|---:|
| synchronous | 69.62 s | 69.396 s | 16.708 s | 16.708 s | 7.99 GiB |
| overlapped | 65.90 s | 65.686 s | 9.801 s | 31.129 s | 8.16 GiB |

Concurrent storage traffic made the layer reads themselves slower, but hid
enough of them to reduce wall time by 3.72 seconds (5.3%). The extra live layer
bundle added about 0.17 GiB RSS and did not cause swapping.

## Decode tables and paired projections

The exact FP8/FP4 kernels now use 256-entry FP8/E8M0 tables and a 16-entry FP4
table in their real inference loops. Previously only auxiliary kernels used the
FP8 table while the production path called `ldexpf` for every weight or scale.
Inputs are still quantized once per projection and shared by every output row;
packed FP4 expert weights are never expanded.

In the AVX2 FP8 row-lane kernel, all eight output lanes belong to the same
128-row scale block. The kernel therefore decodes that scale once and broadcasts
it instead of performing eight identical table reads. Repeated best-of-three
4096x4096 microbenchmarks moved from 0.637-0.646 ms to 0.575-0.600 ms with the
same checksum. End-to-end impact is smaller because FP4 MoE and storage traffic
dominate this workload.

The portable routed/shared expert path computes independent gate/up projections
in one OpenMP region. The native AVX2 path runs the row-lane kernel for each
matrix. Each vector lane is one output row and columns still advance left to
right, so there is no horizontal reduction or reordered accumulation.

On the WSL2 host (Core i5-1340P, 8 threads, ext4 model, 10 GiB plan), the staged
16-token measurement was 420.1 s before decode tables, 192.6 s with tables, and
178.3 s with tables plus paired gate/up projections. The final run produced the
exact 16-token oracle, read 31.85 GiB of experts, and peaked at 8.97 GiB RSS.
The 1-token smoke was 70.4 s, down from the original 251 s baseline.
Laptop thermals and page-cache state add timing variance, so token IDs, byte
counts, and repeated same-binary measurements remain the acceptance criteria.

## Layer-major prefill, startup decode and AVX2

Prompt prefill is layer-major: all prompt positions traverse a loaded layer in
their original order before the layer bundle is released. This preserves the
attention/compressor recurrence while loading each non-expert layer bundle once
per prompt instead of once per prompt token. The final hidden states require
`tokens * hc_mult * hidden * sizeof(float)` temporary memory (about 320 KiB for
the five-token acceptance prompt and at most 256 MiB at context 4096).

Persistent `wo_a` weights are decoded from FP8 to BF16 with the same lookup
tables as the GEMV kernels, in parallel by output row. The vocab head streams
1024 rows (8 MiB) per read and uses one persistent OpenMP region per head call.
`--threads` now sets the OpenMP runtime globally, so the user-visible choice also
controls every GEMV rather than only expert reads and the head. Automatic
selection is capped at 12 threads; machines with fewer available processors use
the smaller count.

Native AVX2/FMA builds process eight output rows as eight independent vector
lanes. Every lane retains the scalar column order and scale-block boundaries;
`DSV4_NO_SIMD=1` selects the scalar path for A/B checks. The unit suite compares
native and scalar FP8/FP4 output byte for byte, and the full oracle remains the
end-to-end gate.

Indicative one-token profiles from the same WSL2 host and command show where the
work moved; thermal and cache state were not fixed:

| stage | forward total | layer load | attention | MoE | vocab head |
|---|---:|---:|---:|---:|---:|
| token-major prefill | 108.74 s | 61.50 s | 13.52 s | 29.98 s | 3.06 s |
| layer-major prefill | 95.99 s | 48.79 s | 13.22 s | 30.00 s | 3.45 s |
| parallel lookup decode for cached `wo_a` | 52.49 s | 10.90 s | 10.18 s | 28.36 s | 2.56 s |
| 1024-row head + AVX2 row lanes | 49.00 s | 11.86 s | 8.23 s | 25.51 s | 2.89 s |

## OpenMP wait policy and current acceptance run

The CLI defaults to `OMP_WAIT_POLICY=PASSIVE` before the OpenMP runtime starts.
The engine alternates parallel GEMVs with streamed expert I/O; sleeping workers
do not compete with read workers or consume a laptop's sustained thermal budget.
An explicitly supplied policy is never overwritten, and the convenience script
sets the same default. On the reference laptop in an already-hot state, a
same-binary 12-thread, 7-token comparison reduced TTFT from about 56 seconds to
39.336 seconds and TPOT from 8.610 to 5.591 seconds/token when switching from
ACTIVE to PASSIVE. These hot-state numbers diagnose policy behavior and are not
substitutes for the controlled reference result below.

MoE route, activation and expert workspaces are allocated once per model rather
than once per layer. One FP8 quantisation of the MoE input is shared by all six
routed experts and the shared expert; attention likewise reuses one quantised
input for `wq_a` and `wkv`. These changes remove duplicate work without changing
matrix traversal or floating-point accumulation order.

The attention path now also reuses the quantised `q_lora` projection in the
ratio-4 indexer, computes one RoPE frequency table for query, key and inverse
output rotation, and writes grouped output projections without intermediate
copies. Hyper-Connection state, final hidden state and the streamed vocabulary
buffer are model-owned scratch. The CLI decodes and flushes each generated
token immediately, so measured TTFT is visible to an interactive user rather
than hidden behind full-response buffering.

The packed-FP4 AVX2 loop consumes both low and high nibbles after loading each
weight byte once. The two FMA operations remain in the original column order;
the unit suite compares every native result byte-for-byte with the scalar path.
The CLI also supports a resident multi-turn mode: later turns keep KV,
compressor and expert-cache state, prefill only the newly appended role suffix,
and continue streaming from the same process.

Expert cache misses use a model-lifetime pool of at most four read workers
instead of creating and joining threads in every MoE layer. The read geometry,
O_DIRECT calls and computation overlap are unchanged;
`DSV4_NO_EXPERT_POOL=1` restores the per-batch worker path for measured A/B.

The four-worker cap comes from a mirrored `3, 4, 6, 6, 4, 3` sweep on the
reference i5-1340P / Samsung NVMe WSL2 machine. Every run used continuous AC
power, active external cooling, the HP Optimized power plan, 12 compute threads,
the 14 GiB / 65536-context plan and the fixed 5-token prompt plus 16-token
oracle. No other CPU- or disk-intensive process was active. All six runs
produced the exact oracle:

| expert I/O workers | mean wall | mean TTFT | mean TPOT | mean expert wait | mean expert I/O window |
|---:|---:|---:|---:|---:|---:|
| 3 | 58.296 s | 24.639 s | 2.109 s/token | 25.245 s | 32.284 s |
| 4 | 53.976 s | 21.783 s | 2.060 s/token | 25.422 s | 32.146 s |
| 6 | 58.982 s | 24.796 s | 2.151 s/token | 25.069 s | 31.752 s |

Major faults varied from 6 to 62,616 as clean mapped checkpoint pages were
reclaimed between processes, so TTFT and wall time have visible cache-state
noise. The two four-worker TPOT samples were 2.052 and 2.068 s/token; this
repeatability, together with the lowest means, determines the automatic cap.
`DSV4_EXPERT_IO_WORKERS` remains available for controlled measurements on other
storage and CPU combinations; the automatic setting is used otherwise.

GCC's OpenMP wait policy was separately measured on the same 15 GiB,
65536-context, 12-thread setup with the 5-token prompt and 64 generated tokens.
Both adjacent runs emitted the same IDs, made 6,252 expert misses and read
77.84 GiB. Pure passive waiting measured 15.953 s TTFT and 1.867 s/token TPOT;
`OMP_WAIT_POLICY=PASSIVE` with `GOMP_SPINCOUNT=10000` measured 15.417 s TTFT and
1.768 s/token TPOT. Voluntary context switches fell from about 1.59 million to
245 thousand. The CLI therefore uses the bounded spin by default with GCC: it
bridges short kernel gaps without keeping workers active throughout expert I/O.
Explicit environment settings remain authoritative.

The expert cache is divided evenly between layers whenever its capacity can
hold every layer's routed top-k set. This prevents one layer's active experts
from evicting another's immediately before the next decode step. An internal
`DSV4_EXPERT_CACHE_GLOBAL=1` switch restores global LRU for controlled A/B and
undersized caches fall back automatically.

The current acceptance run used 12 threads, the automatic 14 GiB / 65536-context
plan, read-only layer mapping over ext4 checkpoint storage and continuous AC
power. With the automatic four-worker expert pool it took 54.029 s wall,
measured 23.679 s TTFT and 1.991 s/token TPOT, produced the exact 16-token
oracle, peaked at 19.29 GiB RSS and read 29.42 GiB of routed experts
(2,446 hits / 2,363 misses). This run incurred 62,560 major faults; the mirrored
worker sweep above shows why cache state must accompany laptop timings. The acceptance
length is a correctness contract, not the sustained-performance workload.

The primary chat measurement used an 11-token prompt and generated 64 tokens.
With the same automatic 14 GiB / 65536-context plan it took 160.45 s wall
(160.066 s profiled), measured 31.755 s TTFT and 2.037 s/token TPOT over 63
intervals, peaked at 19.28 GiB RSS, and read 99.29 GiB of experts. TTFT starts
immediately before model-ready prefill and ends after the first token is
delivered to stdout. Model opening and tokenisation remain part of wall time
but outside TTFT.

An adjacent warm-cache 64-token global/per-layer LRU comparison held the
14 GiB plan, 65536 context, prompt, output IDs and 12 threads fixed:

| cache policy | wall | TTFT | TPOT | misses | expert read | MoE | expert wait |
|---|---:|---:|---:|---:|---:|---:|---:|
| global LRU | 161.28 s | 30.508 s | 2.068 s | 8,223 | 102.39 GiB | 126.470 s | 78.531 s |
| per-layer LRU | 160.45 s | 31.755 s | 2.037 s | 7,974 | 99.29 GiB | 124.722 s | 75.722 s |

The partition reduced expert traffic by 3.10 GiB and TPOT by 1.5% without
changing peak memory. TTFT did not improve in this adjacent pair and is not
claimed as a cache-policy benefit.

An earlier controlled 15 GiB / 4096-context run took 158.60 s with layer mmap
and 178.10 s with streamed `pread`; its TTFT measurements were 35.212 s and
42.852 s, and TPOT was 1.952 s/token and 2.108 s/token. Mapping only the
non-expert layer tensors therefore improved wall time by 11.0% and TPOT by
7.4% in that pair. The higher RSS includes clean checkpoint-backed pages that
Linux can reclaim; the automatic memory budget retains substantial headroom
for those pages. `DSV4_NO_LAYER_MMAP=1` keeps the measured fallback available
and mapping failures fall back automatically.

## Memory and thread budget measurements

More cache is useful only while it leaves enough memory for the runtime, Linux
and active file pages. WSL2 was configured to expose 23.47 GiB of a 31.65 GiB
Windows host. Immediately before stdout streaming was enabled, the same
11-token prompt, 64 generated IDs and 12 threads produced the following
memory-only A/B. TTFT in this table ends at token-ready; the engine path is the
same as the streamed run above.

| plan | slots | wall | TTFT | TPOT | expert read | peak RSS | major faults |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 GiB | 407 | 225.16 s | 42.736 s | 2.892 s | 131.52 GiB | 8.21 GiB | 8 |
| 12 GiB | 647 | 196.73 s | 36.909 s | 2.534 s | 107.73 GiB | 11.23 GiB | 4 |
| 15 GiB | 888 | 185.95 s | 34.115 s | 2.406 s | 93.02 GiB | 14.22 GiB | 4 |
| 18 GiB | 1,129 | 270.66 s | 38.093 s | 3.680 s | 83.78 GiB | 17.13 GiB | 441,550 |

All four runs emitted identical token IDs. The 18 GiB plan demonstrates why
cache hit rate is not the objective: it read less expert data but displaced
active pages, inflated layer loading and was 45.6% slower than 15 GiB. The
automatic rule takes the smaller of two thirds total RAM and three quarters
of current `MemAvailable`; it selects 14 GiB with 23.47 GiB visible, retaining
a margin below the measured optimum. On the earlier 15 GiB WSL2 allocation, an
11 GiB run similarly entered sustained memory pressure and was not viable.

In the measured 9 GiB thread sweep, 16 threads took 95.25 s versus 80.54 s for
12 threads. These streaming GEMVs saturate laptop memory bandwidth before
all logical processors become useful. The convenience script therefore caps
automatic selection at 12 threads and uses the same pressure-aware memory rule
before rounding down to whole GiB. Users can
override both settings, but larger values should be justified by measurement on
their own machine.

## Deliberately not done

- `__builtin_prefetch` as a substitute for real expert prefetch: one expert far
  exceeds a private cache and the kernel already streams sequentially.
- `SCHED_FIFO` / CPU pinning: it can stall an interactive laptop workload.
- Horizontal SIMD reductions or multiple accumulators per output row: they
  change float addition order and are outside the numerical contract.
- AVX2 gather for the row-major packed-FP4 lookup: `vgatherdps` changed the two
  expert microbenchmarks from 0.340/0.331 ms to 0.898/0.872 ms. An integer
  `vpshufb` decode measured 0.375/0.384 ms. Both retained checksums but were
  slower than the scalar lookup plus row-lane FMA, so neither is shipped.

## Lossless prompt-lookup verification

The CLI now looks for repeated three- or four-token suffixes in the committed
conversation and verifies their historical continuation in one layer-major
main-model pass. The automatic draft limit is 16 tokens; an internal diagnostic
override can raise it to 63 so `pending + drafts` still fits the 64-position MoE
batch exactly. A single historical match is capped at four drafts; longer
periodic extension requires a second agreeing occurrence. A zero-accept round
disables further attempts for eight delivered tokens. The path is automatic
and `--no-prompt-lookup` provides the scalar A/B.

The verifier parallelises independent Hyper-Connection work across positions,
decodes each vocabulary row once for up to seven outputs, and expands the three
hash-layer cache partitions only for the verification round. The ordinary
six-slot hash partitions are restored before scalar decode resumes. Tiny-model
tests keep every output float exact across both cache layouts.

This path remains lossless rather than approximate:

- The default 15 GiB / 12-thread command reproduced the fixed 16-token oracle
  exactly with prompt lookup enabled.
- A deliberately weak one-token match produced four verify rounds, rejected
  all 14 drafts, and still generated the same 32 IDs as an adjacent scalar run.
- With `temperature=0.1`, `top_p=1` and `seed=42`, prompt lookup and scalar
  decode generated the same 16-token periodic sequence. The verifier consumes
  sampler RNG only for tokens that ordinary decode would deliver.
- The tiny graph's snapshot test writes a rejected suffix across the 128-token
  compressor boundary, commits only the verified prefix, and then requires
  every subsequent logit float to match an independent model instance.

The primary repeated-output pair was measured on 2026-08-16 with the reference
i5-1340P, ext4 checkpoint, GCC 11.4.0, 12 threads, passive OpenMP waiting,
18 GiB memory plan, 65536 context and AC power. The prompt was
`Say hello 20 times.` and both paths produced the same 60 IDs and all 20 lines:

| path | wall | TTFT | TPOT | accepted drafts | expert read | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| scalar | 124.2 s | 27.932 s | 1.594 s/token | disabled | 62.83 GiB | 21.95 GiB |
| prompt lookup | 79.1 s | 22.875 s | 0.939 s/token | 50/51 in 4 rounds | 53.64 GiB | 22.13 GiB |

A separate post-cleanup performance-freeze run under the same conditions
completed the same 60 IDs and all 20 lines, accepted 50/51 drafts and measured
0.892 s/token TPOT, 26.748 seconds TTFT, 53.64 GiB of expert reads and 22.19
GiB peak RSS. It is kept separate from the adjacent A/B pair so measurements
from different runs are not combined into a synthetic best case.

The 41.1% TPOT reduction, 36.3% wall-time reduction, 18.1% TTFT reduction and
14.6% read reduction are specific to accurate repeated continuations.
A separate ordinary 64-token response found no repeated suffix, launched no
verify rounds and measured 1.705 s/token. That neutral result is as important
as the positive A/B: prompt lookup is not presented as a universal one-second
decode rate.

The earlier controlled 15 GiB pair generated the token triple
`33310 2058 201` exactly 16 times. Scalar decode measured 1.770 s/token and
prompt lookup accepted 40/40 drafts at 1.037 s/token. A capped 18 GiB run of the
same 48 IDs reached 0.928 s/token, but it did not complete the requested 20-line
response; it is historical throughput evidence rather than the primary result.

The final ordinary run took 133.8 seconds wall (131.653 seconds profiled), with
20.544 seconds TTFT, 6,252 expert misses, 77.84 GiB of expert reads and 18.96
GiB peak RSS. Its 64 IDs exactly matched the scalar continuation used throughout
the cache and OpenMP experiments.

A later same-day memory pair kept that prompt, all 64 IDs, context and 12
threads fixed. At 15 GiB it measured 20.560 seconds TTFT, 1.791 s/token, 6,252
misses, 77.84 GiB read and 18.95 GiB peak RSS. At 18 GiB it measured 20.203
seconds TTFT, 1.705 s/token, 5,676 misses, 70.67 GiB read and 21.95 GiB peak
RSS. The 4.8% TPOT improvement is real but leaves only about 1.5 GiB outside
peak RSS in the 23.47 GiB WSL guest, so 15 GiB remains the automatic default.

The optional DSpark path was rechecked rather than inferred from draft-model
accuracy alone. On a short exact seven-token check, a four-draft block accepted
4/4 and measured 1.694 s/token versus 1.843 s/token for adjacent scalar decode.
On a 32-token response it averaged only 2.86 accepted drafts per four-token
block and regressed to 2.018 s/token versus 1.760 s/token scalar. Observed
confidence was not monotonic with accepted length: low-confidence blocks could
accept 4/4 while high-confidence blocks could accept only 1/4. DSpark therefore
remains experimental and disabled by default.

Lowering the minimum suffix from three tokens to two on that same ordinary
response launched two rounds, rejected all six drafts, increased expert reads
from 77.84 to 85.59 GiB and regressed TPOT from 1.818 to 1.988 s/token. The
three-token minimum remains the automatic setting.

An adjacent 32-token memory check on the same day kept prompt, output IDs,
context and 12 threads fixed. All three runs incurred about 61.7k major faults:

| plan | TTFT | TPOT | expert read | peak RSS |
|---:|---:|---:|---:|---:|
| 15 GiB | 21.946 s | 1.852 s/token | 45.32 GiB | 18.95 GiB |
| 16 GiB | 20.616 s | 1.818 s/token | 44.13 GiB | 19.94 GiB |
| 17 GiB | 22.808 s | 1.850 s/token | 43.26 GiB | 20.94 GiB |

Sixteen GiB reduced TPOT by only 1.8% while consuming another GiB of resident
memory, and 17 GiB lost that gain despite reading less. The automatic 15 GiB
choice therefore remains the safer laptop default on a WSL guest with
23.47 GiB visible RAM.

Learned rank-0 route prefetch was also rechecked on the 15 GiB, 32-token run.
Its temporary slots adopted 270 of 324 early reads, but TPOT changed only from
1.852 to 1.844 s/token while total expert traffic increased from 45.32 to
46.36 GiB. It remains an experimental switch rather than an automatic cost.

A later 16-token observation found 3.859/6 mean overlap between pre-attention
and final learned routes. Learned rank zero covered 324 of 362 predicted misses
and wasted 38; hash routes were exact. Even hash-only prefetch adopted all
268 reads without waste but measured 1.981 s/token; the adjacent observation-
only run measured 1.867 s/token despite also paying for learned-route
diagnostics. Starting those reads early competes with the mapped
attention-weight stream on this NVMe, so prediction accuracy alone is not a
reason to enable prefetch.

## Prefill-only experiments

A 17-token prompt with one generated token isolates TTFT. At 18 GiB it measured
36.423 seconds: MoE accounted for 29.096 seconds, including 24.220 seconds of
expert wait, with 1,830 cold misses and 22.79 GiB read. Reading misses in
physical expert order was neutral (36.592 seconds, 24.180 seconds wait), and the
buffered path regressed to 39.073 seconds with 27.554 seconds wait. Six I/O
workers produced a lower 35.039-second wall observation but worsened expert wait
to 24.468 seconds; differing page-fault state means that is not evidence for a
new default. A 15 GiB plan measured 36.154 seconds and 24.138 seconds wait, also
effectively neutral because first-use experts cannot benefit from cache size.

The useful change was to the existing prefill double buffer. Previously, its
group width could consume an entire per-layer cache partition; a second group
then had no disjoint victim slots and the long batch silently fell back to
synchronous reads. The default now limits a multi-group batch to half of that
partition. On a 15 GiB, 17-token raw prompt, two cold-page runs had 64.2k major
faults each. The old grouping measured 34.522 seconds TTFT, 26.701 seconds MoE
and 22.034 seconds expert wait. Half-partition grouping produced the identical
token at 30.148 seconds TTFT, 23.086 seconds MoE and 12.796 seconds expert wait:
a 12.7% TTFT reduction without extra cache memory.

The current complete repeated-output run also compared the new default with its
internal fallback. Both generated the same 60 IDs. Half-partition grouping
measured 79.1 seconds wall, 22.875 seconds TTFT and 0.939 s/token; the fallback
measured 92.8 seconds, 32.805 seconds and 0.976 s/token. Major faults differed
(48,481 versus 63,773), so the entire TTFT gap is not attributed to grouping;
the matched-fault 17-token pair above is the cleaner causal result. A stale-tag
guard prevents the next read group from treating an in-flight pinned victim as
a cache hit, and the synchronous fallback bypasses double buffering entirely.
The tiny graph remains bit-exact across both schedules.

Splitting each expert read into `w1/w3` and `w2` phases was rejected. With warm
checkpoint pages on the same 17-token prompt, half-partition grouping alone
measured 25.335 seconds TTFT; adding the split measured 25.565 seconds and
doubled expert read operations from 3,198 to 6,396. The split remains disabled.

The experimental packed-`wo_a` path combines scale and weight decoding into one
65,536-entry lookup and points cached packed bytes directly into the checkpoint
mapping. At 15 GiB it increased expert slots from 825 to 933 and remained exact,
but measured 19.348 seconds TTFT and 1.737 s/token versus 14.523 seconds and
1.718 s/token for the adjacent decoded-BF16 path. It remains opt-in: lower
storage and expert traffic did not translate into user-visible latency.

## Methodology

The latency names follow the serving convention documented by
[vLLM](https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md):
TTFT is time to first token, while TPOT excludes the first output token and
averages the remaining inter-token interval. In this engine TTFT starts after
the resident model is ready and immediately before prefill; model opening and
tokenizer setup remain visible in wall time.

Controlled A/B measurements use the same prompt, weights and threads, with an
environment switch where one exists. End-to-end timings also record environment
and run-to-run variability. Kernel byte comparisons pin the numerics; the
16-token oracle pins the full graph.
