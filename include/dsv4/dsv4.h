/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4.h, DeepSeek-V4-Flash-0731 inference engine: public API and core types.
 *
 * OVERVIEW
 *   DeepSeek-V4-Flash-0731 is a 284B-parameter (13B active) mixture-of-experts
 *   model released by DeepSeek-AI as native FP8/FP4 weights. This engine runs it
 *   on a CPU with no GPU, CUDA, PyTorch or ONNX Runtime: pure C99 plus OpenMP.
 *
 *   The trick that makes 166.9 GB of checkpoint usable on a small machine is
 *   streaming: the always-active layers are loaded per layer and the 256 routed
 *   FP4 experts are loaded on demand from a bounded LRU cache, exactly like
 *   kimi-k3-in-c streams its MXFP4 experts. See NOTICE.
 *
 *   The model itself (all values verified against the fixed ModelScope revision
 *   f981a343464c25f82b901e5882716b3b2fa514de):
 *     43 layers, hidden 4096, vocab 129280, 64 heads x head_dim 512
 *     q_lora 1024, o_groups 8, o_lora 1024, sliding window 128
 *     256 routed FP4 experts top-6 + 1 shared FP8 expert, moe_inter 2048
 *     Hyper-Connection mult 4, 3 hash layers, compress_ratios alternate 4/128
 *     BF16 trunk, FP8 E4M3 dense matrices, FP4 E2M1 routed experts
 *
 * THREE FIGURES, EASY TO CONFUSE
 *   166.9 GB   total checkpoint bytes on disk (48 shards, 67,612 main tensors)
 *   284B       total parameters (routed experts dominate)
 *   13B        parameters active per token (the 6 routed experts + dense trunk)
 *
 * NUMERICAL CONTRACT (each restated at its point of use)
 *   - BF16 widening is a left shift; F32->BF16 is round-to-nearest-even.
 *   - FP8 E4M3 and E8M0 decode exactly per the OCP/FP8 spec; see dsv4_fp8_e4m3().
 *   - FP4 E2M1 nibbles are packed two-per-byte, low nibble first.
 *   - FP8/FP4 GEMVs decode the quantised ACTIVATION once and reuse it for every
 *     output row; the 256-entry FP8 decode table makes the weight loop a lookup.
 *   - Every weight-tagged struct must be memset before use; a NULL pointer or a
 *     zero wdt selects a different (and wrong) code path.
 */
#ifndef DSV4_H
#define DSV4_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DSV4_VERSION "0.1.0"

/* Hard caps. DSV4_MAX_LAYERS bounds the per-layer compress_ratio array; the
 * official model has 43 layers. DSV4_MAX_TOPK bounds the routing arrays.
 * The checkpoint has 256 experts but the engine sizes caches from config, so
 * n_experts is a config value, not a compile-time constant. */
#define DSV4_MAX_LAYERS 128
#define DSV4_MAX_TOPK   32
#define DSV4_MAX_DSPARK_STAGES 8

/* ----------------------------------------------------------------- config ---- */
typedef struct {
    int hidden;            /* 4096  */
    int n_layers;          /* 43    */
    int vocab;             /* 129280 */
    int moe_inter;         /* 2048  */
    int n_experts;         /* 256   */
    int n_shared;          /* 1     */
    int topk;              /* 6     */
    int n_hash_layers;     /* 3     */
    int n_heads;           /* 64    */
    int head_dim;          /* 512   */
    int rope_dim;          /* 64    */
    int q_lora;            /* 1024  */
    int o_groups;          /* 8     */
    int o_lora;            /* 1024  */
    int window;            /* 128   */
    int index_heads;       /* 64    */
    int index_dim;         /* 128   */
    int index_topk;        /* 512   */
    int hc_mult;           /* 4     */
    int hc_iters;          /* 20    */
    int max_position;      /* 1048576 */
    int original_position; /* 65536 */
    float rope_theta;      /* 10000   */
    float compress_rope_theta; /* 160000 */
    float rope_factor;     /* 16     */
    float beta_fast;       /* 32     */
    float beta_slow;       /* 1      */
    float rms_eps;         /* 1e-6   */
    float hc_eps;          /* 1e-6   */
    float route_scale;     /* 1.5    */
    float swiglu_limit;    /* 10.0   */
    int compress_ratio[DSV4_MAX_LAYERS];
    int dspark_stages;     /* 3; zero when the checkpoint has no draft model */
    int dspark_block_size; /* 5 draft positions per speculative block          */
    int dspark_noise_token;/* 128799                                           */
    int dspark_markov_rank;/* 256                                              */
    int dspark_target_layer[DSV4_MAX_DSPARK_STAGES]; /* 40, 41, 42            */
} DSV4Config;

/* ----------------------------------------------------------------- memory ---- */
typedef struct {
    uint64_t total_budget_bytes;   /* --memory-gib in bytes                     */
    uint64_t runtime_reserve_bytes;/* working set + optional DSpark persistent   */
    uint64_t context_bytes;        /* KV window + compressors + indexers         */
    uint64_t wo_a_cache_bytes;     /* persistent decoded attention projections   */
    uint64_t expert_cache_bytes;   /* the bounded LRU expert cache               */
    uint64_t planned_bytes;        /* reserve + context + wo_a + expert cache     */
    int wo_a_cache_layers;         /* decoded wo_a prefixes kept resident         */
    int expert_cache_slots;        /* how many packed experts fit                */
} DSV4MemoryPlan;

typedef struct {
    int max_context;               /* context selected by caller; 0 = 4096       */
    uint64_t expert_cache_bytes;   /* advanced --cache-gib, mutually exclusive    */
    int wo_a_cache_layers;         /* supplied by the total-memory planner        */
    int threads;                   /* --threads, 0 = automatic (up to 12)        */
} DSV4Options;

/* ----------------------------------------------------------------- model ---- */
typedef struct DSV4Model DSV4Model;

/* Read and validate config.json. Every field the engine needs is REQUIRED; an
 * absent field fails the load (never a default). Returns 1 on success. */
int dsv4_config_load_file(DSV4Config *c, const char *path);

/* Fill a DSV4MemoryPlan from a total budget. Fails (returns 0) when the budget
 * cannot hold runtime reserve + context + at least one expert. */
int dsv4_memory_plan(const DSV4Config *c, int context, uint64_t budget_bytes,
                     DSV4MemoryPlan *plan);

/* Select a useful context from the checkpoint limits and total memory budget.
 * The policy preserves the baseline wo_a prefix and at least 90% of its expert
 * cache slots, so a longer chat window does not silently destroy decode speed.
 * Returns 0 when even a one-token context cannot fit. */
int dsv4_auto_context(const DSV4Config *c, uint64_t budget_bytes);

/* Persistent context memory in bytes for `context` tokens (KV window +
 * compressors + indexers). Used by the CLI to build a --cache-gib budget. */
uint64_t dsv4_context_bytes(const DSV4Config *c, int context);

/* Single-turn chat template. Writes the rendered prompt into buf (NUL
 * terminated). Returns the number of bytes needed (excluding NUL) or a
 * negative value on refusal. See dsv4_prompt.c for the exact concatenation. */
int dsv4_format_prompt(char *buf, size_t cap, const char *user,
                       const char *system, const char *reasoning_prefix);

/* Format a later user turn for an already-open chat context. This deliberately
 * omits BOS, system text and the reasoning instruction prefix. */
int dsv4_format_turn(char *buf, size_t cap, const char *user,
                     const char *reasoning_prefix);

/* Open the checkpoint in dir, sized per options. Returns NULL on failure
 * (prints the reason to stderr). */
DSV4Model *dsv4_model_open(const char *dir, const DSV4Options *opt);
void dsv4_model_close(DSV4Model *m);

/* Start a fresh conversation without unloading weights or persistent caches. */
void dsv4_model_reset_context(DSV4Model *m);

/* One forward step for token_id at absolute position. Fills logits[vocab]
 * (caller-allocated). Returns 0 on success. */
int dsv4_forward_token(DSV4Model *m, int token_id, int position, float *logits);

/* Prefill a contiguous token sequence in layer-major order. Each layer's
 * streamed weights are loaded once for the whole sequence; token order within
 * every layer is unchanged. Fills logits for the final token only. */
int dsv4_prefill(DSV4Model *m, const int *token_ids, int n_tokens,
                 int start_position, float *logits);

const DSV4Config *dsv4_model_config(const DSV4Model *m);

/* Print config, memory plan, cache/I-O counters, OpenMP threads and peak RSS. */
void dsv4_model_report(const DSV4Model *m);

/* ------------------------------------------------------------ numerics ----
 * Public low-level kernels so weightless tests can exercise the exact numerics
 * without building a model. See dsv4_ops.c for the precise semantics.
 */

/* BF16 <-> F32. Widening is a left shift; narrowing is round-to-nearest-even. */
static inline float dsv4_bf16_to_f32(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}
uint16_t dsv4_f32_to_bf16(float f);

/* FP8 E4M3 decode. 448 is the largest finite value; (exp==15, man==7) is NaN. */
float dsv4_fp8_e4m3(uint8_t code);
/* E8M0: 2^(code-127); 255 is NaN by spec, mapped to +inf here. */
float dsv4_e8m0(uint8_t code);

/* Dynamic activation FP8 quantisation: per 128-element group, amax rounded UP
 * to a power of two, values clamped to [-448, 448]. out must hold n bytes;
 * scales out must hold n/128 floats. */
void dsv4_quant_fp8_act(float *out, float *scales, const float *x, int n);

/* y[rows] = W[rows][in] . x[in], W stored as FP8 E4M3 with per-block E8M0
 * scales [ceil(rows/128)][ceil(in/128)]. x is decoded to float ONCE per call.
 * wscale is the decoded weight block scale table (rows*cols/128/128 floats). */
void dsv4_gemv_fp8(float *y, const float *x, const uint8_t *W, const float *wscale,
                   int in, int rows, int block);
/* FP8 decode table version used by the real model. */
void dsv4_gemv_fp8_tbl(float *y, const float *x, const uint8_t *W, const float *wscale,
                       int in, int rows, int block);

/* y[rows] = W[rows][in] . x[in], W packed FP4 E2M1 (2 nibbles/byte, low nibble
 * first) with per-row E8M0 scales [rows][ceil(in/32)]. x is FP8-quantised to
 * qx/qscale with 128-element groups first (activations stay FP8). */
void dsv4_gemv_fp4(float *y, const float *qx, const float *qscale,
                   const uint8_t *W, const float *wscale,
                   int in, int rows, int group);

/* y = w * x / sqrt(mean(x^2) + eps), elementwise, n wide. */
void dsv4_rmsnorm(float *y, const float *x, const float *w, int n, float eps);

/* y = w * x / sqrt(mean(x^2) + eps) with NO gain vector (w = NULL), n wide. */
void dsv4_rmsnorm_plain(float *y, const float *x, int n, float eps);

/* Rotary position embedding on the LAST rope_dim elements of each head of x.
 * x is [heads][head_dim]; positions advance by one per call. theta is the base;
 * when yarn != 0 the frequencies are interpolated with the YaRN ramp
 * (factor/beta_fast/beta_slow/original_position). inplace. */
void dsv4_rope_apply(float *x, int heads, int head_dim, int rope_dim,
                     int position, float theta, int yarn,
                     float factor, float beta_fast, float beta_slow,
                     int original_position);
/* Inverse rotation (attention output). */
void dsv4_rope_apply_inv(float *x, int heads, int head_dim, int rope_dim,
                         int position, float theta, int yarn,
                         float factor, float beta_fast, float beta_slow,
                         int original_position);

/* Normalised Hadamard transform on n elements (n must be a power of two). */
void dsv4_hadamard(float *x, int n);

/* Hyper-Connection Sinkhorn split, matching hc_split_sinkhorn() in the
 * released kernel. mixes layout: [pre(hc) | post(hc) | comb(hc*hc)] with the
 * scale/base per section; pre = sigmoid(x*scale0+base0)+eps; post =
 * 2*sigmoid(x*scale1+base1); comb = softmax rows + eps, then ONE column
 * normalisation, then (iters-1) alternating row/column normalisations (so the
 * final step is a column normalisation). */
void dsv4_hc_split(float *pre, float *post, float *comb,
                   const float *mixes, const float *hc_scale,
                   const float *hc_base, int mult, int iters, float eps);

/* Router: z = gate row dot x, score = sqrt(softplus(z)); choice = score + bias
 * selects the top-k, but the COMBINING weights come from the UNBIASED score.
 * Writes expert ids sorted ascending and their normalised weights (sum = 1,
 * then scaled by route_scale). tid2eid non-NULL only for the first n_hash_layers
 * layers: the expert list is looked up directly. */
void dsv4_router(int *idx, float *wt, const float *x, const float *gate,
                 const float *bias, const int64_t *tid2eid, int token_id,
                 int hidden, int n_experts, int topk, int hash_layer,
                 float route_scale, float rms_eps);

#ifdef __cplusplus
}
#endif

#endif /* DSV4_H */
