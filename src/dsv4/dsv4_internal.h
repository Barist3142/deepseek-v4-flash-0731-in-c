/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_internal.h - engine-internal structures for the DeepSeek-V4-Flash-0731
 * model. Everything here is private to src/dsv4; the public contract lives
 * in include/dsv4/dsv4.h.
 *
 * The layer weight bundle mirrors the checkpoint layout (verified against the
 * shard headers at revision f981a343464c25f82b901e5882716b3b2fa514de):
 *
 *   BF16 matrices       read as raw uint16_t and multiplied directly
 *   FP8 qmats          weight F8_E4M3 [rows, cols] + scale F8_E8M0
 *                      [ceil(rows/128)][ceil(cols/128)]
 *   FP4 experts        weight I8 [rows, ceil(cols/2)] (2 nibbles/byte, low
 *                      nibble first) + scale F8_E8M0 [rows][ceil(cols/32)]
 *                      stored as one contiguous scale run (w1,w2,w3) and one
 *                      contiguous weight run (w1,w2,w3) per expert
 */
#ifndef DSV4_INTERNAL_H
#define DSV4_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#include "dsv4.h"
#include "k3_st.h"

#define DSV4_FP8_BLOCK 128          /* FP8 block size (weight and activation)   */
#define DSV4_FP4_GROUP 32           /* FP4 scale group                          */
#define DSV4_ACT_GROUP 64           /* KV/compressor act quant group (non-rope) */
#define DSV4_ROPE_DIM 64

/* One expert slot in the bounded LRU cache: the packed scale run then the
 * packed weight run, as they sit contiguously in the shard. */
#define EXPERT_SCALE_RUN 786432u      /* 3 * 262144                             */
#define EXPERT_WEIGHT_RUN 12582912u   /* 3 * 4194304                            */
#define EXPERT_RAW (EXPERT_SCALE_RUN + EXPERT_WEIGHT_RUN)   /* 13,369,344      */
#define EXPERT_SLOT (EXPERT_RAW + 4 * 4096u)

/* ------------------------------------------------------------------ weights */
typedef struct {
    /* norms and small vectors */
    const uint16_t *attn_norm, *ffn_norm;          /* BF16 [4096]               */
    const uint16_t *q_norm, *kv_norm;              /* BF16 [1024], [512]        */
    const float    *attn_sink;                     /* F32 [64]                  */
    const float    *bias;                          /* F32 [256] or NULL (hash)  */
    const int64_t  *tid2eid;                       /* I64 [vocab, topk] or NULL */

    /* hyper-connection */
    const float *hc_attn_fn, *hc_ffn_fn;           /* F32 [24][16384]           */
    const float *hc_attn_scale, *hc_ffn_scale;     /* F32 [3]                   */
    const float *hc_attn_base, *hc_ffn_base;       /* F32 [24]                  */

    /* attention projections (wq_a/wq_b/wkv/wo_b are FP8 qmats; wo_a is FP8 on
     * disk but computed as BF16 — decoded per element with its E8M0 scale, with
     * NO activation quantisation, matching the released reference) */
    const uint8_t  *wq_a, *wq_a_scale;              /* FP8 [1024][4096]         */
    const uint8_t  *wq_b, *wq_b_scale;              /* FP8 [32768][1024]        */
    const uint8_t  *wkv, *wkv_scale;                /* FP8 [512][4096]          */
    const uint16_t *wo_a;                           /* BF16 [8192][4096]        */
    const uint8_t  *wo_a_codes, *wo_a_scale;        /* optional packed cache    */
    const uint8_t  *wo_b, *wo_b_scale;              /* FP8 [4096][8192]         */
    int wo_a_cached;                                 /* owned by model cache     */

    /* main compressor (present when compress_ratio != 0) */
    const float    *ape;                           /* F32 [ratio][coff*512]     */
    const uint16_t *comp_wkv, *comp_wgate;         /* BF16 [coff*512][4096]     */
    const uint16_t *comp_norm;                     /* BF16 [512]                */

    /* indexer (present when compress_ratio == 4) */
    const uint8_t  *idx_wq_b, *idx_wq_b_scale;     /* FP8 [8192][1024]          */
    const uint16_t *idx_weights_proj;              /* BF16 [64][4096]           */
    const float    *idx_ape;                       /* F32 [4][coff*128]         */
    const uint16_t *idx_comp_wkv, *idx_comp_wgate; /* BF16 [coff*128][4096]     */
    const uint16_t *idx_comp_norm;                 /* BF16 [128]                */

    /* ffn */
    const uint16_t *gate_w;                        /* BF16 [256][4096]          */
    const uint8_t  *sh1, *sh3, *sh2;               /* FP8 [2048][4096] etc      */
    const uint8_t  *sh1_s, *sh3_s, *sh2_s;         /* F8_E8M0                   */
    int compress_ratio;
    int is_hash;
} DSV4LayerW;

/* ------------------------------------------------------------------ runtime */
/* Persistent per-layer attention state. Sized by the memory plan. */
typedef struct {
    float  *kv_cache;          /* [win + max_comp] x 512 (fp32 storage)        */
    int     kv_cap;            /* slots in kv_cache (win + max_comp)            */
    /* compressor working state: per-token accumulated kv/score before a block
     * boundary fires. overlap (ratio==4) uses coff=2, so the buffer holds
     * coff*ratio slots of coff*512 floats. The indexer keeps a SEPARATE pair
     * of buffers (coff*128 wide) so the two compressors never clobber each
     * other's running state. */
    float  *kv_state, *score_state;
    float  *idx_kv_state, *idx_score_state;
    int     comp_count;        /* compressed positions produced so far          */
    /* indexer state (ratio==4 layers) */
    float  *idx_cache;         /* [max_comp] x 128                             */
    int     idx_count;
} DSV4LayerRun;

/* Bounded LRU expert cache. Each slot is EXPERT_SLOT bytes: 786,432 scale
 * bytes then 12,582,912 weight bytes, plus alignment padding. */
typedef struct {
    uint8_t *scales;           /* w1,w2,w3 E8M0 scales, 786,432 bytes          */
    uint8_t *weights;          /* w1,w2,w3 packed FP4, 12,582,912 bytes        */
    uint8_t *scales_base;      /* page-aligned allocations (for free)         */
    uint8_t *weights_base;
    int layer, expert;
    int64_t last_use;
    unsigned char segment;      /* 0 empty, 1 probationary, 2 protected       */
} DSV4ExpertSlot;

/* Precomputed per-token frequency for the indexer query/key RoPE. */
typedef struct {
    float *cosv, *sinv;
    int n;
} DSV4RopeFreq;

typedef struct {
    const K3Tensor **tensor;
    int count;
} DSV4LayerPrefetch;

/* Per-expert shard geometry (lazily discovered, see dsv4_model.c). */
typedef struct {
    int shard;
    int64_t scale_off;   /* start of the contiguous w1,w2,w3 scale run          */
    int64_t weight_off;  /* start of the contiguous w1,w2,w3 weight run         */
    int64_t scale_run;   /* byte length of the scale run (from the shapes)     */
    int64_t weight_run;  /* byte length of the weight run (from the shapes)    */
    int valid;
} DSV4ExpertMeta;

typedef struct DSV4ExpertPool DSV4ExpertPool;
typedef struct DSV4IoRing DSV4IoRing;
typedef struct DSV4ContextSnapshot DSV4ContextSnapshot;
typedef struct DSV4DSpark DSV4DSpark;

struct DSV4Model {
    DSV4Config cfg;
    DSV4Options opt;
    DSV4MemoryPlan plan;
    K3St st;

    /* global weights */
    const uint16_t *norm;              /* BF16 [4096]                         */
    const float    *hc_head_fn;        /* F32 [4][16384]                      */
    const float    *hc_head_base;      /* F32 [4]                             */
    const float    *hc_head_scale;     /* F32 [1]                             */
    const K3Tensor *embed_t, *head_t;  /* BF16 [vocab][4096] handles          */

    DSV4LayerRun *run;                 /* n_layers runtime state              */
    uint16_t **wo_a_cache;             /* decoded wo_a for a layer prefix     */
    uint8_t **wo_a_code_cache;         /* packed wo_a for a layer prefix      */
    uint8_t **wo_a_scale_cache;
    DSV4LayerPrefetch *prefetch;        /* non-expert tensors grouped by layer */
    DSV4ExpertSlot *cache;             /* expert_cache_slots slots            */
    DSV4ExpertSlot route_prefetch_temp[DSV4_MAX_TOPK]; /* prediction read slots */
    DSV4ExpertMeta *meta;              /* layers x experts shard geometry     */
    DSV4ExpertPool *expert_pool;        /* persistent expert-miss I/O workers  */
    DSV4IoRing *expert_ring;            /* optional Linux io_uring backend     */
    DSV4DSpark *dspark;                 /* optional three-stage draft runtime  */
    DSV4ContextSnapshot *active_snapshot; /* current speculative transaction   */
    FILE *expert_trace;                 /* optional layer/expert access trace  */
    long expert_trace_records;
    int route_prediction[DSV4_MAX_TOPK];
    unsigned char route_prediction_was_miss[DSV4_MAX_TOPK];
    int route_prediction_count;
    int route_prediction_valid;
    long route_prediction_groups[2];
    long route_prediction_matches[2];
    long route_prediction_predicted_misses[2];
    long route_prediction_actual_misses[2];
    long route_prediction_covered_misses[2];
    long route_prediction_wasted_misses[2];
    long route_prediction_rank_misses[2][DSV4_MAX_TOPK];
    long route_prediction_rank_covered[2][DSV4_MAX_TOPK];
    long route_prediction_rank_wasted[2][DSV4_MAX_TOPK];
    long route_prediction_hist[2][DSV4_MAX_TOPK + 1];
    long route_prefetch_reads;
    long route_prefetch_adopted;
    long route_prefetch_wasted;
    long route_prefetch_unplaced;
    int cache_slots;
    int main_cache_slots;
    int draft_cache_slots;
    int cache_per_layer;
    int cache_slru;
    int cache_arc;
    int cache_lrfu;
    int cache_hash_min;
    int cache_hash_slots;
    int *cache_protected;
    int *cache_arc_target;
    int *cache_ghost_b1;
    int *cache_ghost_b2;
    unsigned char *cache_ghost;
    int64_t *cache_ghost_last;
    int64_t cache_ghost_clock;
    float *cache_frequency;
    int64_t *cache_frequency_at;
    int64_t *cache_layer_clock;
    int wo_a_cache_layers;
    int packed_wo_a;
    int64_t lru_clock;

    /* active buffers */
    float *state;                      /* [hc_mult][hidden]                   */
    float *mixes_buf;                  /* [(2+mult)*mult]                     */
    float *hc_state_buf;               /* [hc_mult][hidden], HC post output   */
    float *head_hidden;                /* [hidden], final HC reduction        */
    void *head_map_base;               /* optional read-only head mmap base   */
    size_t head_map_len;               /* mapped bytes including page prefix  */
    const uint16_t *head_map_rows;      /* head tensor start inside mapping    */
    void **layer_shard_map;             /* optional whole-shard read-only maps */
    size_t *layer_shard_map_len;
    uint8_t *head_buf;                 /* streamed vocab rows (8 MiB)         */
    float *scratch;                    /* reusable MoE activation buffers     */
    float *expert_work;                /* reusable expert workspace           */
    int *route_scratch;                /* reusable MoE route indices          */
    unsigned char *attention_scratch;  /* per-token attention temporary arena */
    size_t attention_scratch_cap;
    size_t attention_scratch_used;
    DSV4RopeFreq rope_cache[3];        /* normal, compressed, block-start     */
    int rope_cache_position[3];

    /* counters */
    long cache_hits, cache_misses, coalesced_loads, batch_prefetched;
    long expert_read_ops;
    long wo_a_cache_hits, wo_a_cache_misses;
    uint64_t expert_bytes_read;
    uint64_t prefetch_bytes;
    long prefetch_calls;
    long forward_steps;
    double time_total, time_embed, time_load, time_layer_io, time_prefetch, time_attn;
    double time_moe, time_expert_read, time_expert_io, time_hc, time_release, time_head;
    int threads;
    int context;
    int profiling;
    int cur_layer;               /* layer currently being computed            */
};

/* ---------------------------------------------------------- helpers ---- */
/* BF16 widen; the real checkpoint stores weights as BF16, norms as BF16,
 * small vectors as F32. */
static inline float bf16f(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return v.f;
}
static inline float f32bf(float f) { return bf16f(dsv4_f32_to_bf16(f)); }

/* BF16 x BF16 GEMV with fp32 accumulation; output optionally rounded to bf16. */
void dsv4_gemv_bf16(float *y, const float *x, const uint16_t *W, int in, int out,
                     int round_out);
void dsv4_gemv_bf16_batch(float *const *y, const float *const *x, int batch,
                          const uint16_t *W, int in, int out, int round_out);

/* FP8 E4M3 GEMV with fp32 accumulation, round_out selects BF16 output. */
void dsv4_gemv_fp8_q(float *y, const float *qx, const float *qscale,
                     const uint8_t *W, const uint8_t *wscale_code,
                     int in, int out, int round_out);
void dsv4_gemv_fp8_batch_q(float *const *y, const float *const *qx,
                           const float *const *qscale, int batch,
                           const uint8_t *W, const uint8_t *wscale_code,
                           int in, int out, int round_out);
void dsv4_gemv_fp8_pair_q(float *y0, float *y1, const float *qx,
                          const float *qscale, const uint8_t *W0,
                          const uint8_t *S0, const uint8_t *W1,
                          const uint8_t *S1, int in, int out);

/* FP4 E2M1 GEMV; x is fp32 already-quantised activation (qx, qscale per 128). */
void dsv4_gemv_fp4_q(float *y, const float *qx, const float *qscale,
                     const uint8_t *W, const uint8_t *wscale_code,
                     int in, int rows);
void dsv4_gemv_fp4_batch_q(float *const *y, const float *const *qx,
                           const float *const *qscale, int batch,
                           const uint8_t *W, const uint8_t *wscale_code,
                           int in, int rows);
void dsv4_gemv_fp4_pair_q(float *y0, float *y1, const float *qx,
                          const float *qscale, const uint8_t *W0,
                          const uint8_t *S0, const uint8_t *W1,
                          const uint8_t *S1, int in, int rows);

/* Sparse-attention index selection, exposed internally for large-k tests. */
void dsv4_indexer_select_topk(const float *score, int n, int k, int offset,
                              int *out);

/* Layer-major forward used by the DSpark verifier. all_logits, when non-NULL,
 * receives [n_tokens][vocab]. capture_hidden receives
 * [n_tokens][n_capture_layers][hidden], where each feature is the BF16-rounded
 * mean of the completed Hyper-Connection lanes after the requested layer. */
int dsv4_prefill_capture(DSV4Model *m, const int *token_ids, int n_tokens,
                         int start_position, float *last_logits,
                         float *all_logits, const int *capture_layers,
                         int n_capture_layers, float *capture_hidden);

/* Bounded speculative transaction. The snapshot owns storage for at most
 * max_positions consecutive writes and can be reused for every verify round. */
DSV4ContextSnapshot *dsv4_context_snapshot_create(DSV4Model *m,
                                                   int max_positions);
int dsv4_context_snapshot_take(DSV4ContextSnapshot *s, int start_position,
                               int n_positions);
int dsv4_context_snapshot_restore(DSV4ContextSnapshot *s);
int dsv4_context_snapshot_commit_prefix(DSV4ContextSnapshot *s,
                                         int n_positions);
int dsv4_expert_cache_set_hash_slots(DSV4Model *m, int hash_slots);
void dsv4_context_snapshot_free(DSV4ContextSnapshot *s);

/* DSpark target-side state. capture_hidden uses the same token-major layout as
 * dsv4_prefill_capture and must contain cfg.dspark_stages target layers. */
int dsv4_dspark_ready(const DSV4Model *m);
int dsv4_dspark_commit_target_hidden(DSV4Model *m, const float *capture_hidden,
                                     int n_tokens, int start_position);
int dsv4_dspark_propose(DSV4Model *m, int anchor_token, int anchor_position,
                        int n_drafts, int *draft_tokens);

/* Act quant (E8M0 power-of-two scales, clamp to 448) + in-place dequant to
 * the caller's buffer (BF16-rounded when round_bf16 is set). group may be 128
 * (general) or 64 (KV non-rope dims). */
void dsv4_act_quant_inplace(float *x, int n, int group, int mode);

/* E4M3 activation quant to grid values (qx) plus power-of-two scales. */
void dsv4_quant_fp8_codes(float *qx, float *qscale, const float *x, int n);

/* Round |v| to the nearest representable FP8 E4M3 value (3-bit mantissa,
 * exponent range [-9, 8], max 448), matching torch's float->e4m3 cast. */
float dsv4_round_e4m3(float v);

/* FP4 act quant (E8M0, clamp to 6) + in-place dequant to BF16. n must be a
 * multiple of 32. */
void dsv4_fp4_act_quant_inplace(float *x, int n, int mode);

/* Normalised Hadamard on n elements (power of two). */
void dsv4_hadamard_inplace(float *x, int n);

/* YaRN/rotary frequencies for one position; frees itself on each call. */
void dsv4_rope_freqs(DSV4RopeFreq *f, int rope_dim, float theta, int yarn,
                     float factor, float beta_fast, float beta_slow, int position,
                     int original_position);
void dsv4_rope_apply_buf(float *x, int heads, int head_dim, const DSV4RopeFreq *f);
void dsv4_rope_apply_buf_inv(float *x, int heads, int head_dim, const DSV4RopeFreq *f);

/* Sinkhorn HC split, matching hc_split_sinkhorn() in the released kernel. */
void dsv4_hc_split(float *pre, float *post, float *comb,
                   const float *mixes, const float *hc_scale,
                   const float *hc_base, int mult, int iters, float eps);

#endif /* DSV4_INTERNAL_H */
