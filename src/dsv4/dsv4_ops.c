/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_ops.c - the numeric kernels of the DeepSeek-V4-Flash-0731 engine.
 *
 * Every routine here reproduces a fixed numeric contract; see include/dsv4/dsv4.h
 * for the exact decode rules. The important ones, restated:
 *
 *   FP8 E4M3:  sign=(code>>7), exp=(code>>3)&15, man=code&7
 *              exp==0: ldexp(man, -9); exp==15&&man==7: NaN; else ldexp(8+man, exp-10)
 *              largest finite value 448.
 *   E8M0:      2^(code-127); code 255 is NaN by spec (mapped to +inf).
 *   FP4 E2M1:  abs values {0, 0.5, 1, 1.5, 2, 3, 4, 6}; bit 3 is the sign;
 *              two nibbles per byte, LOW nibble first.
 *   Activation quant: per 128 elements, amax rounded UP to a power of two,
 *              encoded as an E8M0 exponent; values clamped to [-448, 448].
 *
 * Accumulation order is load bearing: with -ffp-contract=off the compiler cannot
 * fuse multiply-adds, and the GEMV loops accumulate strictly left to right so
 * the same binary gives the same logits on every run and the tiny-model oracle
 * matches the C engine to the documented tolerance.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "dsv4.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ BF16 ---- */
uint16_t dsv4_f32_to_bf16(float f)
{
    union { uint32_t u; float f; } v;
    v.f = f;
    uint32_t u = v.u;
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t rounding_bias = 0x7FFFu + lsb;
    u += rounding_bias;                 /* round to nearest even */
    return (uint16_t)(u >> 16);
}

/* ------------------------------------------------------------------ FP8 ---- */
float dsv4_fp8_e4m3(uint8_t code)
{
    const int sign = (code >> 7) & 1;
    const int exp  = (code >> 3) & 15;
    const int man  = code & 7;
    float val;
    if (exp == 0) {
        val = ldexpf((float)man, -9);
    } else if (exp == 15 && man == 7) {
        val = NAN;
    } else {
        val = ldexpf((float)(8 + man), exp - 10);
    }
    return sign ? -val : val;
}

float dsv4_e8m0(uint8_t code)
{
    if (code == 255) return NAN;   /* NaN by spec, not infinity */
    return ldexpf(1.0f, (int)code - 127);
}

/* 256-entry decode table; built once by the GEMV kernels that need it. */
static float g_tbl[256];
static int g_tbl_init = 0;
static void fp8_table_init(void)
{
    if (g_tbl_init) return;
    for (int i = 0; i < 256; i++) g_tbl[i] = dsv4_fp8_e4m3((uint8_t)i);
    g_tbl_init = 1;
}

/* ----------------------------------------------------------------- quant ---- */
/* round-to-nearest-even on a half-integer grid (the E4M3 3-bit mantissa). */
static float round_half_even(float x)
{
    float r = floorf(x);
    float frac = x - r;
    if (frac > 0.5f) return r + 1.0f;
    if (frac < 0.5f) return r;
    if (fmodf(r, 2.0f) == 0.0f) return r;
    return r + 1.0f;
}

/* Round |v| to the nearest representable FP8 E4M3 value. */
float dsv4_round_e4m3(float v)
{
    if (v == 0.0f) return 0.0f;
    float av = fabsf(v);
    int s = v < 0 ? -1 : 1;
    int e;
    float m = frexpf(av, &e);          /* av = m * 2^e, m in [0.5, 1)          */
    if (e >= -5) {                      /* normal E4M3: exponent e-1 in [-6, 8] */
        float mr = round_half_even(m * 16.0f) / 16.0f;  /* 3-bit mantissa       */
        if (mr >= 1.0f) { mr = 0.5f; e++; }
        if (e - 1 > 8) return (float)s * 448.0f;       /* overflow -> 448       */
        return (float)s * mr * ldexpf(1.0f, e);
    }
    /* Subnormal step is 2^-9. Rounding may cross into the smallest normal
     * value (8 * 2^-9); clamping to 7 * 2^-9 would break that boundary. */
    float vr = round_half_even(av * 512.0f) / 512.0f;
    return (float)s * vr;
}


/* Power-of-two amax exponent: smallest e with 2^e >= amax. amax <= 0 -> 0. */
static int pow2_exp(float amax)
{
    if (!(amax > 0.0f)) return 0;
    int e;
    float m = frexpf(amax, &e);  /* amax = m * 2^e, m in [0.5, 1) */
    if (m == 0.5f) e--;          /* ceil(log2(amax)) */
    return e;
}

void dsv4_quant_fp8_act(float *out, float *scales, const float *x, int n)
{
    const int group = 128;
    for (int g = 0; g * group < n; g++) {
        int base = g * group;
        int len = n - base;
        if (len > group) len = group;
        float amax = 0.0f;
        for (int i = 0; i < len; i++) {
            float a = fabsf(x[base + i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        int e = pow2_exp(amax * (1.0f / 448.0f));
        float scale = ldexpf(1.0f, e);
        scales[g] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i < len; i++) {
            float q = x[base + i] * inv;
            if (q > 448.0f) q = 448.0f;
            else if (q < -448.0f) q = -448.0f;
            out[base + i] = dsv4_round_e4m3(q);
        }
    }
}
/* ------------------------------------------------------------ RMSNorm ---- */
void dsv4_rmsnorm_plain(float *y, const float *x, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * r;
}

void dsv4_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / (float)n + eps);
    if (w) {
        for (int i = 0; i < n; i++) y[i] = x[i] * r * w[i];
    } else {
        for (int i = 0; i < n; i++) y[i] = x[i] * r;
    }
}

/* ------------------------------------------------------- FP8 GEMV -------
 * W is [rows][in] of E4M3 codes; wscale is the decoded per-block scale table,
 * laid out [ceil(rows/128)][ceil(in/128)]. x arrives already decoded to float
 * (activation quantisation happened once in the caller). The weight inner loop
 * is a 256-entry table lookup; the scale product is constant per (row block,
 * col block), so per output row we pre-multiply the whole x row by the row's
 * block scales once and let the inner loop be a pure fused multiply-add.
 */
void dsv4_gemv_fp8(float *y, const float *x, const uint8_t *W, const float *wscale,
                   int in, int rows, int block)
{
    fp8_table_init();
    const int rblocks = (rows + block - 1) / block;
    const int cblocks = (in + block - 1) / block;
    memset(y, 0, (size_t)rows * sizeof(float));
    for (int rb = 0; rb < rblocks; rb++) {
        int r0 = rb * block;
        int r1 = r0 + block;
        if (r1 > rows) r1 = rows;
        /* row-block scale for each column block, applied to x once per row */
        float *xs = (float *)malloc((size_t)in * sizeof(float));
        if (!xs) { fprintf(stderr, "dsv4: OOM in gemv_fp8\n"); exit(1); }
        for (int cb = 0; cb < cblocks; cb++) {
            int c0 = cb * block;
            int c1 = c0 + block;
            if (c1 > in) c1 = in;
            float s = wscale[(size_t)rb * cblocks + cb];
            for (int c = c0; c < c1; c++) xs[c] = x[c] * s;
        }
        for (int r = r0; r < r1; r++) {
            float acc = 0.0f;
            const uint8_t *wrow = W + (size_t)r * in;
            for (int c = 0; c < in; c++) {
                acc += g_tbl[wrow[c]] * xs[c];
            }
            y[r] = acc;
        }
        free(xs);
    }
}

/* FP8 decode-table GEMV, single pass: builds the per-block scale product into
 * the activation once for ALL row blocks (block scale product row block, col
 * block), then loops rows. Memory: one float per input element per row block
 * would be 8 GiB traffic for the big matrices; instead decode the weight table
 * per row block and keep x fixed. This is the shape the real model uses. */
void dsv4_gemv_fp8_tbl(float *y, const float *x, const uint8_t *W, const float *wscale,
                       int in, int rows, int block)
{
    fp8_table_init();
    const int rblocks = (rows + block - 1) / block;
    const int cblocks = (in + block - 1) / block;
    for (int rb = 0; rb < rblocks; rb++) {
        int r0 = rb * block;
        int r1 = r0 + block;
        if (r1 > rows) r1 = rows;
        for (int r = r0; r < r1; r++) {
            float acc = 0.0f;
            const uint8_t *wrow = W + (size_t)r * in;
            for (int cb = 0; cb < cblocks; cb++) {
                int c0 = cb * block;
                int c1 = c0 + block;
                if (c1 > in) c1 = in;
                float s = wscale[(size_t)rb * cblocks + cb];
                for (int c = c0; c < c1; c++) {
                    acc += g_tbl[wrow[c]] * x[c] * s;
                }
            }
            y[r] = acc;
        }
    }
}

/* ------------------------------------------------------- FP4 GEMV -------
 * W is [rows][pcols] with pcols = ceil(in/2), two E2M1 nibbles per byte (low
 * nibble first). wscale is the decoded per-row per-32-column scale table,
 * [rows][ceil(in/32)]. The activation was already FP8-quantised by the caller
 * into qx (floats, clamped) and qscale (per-128-group). To keep the inner loop
 * a table lookup plus one multiply, we pre-scale qx by the 128-group scale so
 * the final accumulate is qx_decoded * wdecoded * wscale (a per-row, per-32
 * block constant). */
void dsv4_gemv_fp4(float *y, const float *qx, const float *qscale,
                   const uint8_t *W, const float *wscale,
                   int in, int rows, int group)
{
    fp8_table_init();
    const int pcols = (in + 1) / 2;
    const int gcols = (in + group - 1) / group;
    const int agroups = (in + 127) / 128;

    /* decode activation once: xd[c] = qx[c] * qscale[c/128] */
    float *xd = (float *)malloc((size_t)in * sizeof(float));
    if (!xd) { fprintf(stderr, "dsv4: OOM in gemv_fp4\n"); exit(1); }
    for (int g = 0; g < agroups; g++) {
        int c0 = g * 128, c1 = c0 + 128;
        if (c1 > in) c1 = in;
        float s = qscale[g];
        for (int c = c0; c < c1; c++) xd[c] = qx[c] * s;
    }

    static const float fp4_abs[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        const uint8_t *wrow = W + (size_t)r * pcols;
        /* per-row: multiply each 32-column block by its scale once */
        for (int c = 0; c < in; c++) {
            uint8_t b = wrow[c >> 1];
            uint8_t nib = (c & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xF);
            float wv = fp4_abs[nib & 7];
            if (nib & 8) wv = -wv;
            acc += wv * xd[c] * wscale[(size_t)r * gcols + c / group];
        }
        y[r] = acc;
    }
    free(xd);
}

/* ------------------------------------------------------------- RoPE -------
 * Rotary embedding on the last rope_dim elements of every head. YaRN keeps the
 * original frequencies below the beta_fast correction dimension, transitions
 * linearly, then applies 1/factor above the beta_slow correction dimension. */
typedef struct {
    float *cosv, *sinv;    /* [rope_dim/2] */
    int n;
} RopeFreq;

static void rope_freqs(RopeFreq *f, int rope_dim, float theta, int yarn,
                       float factor, float beta_fast, float beta_slow,
                       int original_position, int position)
{
    f->n = rope_dim / 2;
    f->cosv = (float *)malloc((size_t)f->n * sizeof(float));
    f->sinv = (float *)malloc((size_t)f->n * sizeof(float));
    if (!f->cosv || !f->sinv) { fprintf(stderr, "dsv4: OOM in rope\n"); exit(1); }
    for (int i = 0; i < f->n; i++) {
        float inv = 1.0f / powf(theta, (float)(2 * i) / (float)rope_dim);
        if (yarn) {
            float low = floorf((float)rope_dim
                               * logf((float)original_position /
                                      (beta_fast * 2.0f * (float)M_PI))
                               / (2.0f * logf(theta)));
            float high = ceilf((float)rope_dim
                               * logf((float)original_position /
                                      (beta_slow * 2.0f * (float)M_PI))
                               / (2.0f * logf(theta)));
            if (low < 0) low = 0;
            if (high > (float)(rope_dim - 1)) high = (float)(rope_dim - 1);
            if (high == low) high += 0.001f;
            float ramp = ((float)i - low) / (high - low);
            if (ramp < 0.0f) ramp = 0.0f;
            if (ramp > 1.0f) ramp = 1.0f;
            float inv_interp = inv / factor;
            inv = inv_interp * ramp + inv * (1.0f - ramp);
        }
        float angle = (float)position * inv;
        f->cosv[i] = cosf(angle);
        f->sinv[i] = sinf(angle);
    }
}

static void rope_rotate(float *x, int heads, int head_dim, int rope_dim,
                        const RopeFreq *f, int inverse)
{
    /* the rope section is the LAST rope_dim columns of each head */
    const int start = head_dim - rope_dim;
    for (int h = 0; h < heads; h++) {
        float *p = x + (size_t)h * head_dim + start;
        for (int i = 0; i < f->n; i++) {
            float x0 = p[2 * i], x1 = p[2 * i + 1];
            float c = f->cosv[i], s = f->sinv[i];
            if (inverse) {
                p[2 * i]     = x0 * c + x1 * s;
                p[2 * i + 1] = -x0 * s + x1 * c;
            } else {
                p[2 * i]     = x0 * c - x1 * s;
                p[2 * i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

void dsv4_rope_apply(float *x, int heads, int head_dim, int rope_dim,
                     int position, float theta, int yarn,
                     float factor, float beta_fast, float beta_slow,
                     int original_position)
{
    RopeFreq f;
    rope_freqs(&f, rope_dim, theta, yarn, factor, beta_fast, beta_slow,
               original_position, position);
    rope_rotate(x, heads, head_dim, rope_dim, &f, 0);
    free(f.cosv);
    free(f.sinv);
}

void dsv4_rope_apply_inv(float *x, int heads, int head_dim, int rope_dim,
                         int position, float theta, int yarn,
                         float factor, float beta_fast, float beta_slow,
                         int original_position)
{
    RopeFreq f;
    rope_freqs(&f, rope_dim, theta, yarn, factor, beta_fast, beta_slow,
               original_position, position);
    rope_rotate(x, heads, head_dim, rope_dim, &f, 1);
    free(f.cosv);
    free(f.sinv);
}

/* ---------------------------------------------------------- Hadamard ---- */
/* Normalised in-place Walsh-Hadamard transform, n a power of two. */
void dsv4_hadamard(float *x, int n)
{
    if (n < 2) return;
    float *t = (float *)malloc((size_t)n * sizeof(float));
    if (!t) { fprintf(stderr, "dsv4: OOM in hadamard\n"); exit(1); }
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i++) t[i] = x[i];
        for (int i = 0; i < n; i += len * 2) {
            for (int j = 0; j < len; j++) {
                float a = t[i + j], b = t[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    float inv = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= inv;
    free(t);
}

/* ------------------------------------------------- Hyper-Connection ---- */
/* Numerically stable sigmoid: for x >= 0 use 1/(1+exp(-x)), for x < 0 use
 * exp(x)/(1+exp(x)). Matches the released hc_pre. */
static float sigmoidf_stable(float x)
{
    if (x >= 0.0f) {
        float e = expf(-x);
        return 1.0f / (1.0f + e);
    }
    float e = expf(x);
    return e / (1.0f + e);
}

/* Sinkhorn HC split replicating hc_split_sinkhorn() in the released kernel:
 * mixes layout [pre(hc) | post(hc) | comb(hc*hc)]; pre = sigmoid(..)+eps;
 * post = 2*sigmoid(..); comb = softmax rows + eps, one column normalisation,
 * then (iters-1) alternating row/column normalisations (final = column). */
void dsv4_hc_split(float *pre, float *post, float *comb,
                   const float *mixes, const float *hc_scale,
                   const float *hc_base, int mult, int iters, float eps)
{
    const int m = mult;
    for (int i = 0; i < m; i++) {
        float z = mixes[i] * hc_scale[0] + hc_base[i];
        pre[i] = sigmoidf_stable(z) + eps;
        float zp = mixes[m + i] * hc_scale[1] + hc_base[m + i];
        post[i] = 2.0f * sigmoidf_stable(zp);
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            comb[i * m + j] = mixes[2 * m + i * m + j] * hc_scale[2] + hc_base[2 * m + i * m + j];
    /* softmax rows */
    for (int i = 0; i < m; i++) {
        float mx = comb[i * m];
        for (int j = 1; j < m; j++) if (comb[i * m + j] > mx) mx = comb[i * m + j];
        float sum = 0.0f;
        for (int j = 0; j < m; j++) sum += expf(comb[i * m + j] - mx);
        for (int j = 0; j < m; j++) comb[i * m + j] = expf(comb[i * m + j] - mx) / sum + eps;
    }
    /* one column normalisation */
    for (int j = 0; j < m; j++) {
        float sum = 0.0f;
        for (int i = 0; i < m; i++) sum += comb[i * m + j];
        for (int i = 0; i < m; i++) comb[i * m + j] /= (sum + eps);
    }
    for (int it = 0; it < iters - 1; it++) {
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < m; j++) sum += comb[i * m + j];
            for (int j = 0; j < m; j++) comb[i * m + j] /= (sum + eps);
        }
        for (int j = 0; j < m; j++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) sum += comb[i * m + j];
            for (int i = 0; i < m; i++) comb[i * m + j] /= (sum + eps);
        }
    }
    if (getenv("DSV4_DEBUG_HC")) {
        fprintf(stderr, "hc_split iters=%d rows=", iters);
        for (int i = 0; i < m; i++) { float s = 0; for (int j = 0; j < m; j++) s += comb[i*m+j]; fprintf(stderr, "%.6f ", s); }
        fprintf(stderr, "cols=");
        for (int j = 0; j < m; j++) { float s = 0; for (int i = 0; i < m; i++) s += comb[i*m+j]; fprintf(stderr, "%.6f ", s); }
        fprintf(stderr, "\n");
    }
}

/* ----------------------------------------------------------- Router ---- */
/* sqrt(softplus(z)) with a numerically tame softplus for large negative z. */
static float sqrt_softplus(float z)
{
    if (z > 20.0f) return sqrtf(z);
    return sqrtf(log1pf(expf(z)));
}

/* z_e = BF16 gate row dot x. gate is a BF16 matrix [n_experts][hidden]. */
void dsv4_router(int *idx, float *wt, const float *x, const float *gate,
                 const float *bias, const int64_t *tid2eid, int token_id,
                 int hidden, int n_experts, int topk, int hash_layer,
                 float route_scale, float rms_eps)
{
    const int m = n_experts;
    float *score = (float *)malloc((size_t)m * sizeof(float));
    float *choice = (float *)malloc((size_t)m * sizeof(float));
    int *order = (int *)malloc((size_t)m * sizeof(int));
    if (!score || !choice || !order) { fprintf(stderr, "dsv4: OOM in router\n"); exit(1); }

    if (hash_layer) {
        /* first n_hash_layers: the expert list comes straight from the token */
        for (int k = 0; k < topk; k++) {
            int64_t e = tid2eid[(size_t)token_id * topk + k];
            idx[k] = (int)e;
            wt[k] = 0.0f;      /* filled below with the score-normalised value */
        }
        /* scores for the chosen experts only */
        for (int k = 0; k < topk; k++) {
            int e = idx[k];
            const uint16_t *grow = (const uint16_t *)gate + (size_t)e * hidden;
            float z = 0.0f;
            for (int i = 0; i < hidden; i++) z += dsv4_bf16_to_f32(grow[i]) * x[i];
            score[e] = sqrt_softplus(z);
            choice[e] = score[e] + (bias ? bias[e] : 0.0f);
        }
    } else {
        for (int e = 0; e < m; e++) {
            const uint16_t *grow = (const uint16_t *)gate + (size_t)e * hidden;
            float z = 0.0f;
            for (int i = 0; i < hidden; i++) z += dsv4_bf16_to_f32(grow[i]) * x[i];
            score[e] = sqrt_softplus(z);
            choice[e] = score[e] + (bias ? bias[e] : 0.0f);
        }
    }

    /* selection by choice */
    for (int k = 0; k < topk; k++) {
        int best = -1;
        float bv = -INFINITY;
        for (int e = 0; e < m; e++) {
            int used = 0;
            for (int t = 0; t < k; t++) if (order[t] == e) { used = 1; break; }
            if (!used && choice[e] > bv) { bv = choice[e]; best = e; }
        }
        order[k] = best;
    }

    if (hash_layer) {
        /* hash layers: experts are already fixed; keep their choice order */
        for (int k = 0; k < topk; k++) order[k] = idx[k];
    }

    /* combining weights from the UNBIASED scores */
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) sum += score[order[k]];
    float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (int k = 0; k < topk; k++) wt[k] = score[order[k]] * inv * route_scale;

    /* sort by expert id ascending, carrying the weights */
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < topk; j++) {
            if (order[j] < order[i]) {
                int ti = order[i]; order[i] = order[j]; order[j] = ti;
                float tw = wt[i]; wt[i] = wt[j]; wt[j] = tw;
            }
        }
    }
    for (int k = 0; k < topk; k++) idx[k] = order[k];

    free(score); free(choice); free(order);
}
