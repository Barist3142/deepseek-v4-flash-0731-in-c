/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_model.c - the DeepSeek-V4-Flash-0731 forward pass.
 *
 * Data flow per token (matching the released inference/model.py at revision
 * f981a343464c25f82b901e5882716b3b2fa514de, statement for statement):
 *
 *   embed row -> 4-way Hyper-Connection state
 *   per layer:
 *     HC pre-reduce (attn) -> RMSNorm -> compressed/sliding MLA -> HC post-expand
 *     HC pre-reduce (ffn)  -> RMSNorm -> router + 6 FP4 experts + shared FP8
 *                                          -> HC post-expand
 *   HC head reduce -> RMSNorm -> streamed vocab head -> argmax
 *
 * Prefill is layer-major so each layer bundle is streamed once for the whole
 * prompt. Positions remain ordered inside every layer because attention,
 * compressor state and KV writes depend on earlier positions.
 *
 * All intermediate tensors follow the released dtypes: BF16 at every module
 * boundary, FP32 accumulation inside the kernels, and the quantised KV path
 * (FP8 act quant with power-of-two E8M0 scales, in-place dequant back to BF16).
 */
#define _DEFAULT_SOURCE
#define _POSIX_C_SOURCE 200809L
#include "k3_portable_io.h"
#include <errno.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/syscall.h>
#endif
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#include "dsv4.h"
#include "dsv4_internal.h"

static int expert_pool_init(DSV4Model *m);
static void expert_pool_close(DSV4Model *m);
static int expert_ring_init(DSV4Model *m);
static void expert_ring_close(DSV4Model *m);

static int tensor_natural_alignment(K3Dtype dtype)
{
    if (dtype == K3_DT_I64) return 8;
    if (dtype == K3_DT_F32) return 4;
    if (dtype == K3_DT_BF16 || dtype == K3_DT_F16) return 2;
    return 1;
}

static int map_vocab_head(DSV4Model *m)
{
    if (!m->head_t || getenv("DSV4_NO_HEAD_MMAP")) return 0;

    long page_size = sysconf(_SC_PAGESIZE);
    int alignment = tensor_natural_alignment(m->head_t->dtype);
    if (page_size <= 0 || m->head_t->off < 0 || m->head_t->nbytes <= 0 ||
        m->head_t->off % alignment != 0) return 0;

    int64_t map_off = (m->head_t->off / page_size) * page_size;
    int64_t prefix = m->head_t->off - map_off;
    uint64_t map_len = (uint64_t)prefix + (uint64_t)m->head_t->nbytes;
    if (map_len > SIZE_MAX) return 0;

    void *base = mmap(NULL, (size_t)map_len, PROT_READ, MAP_PRIVATE,
                      m->st.fd[m->head_t->shard], (off_t)map_off);
    if (base == MAP_FAILED) return 0;

    m->head_map_base = base;
    m->head_map_len = (size_t)map_len;
    m->head_map_rows = (const uint16_t *)((const uint8_t *)base + prefix);
    return 1;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ================================================================ kernels == */

static float fp8_decode_table[256];
static float e8m0_decode_table[256];
static float wo_a_bf16_decode_table[256 * 256];
static int decode_tables_ready;

#ifdef _OPENMP
static int dsv4_gemv_threads(void)
{
    const int fallback = omp_get_max_threads();
    const char *value = getenv("DSV4_GEMV_THREADS");
    if (!value || !value[0]) return fallback;

    char *end = NULL;
    long requested = strtol(value, &end, 10);
    if (end == value || *end != '\0' || requested < 1 || requested > fallback)
        return fallback;
    return (int)requested;
}
#endif

static void init_decode_tables(void)
{
#ifdef _OPENMP
    #pragma omp critical(dsv4_decode_tables)
#endif
    {
        if (!decode_tables_ready) {
            for (int i = 0; i < 256; i++) {
                fp8_decode_table[i] = dsv4_fp8_e4m3((uint8_t)i);
                e8m0_decode_table[i] = dsv4_e8m0((uint8_t)i);
            }
            for (int scale = 0; scale < 256; scale++) {
                for (int code = 0; code < 256; code++) {
                    wo_a_bf16_decode_table[(scale << 8) | code] =
                        f32bf(fp8_decode_table[code] *
                              e8m0_decode_table[scale]);
                }
            }
            decode_tables_ready = 1;
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
/* Gather one packed-FP4 byte from each of eight row-major rows.  Keeping the
 * bytes packed lets the decoder handle all eight output rows without scalar
 * float-table loads. */
static inline __m128i load_u8_8rows(const uint8_t *base, size_t row_stride)
{
    uint64_t packed =
        (uint64_t)base[0 * row_stride] |
        (uint64_t)base[1 * row_stride] << 8 |
        (uint64_t)base[2 * row_stride] << 16 |
        (uint64_t)base[3 * row_stride] << 24 |
        (uint64_t)base[4 * row_stride] << 32 |
        (uint64_t)base[5 * row_stride] << 40 |
        (uint64_t)base[6 * row_stride] << 48 |
        (uint64_t)base[7 * row_stride] << 56;
    return _mm_cvtsi64_si128((long long)packed);
}

/* Decode eight FP4 nibbles exactly.  The byte LUT stores magnitudes in units
 * of 0.5; all conversions and the 0.5 multiply are exact for the FP4 values.
 * The sign is applied as a bit, preserving negative zero. */
static inline __m256 fp4_decode_8rows(__m128i nibble)
{
    const __m128i magnitude_lut = _mm_setr_epi8(
        0, 1, 2, 3, 4, 6, 8, 12, 0, 1, 2, 3, 4, 6, 8, 12);
    __m128i magnitude8 = _mm_shuffle_epi8(magnitude_lut, nibble);
    __m256 magnitude = _mm256_mul_ps(
        _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(magnitude8)),
        _mm256_set1_ps(0.5f));
    __m256i sign = _mm256_slli_epi32(
        _mm256_cvtepu8_epi32(
            _mm_and_si128(nibble, _mm_set1_epi8(8))), 28);
    return _mm256_xor_ps(magnitude, _mm256_castsi256_ps(sign));
}

static inline void fp4_decode_byte_8rows(__m128i packed, __m256 *lo,
                                         __m256 *hi)
{
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    *lo = fp4_decode_8rows(_mm_and_si128(packed, nibble_mask));
    *hi = fp4_decode_8rows(
        _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask));
}

/* Transpose eight row-major 16-byte runs into eight pairs of columns.  Each
 * output contains one eight-row column in its low 64 bits and the following
 * column in its high 64 bits. */
static inline void transpose_u8_8x16(const uint8_t *base, size_t row_stride,
                                     __m128i columns[8])
{
    __m128i r0 = _mm_loadu_si128((const __m128i *)(base + 0 * row_stride));
    __m128i r1 = _mm_loadu_si128((const __m128i *)(base + 1 * row_stride));
    __m128i r2 = _mm_loadu_si128((const __m128i *)(base + 2 * row_stride));
    __m128i r3 = _mm_loadu_si128((const __m128i *)(base + 3 * row_stride));
    __m128i r4 = _mm_loadu_si128((const __m128i *)(base + 4 * row_stride));
    __m128i r5 = _mm_loadu_si128((const __m128i *)(base + 5 * row_stride));
    __m128i r6 = _mm_loadu_si128((const __m128i *)(base + 6 * row_stride));
    __m128i r7 = _mm_loadu_si128((const __m128i *)(base + 7 * row_stride));
    __m128i a0 = _mm_unpacklo_epi8(r0, r1);
    __m128i a1 = _mm_unpackhi_epi8(r0, r1);
    __m128i a2 = _mm_unpacklo_epi8(r2, r3);
    __m128i a3 = _mm_unpackhi_epi8(r2, r3);
    __m128i a4 = _mm_unpacklo_epi8(r4, r5);
    __m128i a5 = _mm_unpackhi_epi8(r4, r5);
    __m128i a6 = _mm_unpacklo_epi8(r6, r7);
    __m128i a7 = _mm_unpackhi_epi8(r6, r7);
    __m128i b0 = _mm_unpacklo_epi16(a0, a2);
    __m128i b1 = _mm_unpackhi_epi16(a0, a2);
    __m128i b2 = _mm_unpacklo_epi16(a4, a6);
    __m128i b3 = _mm_unpackhi_epi16(a4, a6);
    __m128i b4 = _mm_unpacklo_epi16(a1, a3);
    __m128i b5 = _mm_unpackhi_epi16(a1, a3);
    __m128i b6 = _mm_unpacklo_epi16(a5, a7);
    __m128i b7 = _mm_unpackhi_epi16(a5, a7);
    columns[0] = _mm_unpacklo_epi32(b0, b2);
    columns[1] = _mm_unpackhi_epi32(b0, b2);
    columns[2] = _mm_unpacklo_epi32(b1, b3);
    columns[3] = _mm_unpackhi_epi32(b1, b3);
    columns[4] = _mm_unpacklo_epi32(b4, b6);
    columns[5] = _mm_unpackhi_epi32(b4, b6);
    columns[6] = _mm_unpacklo_epi32(b5, b7);
    columns[7] = _mm_unpackhi_epi32(b5, b7);
}

/* Convert eight E4M3 codes to their exact FP32 values without scalar table
 * loads.  Normal values map directly into FP32 exponent/mantissa fields;
 * subnormals are small integers times 2^-9. */
static inline __m256 fp8_decode_8rows(__m128i packed)
{
    __m256i code = _mm256_cvtepu8_epi32(packed);
    __m256i sign = _mm256_slli_epi32(
        _mm256_and_si256(code, _mm256_set1_epi32(0x80)), 24);
    __m256i absolute = _mm256_and_si256(code, _mm256_set1_epi32(0x7f));
    /* For every normal E4M3FN value, the seven magnitude bits are already
     * exponent:mantissa in the same order as FP32. Shift them into place and
     * add the exponent-bias delta (127 - 7 = 120). */
    __m256i normal_bits = _mm256_or_si256(
        sign, _mm256_add_epi32(_mm256_slli_epi32(absolute, 20),
                               _mm256_set1_epi32(120 << 23)));
    __m256 normal = _mm256_castsi256_ps(normal_bits);
    __m256 subnormal = _mm256_mul_ps(
        _mm256_cvtepi32_ps(absolute), _mm256_set1_ps(1.0f / 512.0f));
    subnormal = _mm256_xor_ps(subnormal, _mm256_castsi256_ps(sign));
    __m256 value = _mm256_blendv_ps(
        normal, subnormal,
        _mm256_castsi256_ps(_mm256_cmpgt_epi32(
            _mm256_set1_epi32(8), absolute)));
    __m256i nan_mask = _mm256_cmpeq_epi32(
        absolute, _mm256_set1_epi32(0x7f));
    __m256 nan_value = _mm256_castsi256_ps(
        _mm256_or_si256(sign, _mm256_set1_epi32(0x7fc00000)));
    return _mm256_blendv_ps(value, nan_value, _mm256_castsi256_ps(nan_mask));
}

static inline __m256 fp8_lookup_8rows(__m128i packed)
{
    return _mm256_i32gather_ps(
        fp8_decode_table, _mm256_cvtepu8_epi32(packed), 4);
}
#endif

/* BF16 GEMV: y[r] = sum_c W[r,c] * x[c], fp32 accumulation. */
void dsv4_gemv_bf16(float *y, const float *x, const uint16_t *W, int in, int out,
                    int round_out)
{
#if defined(__AVX2__) && defined(__FMA__)
    const int simd_rows = (getenv("DSV4_NO_SIMD") || getenv("DSV4_NO_BF16_SIMD"))
                        ? 0 : out & ~7;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r0 = 0; r0 < simd_rows; r0 += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int c = 0; c < in; c++) {
            __m256 wv = _mm256_set_ps(
                bf16f(W[(size_t)(r0 + 7) * in + c]),
                bf16f(W[(size_t)(r0 + 6) * in + c]),
                bf16f(W[(size_t)(r0 + 5) * in + c]),
                bf16f(W[(size_t)(r0 + 4) * in + c]),
                bf16f(W[(size_t)(r0 + 3) * in + c]),
                bf16f(W[(size_t)(r0 + 2) * in + c]),
                bf16f(W[(size_t)(r0 + 1) * in + c]),
                bf16f(W[(size_t)r0 * in + c]));
            acc = _mm256_fmadd_ps(wv, _mm256_set1_ps(x[c]), acc);
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int lane = 0; lane < 8; lane++)
            y[r0 + lane] = round_out ? f32bf(tmp[lane]) : tmp[lane];
    }
    if (simd_rows == out) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = simd_rows; r < out; r++) {
#else
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = 0; r < out; r++) {
#endif
        const uint16_t *wrow = W + (size_t)r * in;
        float acc = 0.0f;
        for (int c = 0; c < in; c++) acc = fmaf(bf16f(wrow[c]), x[c], acc);
        y[r] = round_out ? f32bf(acc) : acc;
    }
}

/* Apply one BF16 matrix to several independent activations. Weight conversion
 * is shared, while every token retains the scalar kernel's column order. */
void dsv4_gemv_bf16_batch(float *const *y, const float *const *x, int batch,
                          const uint16_t *W, int in, int out, int round_out)
{
    if (batch <= 0) return;
    if (batch == 1) {
        dsv4_gemv_bf16(y[0], x[0], W, in, out, round_out);
        return;
    }
    if (batch > DSV4_MAX_TOPK + 1) {
        fprintf(stderr, "dsv4: BF16 batch %d exceeds %d\n", batch,
                DSV4_MAX_TOPK + 1);
        exit(1);
    }
#if defined(__AVX2__) && defined(__FMA__)
    const int simd_rows =
        (getenv("DSV4_NO_SIMD") || getenv("DSV4_NO_BF16_SIMD"))
        ? 0 : out & ~7;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r0 = 0; r0 < simd_rows; r0 += 8) {
        __m256 acc[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < batch; t++) acc[t] = _mm256_setzero_ps();
        for (int c = 0; c < in; c++) {
            __m256 wv = _mm256_set_ps(
                bf16f(W[(size_t)(r0 + 7) * in + c]),
                bf16f(W[(size_t)(r0 + 6) * in + c]),
                bf16f(W[(size_t)(r0 + 5) * in + c]),
                bf16f(W[(size_t)(r0 + 4) * in + c]),
                bf16f(W[(size_t)(r0 + 3) * in + c]),
                bf16f(W[(size_t)(r0 + 2) * in + c]),
                bf16f(W[(size_t)(r0 + 1) * in + c]),
                bf16f(W[(size_t)r0 * in + c]));
            for (int t = 0; t < batch; t++)
                acc[t] = _mm256_fmadd_ps(
                    wv, _mm256_set1_ps(x[t][c]), acc[t]);
        }
        for (int t = 0; t < batch; t++) {
            float tmp[8];
            _mm256_storeu_ps(tmp, acc[t]);
            for (int lane = 0; lane < 8; lane++)
                y[t][r0 + lane] = round_out ? f32bf(tmp[lane]) : tmp[lane];
        }
    }
    if (simd_rows == out) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = simd_rows; r < out; r++) {
#else
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = 0; r < out; r++) {
#endif
        const uint16_t *wrow = W + (size_t)r * in;
        float acc[DSV4_MAX_TOPK + 1] = {0};
        for (int c = 0; c < in; c++) {
            float weight = bf16f(wrow[c]);
            for (int t = 0; t < batch; t++)
                acc[t] = fmaf(weight, x[t][c], acc[t]);
        }
        for (int t = 0; t < batch; t++)
            y[t][r] = round_out ? f32bf(acc[t]) : acc[t];
    }
}

/* Block-diagonal BF16 GEMV used by wo_a. Each group has its own input slice
 * and output rows, but all groups share one OpenMP region. */
static void gemv_bf16_grouped(float *y, const float *x, const uint16_t *W,
                              int groups, int in_per_group, int out_per_group)
{
    const int blocks_per_group = (out_per_group + 7) / 8;
    const int tasks = groups * blocks_per_group;
#if defined(__AVX2__) && defined(__FMA__)
    const int use_simd = !getenv("DSV4_NO_SIMD") &&
                         !getenv("DSV4_NO_BF16_SIMD");
#endif
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int task = 0; task < tasks; task++) {
        int group = task / blocks_per_group;
        int local_r = (task % blocks_per_group) * 8;
        int lanes = out_per_group - local_r;
        if (lanes > 8) lanes = 8;
        const float *xg = x + (size_t)group * in_per_group;
        const uint16_t *wg = W +
            ((size_t)group * out_per_group + local_r) * in_per_group;
        float *yg = y + (size_t)group * out_per_group + local_r;
#if defined(__AVX2__) && defined(__FMA__)
        if (use_simd && lanes == 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int c = 0; c < in_per_group; c++) {
                __m256 wv = _mm256_set_ps(
                    bf16f(wg[(size_t)7 * in_per_group + c]),
                    bf16f(wg[(size_t)6 * in_per_group + c]),
                    bf16f(wg[(size_t)5 * in_per_group + c]),
                    bf16f(wg[(size_t)4 * in_per_group + c]),
                    bf16f(wg[(size_t)3 * in_per_group + c]),
                    bf16f(wg[(size_t)2 * in_per_group + c]),
                    bf16f(wg[(size_t)1 * in_per_group + c]),
                    bf16f(wg[c]));
                acc = _mm256_fmadd_ps(wv, _mm256_set1_ps(xg[c]), acc);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, acc);
            for (int lane = 0; lane < 8; lane++) yg[lane] = f32bf(tmp[lane]);
            continue;
        }
#endif
        for (int lane = 0; lane < lanes; lane++) {
            float acc = 0.0f;
            const uint16_t *row = wg + (size_t)lane * in_per_group;
            for (int c = 0; c < in_per_group; c++)
                acc = fmaf(bf16f(row[c]), xg[c], acc);
            yg[lane] = f32bf(acc);
        }
    }
}

static void gemv_bf16_grouped_batch(float *const *y,
                                    const float *const *x, int batch,
                                    const uint16_t *W, int groups,
                                    int in_per_group, int out_per_group)
{
    if (batch == 1) {
        gemv_bf16_grouped(y[0], x[0], W, groups, in_per_group,
                          out_per_group);
        return;
    }
    const int blocks_per_group = (out_per_group + 7) / 8;
    const int tasks = groups * blocks_per_group;
#if defined(__AVX2__) && defined(__FMA__)
    const int use_simd = !getenv("DSV4_NO_SIMD") &&
                         !getenv("DSV4_NO_BF16_SIMD");
#endif
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int task = 0; task < tasks; task++) {
        int group = task / blocks_per_group;
        int local_r = (task % blocks_per_group) * 8;
        int lanes = out_per_group - local_r;
        if (lanes > 8) lanes = 8;
        const uint16_t *wg = W +
            ((size_t)group * out_per_group + local_r) * in_per_group;
#if defined(__AVX2__) && defined(__FMA__)
        if (use_simd && lanes == 8) {
            __m256 acc[DSV4_MAX_TOPK + 1];
            for (int t = 0; t < batch; t++) acc[t] = _mm256_setzero_ps();
            for (int c = 0; c < in_per_group; c++) {
                __m256 wv = _mm256_set_ps(
                    bf16f(wg[(size_t)7 * in_per_group + c]),
                    bf16f(wg[(size_t)6 * in_per_group + c]),
                    bf16f(wg[(size_t)5 * in_per_group + c]),
                    bf16f(wg[(size_t)4 * in_per_group + c]),
                    bf16f(wg[(size_t)3 * in_per_group + c]),
                    bf16f(wg[(size_t)2 * in_per_group + c]),
                    bf16f(wg[(size_t)1 * in_per_group + c]),
                    bf16f(wg[c]));
                for (int t = 0; t < batch; t++) {
                    const float *xg = x[t] + (size_t)group * in_per_group;
                    acc[t] = _mm256_fmadd_ps(
                        wv, _mm256_set1_ps(xg[c]), acc[t]);
                }
            }
            for (int t = 0; t < batch; t++) {
                float tmp[8];
                float *yg = y[t] + (size_t)group * out_per_group + local_r;
                _mm256_storeu_ps(tmp, acc[t]);
                for (int lane = 0; lane < 8; lane++)
                    yg[lane] = f32bf(tmp[lane]);
            }
            continue;
        }
#endif
        for (int lane = 0; lane < lanes; lane++) {
            const uint16_t *row = wg + (size_t)lane * in_per_group;
            float acc[DSV4_MAX_TOPK + 1] = {0};
            for (int c = 0; c < in_per_group; c++) {
                float weight = bf16f(row[c]);
                for (int t = 0; t < batch; t++) {
                    const float *xg = x[t] + (size_t)group * in_per_group;
                    acc[t] = fmaf(weight, xg[c], acc[t]);
                }
            }
            for (int t = 0; t < batch; t++)
                y[t][(size_t)group * out_per_group + local_r + lane] =
                    f32bf(acc[t]);
        }
    }
}

/* wo_a is FP8 on disk but its dequantised weights are rounded to BF16 before
 * multiplication. Decode that exact BF16 value on demand so a short token
 * batch can share both the packed bytes and the conversion. */
static void gemv_packed_wo_a_grouped_batch(
    float *const *y, const float *const *x, int batch, const uint8_t *W,
    const uint8_t *scale_code, int groups, int in_per_group,
    int out_per_group)
{
    init_decode_tables();
    const int blocks_per_group = (out_per_group + 7) / 8;
    const int tasks = groups * blocks_per_group;
    const int cblocks =
        (in_per_group + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int task = 0; task < tasks; task++) {
        const int group = task / blocks_per_group;
        const int local_r = (task % blocks_per_group) * 8;
        int lanes = out_per_group - local_r;
        if (lanes > 8) lanes = 8;
        const int global_r = group * out_per_group + local_r;
#if defined(__AVX2__) && defined(__FMA__)
        if (!getenv("DSV4_NO_SIMD") && lanes == 8) {
            __m256 acc[DSV4_MAX_TOPK + 1];
            for (int t = 0; t < batch; t++) acc[t] = _mm256_setzero_ps();
            for (int c = 0; c < in_per_group; c++) {
                float weight[8];
                for (int lane = 0; lane < 8; lane++) {
                    const int row = global_r + lane;
                    const uint8_t code =
                        W[(size_t)row * in_per_group + c];
                    const uint8_t scale = scale_code[
                        (size_t)(row / DSV4_FP8_BLOCK) * cblocks +
                        c / DSV4_FP8_BLOCK];
                    weight[lane] = wo_a_bf16_decode_table[
                        ((unsigned)scale << 8) | code];
                }
                __m256 wv = _mm256_loadu_ps(weight);
                for (int t = 0; t < batch; t++) {
                    const float *xg = x[t] + (size_t)group * in_per_group;
                    acc[t] = _mm256_fmadd_ps(
                        wv, _mm256_set1_ps(xg[c]), acc[t]);
                }
            }
            for (int t = 0; t < batch; t++) {
                float tmp[8];
                float *yg = y[t] + (size_t)group * out_per_group + local_r;
                _mm256_storeu_ps(tmp, acc[t]);
                for (int lane = 0; lane < 8; lane++)
                    yg[lane] = f32bf(tmp[lane]);
            }
            continue;
        }
#endif
        for (int lane = 0; lane < lanes; lane++) {
            const int row_index = global_r + lane;
            const uint8_t *row = W + (size_t)row_index * in_per_group;
            float acc[DSV4_MAX_TOPK + 1] = {0};
            for (int c = 0; c < in_per_group; c++) {
                const uint8_t scale = scale_code[
                    (size_t)(row_index / DSV4_FP8_BLOCK) * cblocks +
                    c / DSV4_FP8_BLOCK];
                float weight = wo_a_bf16_decode_table[
                    ((unsigned)scale << 8) | row[c]];
                for (int t = 0; t < batch; t++) {
                    const float *xg = x[t] + (size_t)group * in_per_group;
                    acc[t] = fmaf(weight, xg[c], acc[t]);
                }
            }
            for (int t = 0; t < batch; t++)
                y[t][(size_t)group * out_per_group + local_r + lane] =
                    f32bf(acc[t]);
        }
    }
}

/* FP8 GEMV. qx is the E4M3-rounded activation value; qscale the per-128-group
 * activation scale. Weight scale codes are per [128][128] block. The scale
 * product is applied AFTER each 128-column partial sum, matching the released
 * kernel (C_accum += C_local * scale_a * scale_b). */
void dsv4_gemv_fp8_q(float *y, const float *qx, const float *qscale,
                     const uint8_t *W, const uint8_t *wscale_code,
                     int in, int out, int round_out)
{
    init_decode_tables();
    const int cblocks = (in + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
#if defined(__AVX2__) && defined(__FMA__)
    const int use_gather = getenv("DSV4_FP8_GATHER") != NULL;
    const int use_direct = getenv("DSV4_FP8_DIRECT") != NULL;
    const int use_transpose = getenv("DSV4_FP8_TRANSPOSE") != NULL;
    const int transpose_gather =
        getenv("DSV4_FP8_TRANSPOSE_GATHER") != NULL;
    const int simd_rows = getenv("DSV4_NO_SIMD") ? 0 : out & ~7;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r0 = 0; r0 < simd_rows; r0 += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int cb = 0; cb < cblocks; cb++) {
            int c0 = cb * DSV4_FP8_BLOCK, c1 = c0 + DSV4_FP8_BLOCK;
            if (c1 > in) c1 = in;
            __m256 partial = _mm256_setzero_ps();
            int c = c0;
            if (use_transpose) {
                for (; c + 15 < c1; c += 16) {
                    __m128i columns[8];
                    transpose_u8_8x16(
                        W + (size_t)r0 * in + c, (size_t)in, columns);
                    for (int j = 0; j < 8; j++) {
                        __m256 wv = transpose_gather
                            ? fp8_lookup_8rows(columns[j])
                            : fp8_decode_8rows(columns[j]);
                        partial = _mm256_fmadd_ps(
                            wv, _mm256_set1_ps(qx[c + 2 * j]), partial);
                        __m128i next = _mm_srli_si128(columns[j], 8);
                        wv = transpose_gather
                            ? fp8_lookup_8rows(next)
                            : fp8_decode_8rows(next);
                        partial = _mm256_fmadd_ps(
                            wv, _mm256_set1_ps(qx[c + 2 * j + 1]), partial);
                    }
                }
            }
            for (; c < c1; c++) {
                __m256 wv;
                if (use_direct) {
                    wv = fp8_decode_8rows(load_u8_8rows(
                        W + (size_t)r0 * in + c, (size_t)in));
                } else if (use_gather) {
                    __m128i packed = load_u8_8rows(
                        W + (size_t)r0 * in + c, (size_t)in);
                    __m256i index = _mm256_cvtepu8_epi32(packed);
                    wv = _mm256_i32gather_ps(fp8_decode_table, index, 4);
                } else {
                    wv = _mm256_set_ps(
                        fp8_decode_table[W[(size_t)(r0 + 7) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 6) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 5) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 4) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 3) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 2) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 1) * in + c]],
                        fp8_decode_table[W[(size_t)r0 * in + c]]);
                }
                partial = _mm256_fmadd_ps(wv, _mm256_set1_ps(qx[c]), partial);
            }
            /* r0 advances by eight and FP8 row blocks are 128 rows, so all
             * lanes share one weight scale. Broadcast it instead of issuing
             * eight table loads and assembling an identical vector. */
            float ws = e8m0_decode_table[
                wscale_code[(size_t)(r0 / DSV4_FP8_BLOCK) * cblocks + cb]];
            if (!isnan(ws)) {
                __m256 scale = _mm256_set1_ps(qscale[cb] * ws);
                acc = _mm256_fmadd_ps(partial, scale, acc);
            }
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int lane = 0; lane < 8; lane++)
            y[r0 + lane] = round_out ? f32bf(tmp[lane]) : tmp[lane];
    }
    if (simd_rows == out) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = simd_rows; r < out; r++) {
#else
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = 0; r < out; r++) {
#endif
        const uint8_t *wrow = W + (size_t)r * in;
        const int rb = r / DSV4_FP8_BLOCK;
        float acc = 0.0f;
        for (int cb = 0; cb < cblocks; cb++) {
            int c0 = cb * DSV4_FP8_BLOCK, c1 = c0 + DSV4_FP8_BLOCK;
            if (c1 > in) c1 = in;
            float partial = 0.0f;
            for (int c = c0; c < c1; c++)
                partial = fmaf(fp8_decode_table[wrow[c]], qx[c], partial);
            float ws = e8m0_decode_table[wscale_code[(size_t)rb * cblocks + cb]];
            if (!isnan(ws)) {
                float scale = qscale[cb] * ws;
                acc = fmaf(partial, scale, acc);
            }
        }
        y[r] = round_out ? f32bf(acc) : acc;
    }
}

/* FP8 counterpart of dsv4_gemv_bf16_batch(). The per-token partial and final
 * accumulators preserve the released 128-column scaling order exactly. */
void dsv4_gemv_fp8_batch_q(float *const *y, const float *const *qx,
                           const float *const *qscale, int batch,
                           const uint8_t *W, const uint8_t *wscale_code,
                           int in, int out, int round_out)
{
    if (batch <= 0) return;
    if (batch == 1) {
        dsv4_gemv_fp8_q(y[0], qx[0], qscale[0], W, wscale_code,
                        in, out, round_out);
        return;
    }
    if (batch > DSV4_MAX_TOPK + 1) {
        fprintf(stderr, "dsv4: FP8 batch %d exceeds %d\n", batch,
                DSV4_MAX_TOPK + 1);
        exit(1);
    }
    init_decode_tables();
    const int cblocks = (in + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
#if defined(__AVX2__) && defined(__FMA__)
    const int use_gather = getenv("DSV4_FP8_GATHER") != NULL;
    const int use_direct = getenv("DSV4_FP8_DIRECT") != NULL;
    const int use_transpose = getenv("DSV4_FP8_BATCH_TRANSPOSE") != NULL;
    const int transpose_gather =
        getenv("DSV4_FP8_TRANSPOSE_GATHER") != NULL;
    const int simd_rows = getenv("DSV4_NO_SIMD") ? 0 : out & ~7;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r0 = 0; r0 < simd_rows; r0 += 8) {
        __m256 acc[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < batch; t++) acc[t] = _mm256_setzero_ps();
        for (int cb = 0; cb < cblocks; cb++) {
            int c0 = cb * DSV4_FP8_BLOCK, c1 = c0 + DSV4_FP8_BLOCK;
            if (c1 > in) c1 = in;
            __m256 partial[DSV4_MAX_TOPK + 1];
            for (int t = 0; t < batch; t++) partial[t] = _mm256_setzero_ps();
            int c = c0;
            if (use_transpose) {
                for (; c + 15 < c1; c += 16) {
                    __m128i columns[8];
                    transpose_u8_8x16(
                        W + (size_t)r0 * in + c, (size_t)in, columns);
                    for (int j = 0; j < 8; j++) {
                        __m256 wv = transpose_gather
                            ? fp8_lookup_8rows(columns[j])
                            : fp8_decode_8rows(columns[j]);
                        for (int t = 0; t < batch; t++)
                            partial[t] = _mm256_fmadd_ps(
                                wv, _mm256_set1_ps(qx[t][c + 2 * j]),
                                partial[t]);
                        __m128i next = _mm_srli_si128(columns[j], 8);
                        wv = transpose_gather
                            ? fp8_lookup_8rows(next)
                            : fp8_decode_8rows(next);
                        for (int t = 0; t < batch; t++)
                            partial[t] = _mm256_fmadd_ps(
                                wv, _mm256_set1_ps(qx[t][c + 2 * j + 1]),
                                partial[t]);
                    }
                }
            }
            for (; c < c1; c++) {
                __m256 wv;
                if (use_direct) {
                    wv = fp8_decode_8rows(load_u8_8rows(
                        W + (size_t)r0 * in + c, (size_t)in));
                } else if (use_gather) {
                    __m128i packed = load_u8_8rows(
                        W + (size_t)r0 * in + c, (size_t)in);
                    wv = _mm256_i32gather_ps(
                        fp8_decode_table, _mm256_cvtepu8_epi32(packed), 4);
                } else {
                    wv = _mm256_set_ps(
                        fp8_decode_table[W[(size_t)(r0 + 7) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 6) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 5) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 4) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 3) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 2) * in + c]],
                        fp8_decode_table[W[(size_t)(r0 + 1) * in + c]],
                        fp8_decode_table[W[(size_t)r0 * in + c]]);
                }
                for (int t = 0; t < batch; t++)
                    partial[t] = _mm256_fmadd_ps(
                        wv, _mm256_set1_ps(qx[t][c]), partial[t]);
            }
            float ws = e8m0_decode_table[
                wscale_code[(size_t)(r0 / DSV4_FP8_BLOCK) * cblocks + cb]];
            if (!isnan(ws)) {
                for (int t = 0; t < batch; t++) {
                    __m256 scale = _mm256_set1_ps(qscale[t][cb] * ws);
                    acc[t] = _mm256_fmadd_ps(partial[t], scale, acc[t]);
                }
            }
        }
        for (int t = 0; t < batch; t++) {
            float tmp[8];
            _mm256_storeu_ps(tmp, acc[t]);
            for (int lane = 0; lane < 8; lane++)
                y[t][r0 + lane] = round_out ? f32bf(tmp[lane]) : tmp[lane];
        }
    }
    if (simd_rows == out) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = simd_rows; r < out; r++) {
#else
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = 0; r < out; r++) {
#endif
        const uint8_t *wrow = W + (size_t)r * in;
        const int rb = r / DSV4_FP8_BLOCK;
        float acc[DSV4_MAX_TOPK + 1] = {0};
        for (int cb = 0; cb < cblocks; cb++) {
            int c0 = cb * DSV4_FP8_BLOCK, c1 = c0 + DSV4_FP8_BLOCK;
            if (c1 > in) c1 = in;
            float partial[DSV4_MAX_TOPK + 1] = {0};
            for (int c = c0; c < c1; c++) {
                float weight = fp8_decode_table[wrow[c]];
                for (int t = 0; t < batch; t++)
                    partial[t] = fmaf(weight, qx[t][c], partial[t]);
            }
            float ws = e8m0_decode_table[
                wscale_code[(size_t)rb * cblocks + cb]];
            if (!isnan(ws)) {
                for (int t = 0; t < batch; t++)
                    acc[t] = fmaf(partial[t], qscale[t][cb] * ws, acc[t]);
            }
        }
        for (int t = 0; t < batch; t++)
            y[t][r] = round_out ? f32bf(acc[t]) : acc[t];
    }
}

void dsv4_gemv_fp8_pair_q(float *y0, float *y1, const float *qx,
                          const float *qscale, const uint8_t *W0,
                          const uint8_t *S0, const uint8_t *W1,
                          const uint8_t *S1, int in, int out)
{
    if (getenv("DSV4_FP8_PAIR_SEPARATE")) {
        dsv4_gemv_fp8_q(y0, qx, qscale, W0, S0, in, out, 1);
        dsv4_gemv_fp8_q(y1, qx, qscale, W1, S1, in, out, 1);
        return;
    }
    init_decode_tables();
    const int cblocks = (in + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
    int scalar_start = 0;
#if defined(__AVX2__) && defined(__FMA__)
    const int use_gather = getenv("DSV4_FP8_GATHER") != NULL;
    const int use_direct = getenv("DSV4_FP8_DIRECT") != NULL;
    const int use_transpose =
        getenv("DSV4_FP8_NO_PAIR_TRANSPOSE") == NULL;
    const int transpose_gather =
        getenv("DSV4_FP8_TRANSPOSE_GATHER") != NULL;
    if (!getenv("DSV4_NO_SIMD")) {
        const int simd_rows = out & ~7;
        #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
        for (int r0 = 0; r0 < simd_rows; r0 += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (int cb = 0; cb < cblocks; cb++) {
                int c0 = cb * DSV4_FP8_BLOCK, c1 = c0 + DSV4_FP8_BLOCK;
                if (c1 > in) c1 = in;
                __m256 partial0 = _mm256_setzero_ps();
                __m256 partial1 = _mm256_setzero_ps();
                int c = c0;
                if (use_transpose) {
                    for (; c + 15 < c1; c += 16) {
                        __m128i columns0[8], columns1[8];
                        transpose_u8_8x16(
                            W0 + (size_t)r0 * in + c,
                            (size_t)in, columns0);
                        transpose_u8_8x16(
                            W1 + (size_t)r0 * in + c,
                            (size_t)in, columns1);
                        for (int j = 0; j < 8; j++) {
                            __m256 wv0 = transpose_gather
                                ? fp8_lookup_8rows(columns0[j])
                                : fp8_decode_8rows(columns0[j]);
                            __m256 wv1 = transpose_gather
                                ? fp8_lookup_8rows(columns1[j])
                                : fp8_decode_8rows(columns1[j]);
                            __m256 xv = _mm256_set1_ps(qx[c + 2 * j]);
                            partial0 = _mm256_fmadd_ps(wv0, xv, partial0);
                            partial1 = _mm256_fmadd_ps(wv1, xv, partial1);
                            __m128i next0 = _mm_srli_si128(columns0[j], 8);
                            __m128i next1 = _mm_srli_si128(columns1[j], 8);
                            wv0 = transpose_gather
                                ? fp8_lookup_8rows(next0)
                                : fp8_decode_8rows(next0);
                            wv1 = transpose_gather
                                ? fp8_lookup_8rows(next1)
                                : fp8_decode_8rows(next1);
                            xv = _mm256_set1_ps(qx[c + 2 * j + 1]);
                            partial0 = _mm256_fmadd_ps(wv0, xv, partial0);
                            partial1 = _mm256_fmadd_ps(wv1, xv, partial1);
                        }
                    }
                }
                for (; c < c1; c++) {
                    __m256 wv0, wv1;
                    if (use_direct) {
                        wv0 = fp8_decode_8rows(load_u8_8rows(
                            W0 + (size_t)r0 * in + c, (size_t)in));
                        wv1 = fp8_decode_8rows(load_u8_8rows(
                            W1 + (size_t)r0 * in + c, (size_t)in));
                    } else if (use_gather) {
                        __m128i packed0 = load_u8_8rows(
                            W0 + (size_t)r0 * in + c, (size_t)in);
                        __m128i packed1 = load_u8_8rows(
                            W1 + (size_t)r0 * in + c, (size_t)in);
                        wv0 = _mm256_i32gather_ps(
                            fp8_decode_table, _mm256_cvtepu8_epi32(packed0), 4);
                        wv1 = _mm256_i32gather_ps(
                            fp8_decode_table, _mm256_cvtepu8_epi32(packed1), 4);
                    } else {
                        wv0 = _mm256_set_ps(
                            fp8_decode_table[W0[(size_t)(r0 + 7) * in + c]],
                            fp8_decode_table[W0[(size_t)(r0 + 6) * in + c]],
                            fp8_decode_table[W0[(size_t)(r0 + 5) * in + c]],
                            fp8_decode_table[W0[(size_t)(r0 + 4) * in + c]],
                            fp8_decode_table[W0[(size_t)(r0 + 3) * in + c]],
                            fp8_decode_table[W0[(size_t)(r0 + 2) * in + c]],
                            fp8_decode_table[W0[(size_t)(r0 + 1) * in + c]],
                            fp8_decode_table[W0[(size_t)r0 * in + c]]);
                        wv1 = _mm256_set_ps(
                            fp8_decode_table[W1[(size_t)(r0 + 7) * in + c]],
                            fp8_decode_table[W1[(size_t)(r0 + 6) * in + c]],
                            fp8_decode_table[W1[(size_t)(r0 + 5) * in + c]],
                            fp8_decode_table[W1[(size_t)(r0 + 4) * in + c]],
                            fp8_decode_table[W1[(size_t)(r0 + 3) * in + c]],
                            fp8_decode_table[W1[(size_t)(r0 + 2) * in + c]],
                            fp8_decode_table[W1[(size_t)(r0 + 1) * in + c]],
                            fp8_decode_table[W1[(size_t)r0 * in + c]]);
                    }
                    __m256 xv = _mm256_set1_ps(qx[c]);
                    partial0 = _mm256_fmadd_ps(wv0, xv, partial0);
                    partial1 = _mm256_fmadd_ps(wv1, xv, partial1);
                }
                float ws0 = e8m0_decode_table[
                    S0[(size_t)(r0 / DSV4_FP8_BLOCK) * cblocks + cb]];
                float ws1 = e8m0_decode_table[
                    S1[(size_t)(r0 / DSV4_FP8_BLOCK) * cblocks + cb]];
                if (!isnan(ws0)) {
                    __m256 scale0 = _mm256_set1_ps(qscale[cb] * ws0);
                    acc0 = _mm256_fmadd_ps(partial0, scale0, acc0);
                }
                if (!isnan(ws1)) {
                    __m256 scale1 = _mm256_set1_ps(qscale[cb] * ws1);
                    acc1 = _mm256_fmadd_ps(partial1, scale1, acc1);
                }
            }
            float tmp0[8], tmp1[8];
            _mm256_storeu_ps(tmp0, acc0);
            _mm256_storeu_ps(tmp1, acc1);
            for (int lane = 0; lane < 8; lane++) {
                y0[r0 + lane] = f32bf(tmp0[lane]);
                y1[r0 + lane] = f32bf(tmp1[lane]);
            }
        }
        scalar_start = simd_rows;
    }
#endif
    if (scalar_start == out) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = scalar_start; r < out; r++) {
        const uint8_t *wrow0 = W0 + (size_t)r * in;
        const uint8_t *wrow1 = W1 + (size_t)r * in;
        const int rb = r / DSV4_FP8_BLOCK;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int cb = 0; cb < cblocks; cb++) {
            int c0 = cb * DSV4_FP8_BLOCK, c1 = c0 + DSV4_FP8_BLOCK;
            if (c1 > in) c1 = in;
            float partial0 = 0.0f, partial1 = 0.0f;
            for (int c = c0; c < c1; c++) {
                partial0 = fmaf(fp8_decode_table[wrow0[c]], qx[c], partial0);
                partial1 = fmaf(fp8_decode_table[wrow1[c]], qx[c], partial1);
            }
            float ws0 = e8m0_decode_table[S0[(size_t)rb * cblocks + cb]];
            float ws1 = e8m0_decode_table[S1[(size_t)rb * cblocks + cb]];
            if (!isnan(ws0)) acc0 = fmaf(partial0, qscale[cb] * ws0, acc0);
            if (!isnan(ws1)) acc1 = fmaf(partial1, qscale[cb] * ws1, acc1);
        }
        y0[r] = f32bf(acc0);
        y1[r] = f32bf(acc1);
    }
}

/* FP4 GEMV. qx is the E4M3-rounded activation value; qscale per-128 activation
 * scale. Weight scale codes are per row, per-32 columns. The scale product is
 * applied after each 32-column partial sum (the released kernel's
 * C_accum += C_local * scale_a * scale_b). Output is BF16. */
void dsv4_gemv_fp4_q(float *y, const float *qx, const float *qscale,
                     const uint8_t *W, const uint8_t *wscale_code,
                     int in, int rows)
{
    static const float fp4_decode[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    init_decode_tables();
    const int pcols = (in + 1) / 2;
    const int gcols = (in + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP;

#if defined(__AVX2__) && defined(__FMA__)
    const int use_transpose = getenv("DSV4_FP4_NO_TRANSPOSE") == NULL;
    const int simd_rows = getenv("DSV4_NO_SIMD") ? 0 : rows & ~7;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r0 = 0; r0 < simd_rows; r0 += 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int c0 = 0; c0 < in; c0 += DSV4_FP4_GROUP) {
            int c1 = c0 + DSV4_FP4_GROUP;
            if (c1 > in) c1 = in;
            __m256 partial = _mm256_setzero_ps();
            int c = c0;
            if (use_transpose && c1 - c0 == DSV4_FP4_GROUP) {
                __m128i columns[8];
                transpose_u8_8x16(
                    W + (size_t)r0 * pcols + (c0 >> 1),
                    (size_t)pcols, columns);
                for (int j = 0; j < 8; j++) {
                    __m256 lo, hi;
                    fp4_decode_byte_8rows(columns[j], &lo, &hi);
                    partial = _mm256_fmadd_ps(
                        lo, _mm256_set1_ps(qx[c0 + 4 * j]), partial);
                    partial = _mm256_fmadd_ps(
                        hi, _mm256_set1_ps(qx[c0 + 4 * j + 1]), partial);
                    fp4_decode_byte_8rows(_mm_srli_si128(columns[j], 8),
                                          &lo, &hi);
                    partial = _mm256_fmadd_ps(
                        lo, _mm256_set1_ps(qx[c0 + 4 * j + 2]), partial);
                    partial = _mm256_fmadd_ps(
                        hi, _mm256_set1_ps(qx[c0 + 4 * j + 3]), partial);
                }
                c = c1;
            } else {
                for (; c + 1 < c1; c += 2) {
                    __m128i packed = load_u8_8rows(
                        W + (size_t)r0 * pcols + (c >> 1), (size_t)pcols);
                    __m256 lo, hi;
                    fp4_decode_byte_8rows(packed, &lo, &hi);
                    partial = _mm256_fmadd_ps(
                        lo, _mm256_set1_ps(qx[c]), partial);
                    partial = _mm256_fmadd_ps(
                        hi, _mm256_set1_ps(qx[c + 1]), partial);
                }
                if (c < c1) {
                    __m128i packed = load_u8_8rows(
                        W + (size_t)r0 * pcols + (c >> 1), (size_t)pcols);
                    __m256 lo = fp4_decode_8rows(
                        _mm_and_si128(packed, _mm_set1_epi8(0x0f)));
                    partial = _mm256_fmadd_ps(
                        lo, _mm256_set1_ps(qx[c]), partial);
                }
            }
            int g = c0 / DSV4_FP4_GROUP;
            __m256 ws = _mm256_set_ps(
                e8m0_decode_table[wscale_code[(size_t)(r0 + 7) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 6) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 5) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 4) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 3) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 2) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 1) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)r0 * gcols + g]]);
            __m256 scale = _mm256_mul_ps(ws, _mm256_set1_ps(qscale[c0 / 128]));
            __m256 next = _mm256_fmadd_ps(partial, scale, acc);
            acc = _mm256_blendv_ps(acc, next, _mm256_cmp_ps(ws, ws, _CMP_ORD_Q));
        }
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        for (int lane = 0; lane < 8; lane++) y[r0 + lane] = f32bf(tmp[lane]);
    }
    if (simd_rows == rows) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = simd_rows; r < rows; r++) {
#else
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = 0; r < rows; r++) {
#endif
        const uint8_t *wrow = W + (size_t)r * pcols;
        float acc = 0.0f;
        for (int c0 = 0; c0 < in; c0 += DSV4_FP4_GROUP) {
            int c1 = c0 + DSV4_FP4_GROUP;
            if (c1 > in) c1 = in;
            float partial = 0.0f;
            for (int c = c0; c < c1; c++) {
                uint8_t b = wrow[c >> 1];
                uint8_t nib = (c & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0xF);
                partial = fmaf(fp4_decode[nib], qx[c], partial);
            }
            float ws = e8m0_decode_table[
                wscale_code[(size_t)r * gcols + c0 / DSV4_FP4_GROUP]];
            if (!isnan(ws)) {
                float scale = qscale[c0 / 128] * ws;
                acc = fmaf(partial, scale, acc);
            }
        }
        y[r] = f32bf(acc);
    }
}

/* Apply one packed-FP4 matrix to several independent activations. The weight
 * decode is shared, while every activation keeps the exact accumulator and
 * column order used by dsv4_gemv_fp4_q(). */
void dsv4_gemv_fp4_batch_q(float *const *y, const float *const *qx,
                           const float *const *qscale, int batch,
                           const uint8_t *W, const uint8_t *wscale_code,
                           int in, int rows)
{
    static const float fp4_decode[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    if (batch <= 0) return;
    if (batch == 1) {
        dsv4_gemv_fp4_q(y[0], qx[0], qscale[0], W, wscale_code, in, rows);
        return;
    }
    if (batch > DSV4_MAX_TOPK + 1) {
        fprintf(stderr, "dsv4: FP4 batch %d exceeds %d\n", batch,
                DSV4_MAX_TOPK + 1);
        exit(1);
    }
    init_decode_tables();
    const int pcols = (in + 1) / 2;
    const int gcols = (in + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP;

#if defined(__AVX2__) && defined(__FMA__)
    const int use_transpose = getenv("DSV4_FP4_NO_TRANSPOSE") == NULL;
    const int simd_rows = getenv("DSV4_NO_SIMD") ? 0 : rows & ~7;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r0 = 0; r0 < simd_rows; r0 += 8) {
        __m256 acc[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < batch; t++) acc[t] = _mm256_setzero_ps();
        for (int c0 = 0; c0 < in; c0 += DSV4_FP4_GROUP) {
            int c1 = c0 + DSV4_FP4_GROUP;
            if (c1 > in) c1 = in;
            __m256 partial[DSV4_MAX_TOPK + 1];
            for (int t = 0; t < batch; t++) partial[t] = _mm256_setzero_ps();
            int c = c0;
            if (use_transpose && c1 - c0 == DSV4_FP4_GROUP) {
                __m128i columns[8];
                transpose_u8_8x16(W + (size_t)r0 * pcols + (c0 >> 1),
                                  (size_t)pcols, columns);
                for (int j = 0; j < 8; j++) {
                    __m256 lo, hi;
                    fp4_decode_byte_8rows(columns[j], &lo, &hi);
                    for (int t = 0; t < batch; t++) {
                        partial[t] = _mm256_fmadd_ps(
                            lo, _mm256_set1_ps(qx[t][c0 + 4 * j]), partial[t]);
                        partial[t] = _mm256_fmadd_ps(
                            hi, _mm256_set1_ps(qx[t][c0 + 4 * j + 1]), partial[t]);
                    }
                    fp4_decode_byte_8rows(_mm_srli_si128(columns[j], 8),
                                          &lo, &hi);
                    for (int t = 0; t < batch; t++) {
                        partial[t] = _mm256_fmadd_ps(
                            lo, _mm256_set1_ps(qx[t][c0 + 4 * j + 2]), partial[t]);
                        partial[t] = _mm256_fmadd_ps(
                            hi, _mm256_set1_ps(qx[t][c0 + 4 * j + 3]), partial[t]);
                    }
                }
                c = c1;
            } else {
                for (; c + 1 < c1; c += 2) {
                    __m128i packed = load_u8_8rows(
                        W + (size_t)r0 * pcols + (c >> 1), (size_t)pcols);
                    __m256 lo, hi;
                    fp4_decode_byte_8rows(packed, &lo, &hi);
                    for (int t = 0; t < batch; t++) {
                        partial[t] = _mm256_fmadd_ps(
                            lo, _mm256_set1_ps(qx[t][c]), partial[t]);
                        partial[t] = _mm256_fmadd_ps(
                            hi, _mm256_set1_ps(qx[t][c + 1]), partial[t]);
                    }
                }
                if (c < c1) {
                    __m128i packed = load_u8_8rows(
                        W + (size_t)r0 * pcols + (c >> 1), (size_t)pcols);
                    __m256 lo = fp4_decode_8rows(
                        _mm_and_si128(packed, _mm_set1_epi8(0x0f)));
                    for (int t = 0; t < batch; t++)
                        partial[t] = _mm256_fmadd_ps(
                            lo, _mm256_set1_ps(qx[t][c]), partial[t]);
                }
            }
            int g = c0 / DSV4_FP4_GROUP;
            __m256 ws = _mm256_set_ps(
                e8m0_decode_table[wscale_code[(size_t)(r0 + 7) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 6) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 5) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 4) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 3) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 2) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)(r0 + 1) * gcols + g]],
                e8m0_decode_table[wscale_code[(size_t)r0 * gcols + g]]);
            __m256 valid = _mm256_cmp_ps(ws, ws, _CMP_ORD_Q);
            for (int t = 0; t < batch; t++) {
                __m256 scale = _mm256_mul_ps(
                    ws, _mm256_set1_ps(qscale[t][c0 / 128]));
                __m256 next = _mm256_fmadd_ps(partial[t], scale, acc[t]);
                acc[t] = _mm256_blendv_ps(acc[t], next, valid);
            }
        }
        for (int t = 0; t < batch; t++) {
            float tmp[8];
            _mm256_storeu_ps(tmp, acc[t]);
            for (int lane = 0; lane < 8; lane++)
                y[t][r0 + lane] = f32bf(tmp[lane]);
        }
    }
    if (simd_rows == rows) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = simd_rows; r < rows; r++) {
#else
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = 0; r < rows; r++) {
#endif
        const uint8_t *wrow = W + (size_t)r * pcols;
        float acc[DSV4_MAX_TOPK + 1] = {0};
        for (int c0 = 0; c0 < in; c0 += DSV4_FP4_GROUP) {
            int c1 = c0 + DSV4_FP4_GROUP;
            if (c1 > in) c1 = in;
            float partial[DSV4_MAX_TOPK + 1] = {0};
            for (int c = c0; c < c1; c++) {
                uint8_t packed = wrow[c >> 1];
                uint8_t nibble = (c & 1) ? (uint8_t)(packed >> 4)
                                           : (uint8_t)(packed & 0x0f);
                float weight = fp4_decode[nibble];
                for (int t = 0; t < batch; t++)
                    partial[t] = fmaf(weight, qx[t][c], partial[t]);
            }
            float ws = e8m0_decode_table[
                wscale_code[(size_t)r * gcols + c0 / DSV4_FP4_GROUP]];
            if (!isnan(ws)) {
                for (int t = 0; t < batch; t++)
                    acc[t] = fmaf(partial[t], qscale[t][c0 / 128] * ws,
                                  acc[t]);
            }
        }
        for (int t = 0; t < batch; t++) y[t][r] = f32bf(acc[t]);
    }
}

void dsv4_gemv_fp4_pair_q(float *y0, float *y1, const float *qx,
                          const float *qscale, const uint8_t *W0,
                          const uint8_t *S0, const uint8_t *W1,
                          const uint8_t *S1, int in, int rows)
{
    static const float fp4_decode[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    init_decode_tables();
    const int pcols = (in + 1) / 2;
    const int gcols = (in + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP;
    int scalar_start = 0;

#if defined(__AVX2__) && defined(__FMA__)
    const int use_transpose = getenv("DSV4_FP4_NO_TRANSPOSE") == NULL;
    if (!getenv("DSV4_NO_SIMD")) {
        const int simd_rows = rows & ~7;
        #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
        for (int r0 = 0; r0 < simd_rows; r0 += 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (int c0 = 0; c0 < in; c0 += DSV4_FP4_GROUP) {
                int c1 = c0 + DSV4_FP4_GROUP;
                if (c1 > in) c1 = in;
                __m256 partial0 = _mm256_setzero_ps();
                __m256 partial1 = _mm256_setzero_ps();
                int c = c0;
                if (use_transpose && c1 - c0 == DSV4_FP4_GROUP) {
                    __m128i columns0[8], columns1[8];
                    transpose_u8_8x16(
                        W0 + (size_t)r0 * pcols + (c0 >> 1),
                        (size_t)pcols, columns0);
                    transpose_u8_8x16(
                        W1 + (size_t)r0 * pcols + (c0 >> 1),
                        (size_t)pcols, columns1);
                    for (int j = 0; j < 8; j++) {
                        __m256 alo, ahi, blo, bhi;
                        fp4_decode_byte_8rows(columns0[j], &alo, &ahi);
                        fp4_decode_byte_8rows(columns1[j], &blo, &bhi);
                        __m256 xv = _mm256_set1_ps(qx[c0 + 4 * j]);
                        partial0 = _mm256_fmadd_ps(alo, xv, partial0);
                        partial1 = _mm256_fmadd_ps(blo, xv, partial1);
                        xv = _mm256_set1_ps(qx[c0 + 4 * j + 1]);
                        partial0 = _mm256_fmadd_ps(ahi, xv, partial0);
                        partial1 = _mm256_fmadd_ps(bhi, xv, partial1);
                        fp4_decode_byte_8rows(
                            _mm_srli_si128(columns0[j], 8), &alo, &ahi);
                        fp4_decode_byte_8rows(
                            _mm_srli_si128(columns1[j], 8), &blo, &bhi);
                        xv = _mm256_set1_ps(qx[c0 + 4 * j + 2]);
                        partial0 = _mm256_fmadd_ps(alo, xv, partial0);
                        partial1 = _mm256_fmadd_ps(blo, xv, partial1);
                        xv = _mm256_set1_ps(qx[c0 + 4 * j + 3]);
                        partial0 = _mm256_fmadd_ps(ahi, xv, partial0);
                        partial1 = _mm256_fmadd_ps(bhi, xv, partial1);
                    }
                    c = c1;
                } else {
                    for (; c + 1 < c1; c += 2) {
                        __m128i packed0 = load_u8_8rows(
                            W0 + (size_t)r0 * pcols + (c >> 1),
                            (size_t)pcols);
                        __m128i packed1 = load_u8_8rows(
                            W1 + (size_t)r0 * pcols + (c >> 1),
                            (size_t)pcols);
                        __m256 alo, ahi, blo, bhi;
                        fp4_decode_byte_8rows(packed0, &alo, &ahi);
                        fp4_decode_byte_8rows(packed1, &blo, &bhi);
                        __m256 xlo = _mm256_set1_ps(qx[c]);
                        partial0 = _mm256_fmadd_ps(alo, xlo, partial0);
                        partial1 = _mm256_fmadd_ps(blo, xlo, partial1);
                        __m256 xhi = _mm256_set1_ps(qx[c + 1]);
                        partial0 = _mm256_fmadd_ps(ahi, xhi, partial0);
                        partial1 = _mm256_fmadd_ps(bhi, xhi, partial1);
                    }
                    if (c < c1) {
                        __m128i packed0 = load_u8_8rows(
                            W0 + (size_t)r0 * pcols + (c >> 1),
                            (size_t)pcols);
                        __m128i packed1 = load_u8_8rows(
                            W1 + (size_t)r0 * pcols + (c >> 1),
                            (size_t)pcols);
                        __m256 av = fp4_decode_8rows(
                            _mm_and_si128(packed0, _mm_set1_epi8(0x0f)));
                        __m256 bv = fp4_decode_8rows(
                            _mm_and_si128(packed1, _mm_set1_epi8(0x0f)));
                        __m256 xv = _mm256_set1_ps(qx[c]);
                        partial0 = _mm256_fmadd_ps(av, xv, partial0);
                        partial1 = _mm256_fmadd_ps(bv, xv, partial1);
                    }
                }
                int g = c0 / DSV4_FP4_GROUP;
                __m256 ws0 = _mm256_set_ps(
                    e8m0_decode_table[S0[(size_t)(r0 + 7) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)(r0 + 6) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)(r0 + 5) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)(r0 + 4) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)(r0 + 3) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)(r0 + 2) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)(r0 + 1) * gcols + g]],
                    e8m0_decode_table[S0[(size_t)r0 * gcols + g]]);
                __m256 ws1 = _mm256_set_ps(
                    e8m0_decode_table[S1[(size_t)(r0 + 7) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)(r0 + 6) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)(r0 + 5) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)(r0 + 4) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)(r0 + 3) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)(r0 + 2) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)(r0 + 1) * gcols + g]],
                    e8m0_decode_table[S1[(size_t)r0 * gcols + g]]);
                __m256 as = _mm256_set1_ps(qscale[c0 / DSV4_FP8_BLOCK]);
                __m256 next0 = _mm256_fmadd_ps(partial0,
                                                _mm256_mul_ps(ws0, as), acc0);
                __m256 next1 = _mm256_fmadd_ps(partial1,
                                                _mm256_mul_ps(ws1, as), acc1);
                acc0 = _mm256_blendv_ps(acc0, next0,
                                        _mm256_cmp_ps(ws0, ws0, _CMP_ORD_Q));
                acc1 = _mm256_blendv_ps(acc1, next1,
                                        _mm256_cmp_ps(ws1, ws1, _CMP_ORD_Q));
            }
            float tmp0[8], tmp1[8];
            _mm256_storeu_ps(tmp0, acc0);
            _mm256_storeu_ps(tmp1, acc1);
            for (int lane = 0; lane < 8; lane++) {
                y0[r0 + lane] = f32bf(tmp0[lane]);
                y1[r0 + lane] = f32bf(tmp1[lane]);
            }
        }
        scalar_start = simd_rows;
    }
#endif
    if (scalar_start == rows) return;
    #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
    for (int r = scalar_start; r < rows; r++) {
        const uint8_t *wrow0 = W0 + (size_t)r * pcols;
        const uint8_t *wrow1 = W1 + (size_t)r * pcols;
        float acc0 = 0.0f, acc1 = 0.0f;
        for (int c0 = 0; c0 < in; c0 += DSV4_FP4_GROUP) {
            int c1 = c0 + DSV4_FP4_GROUP;
            if (c1 > in) c1 = in;
            float partial0 = 0.0f, partial1 = 0.0f;
            for (int c = c0; c < c1; c++) {
                uint8_t b0 = wrow0[c >> 1];
                uint8_t b1 = wrow1[c >> 1];
                uint8_t n0 = (c & 1) ? (uint8_t)(b0 >> 4) : (uint8_t)(b0 & 0xF);
                uint8_t n1 = (c & 1) ? (uint8_t)(b1 >> 4) : (uint8_t)(b1 & 0xF);
                partial0 = fmaf(fp4_decode[n0], qx[c], partial0);
                partial1 = fmaf(fp4_decode[n1], qx[c], partial1);
            }
            size_t si = (size_t)r * gcols + c0 / DSV4_FP4_GROUP;
            float ws0 = e8m0_decode_table[S0[si]];
            float ws1 = e8m0_decode_table[S1[si]];
            float as = qscale[c0 / DSV4_FP8_BLOCK];
            if (!isnan(ws0)) acc0 = fmaf(partial0, as * ws0, acc0);
            if (!isnan(ws1)) acc1 = fmaf(partial1, as * ws1, acc1);
        }
        y0[r] = f32bf(acc0);
        y1[r] = f32bf(acc1);
    }
}

/* FP8 activation quant + in-place dequant (the QAT simulation). round_bf16
 * selects BF16-rounded storage; the result is written back into x. group is
 * 128 or 64. E8M0 power-of-two scales, clamp to [-448, 448], values rounded
 * to the E4M3 grid. The amax floor matches the released kernel (1e-4). */
void dsv4_act_quant_inplace(float *x, int n, int group, int mode)
{
    for (int g = 0; g * group < n; g++) {
        int base = g * group, len = n - base;
        if (len > group) len = group;
        float amax = 0.0f;
        for (int i = 0; i < len; i++) {
            float a = fabsf(x[base + i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;          /* released kernel floor */
        /* scale = 2^ceil(log2(amax/448)) via the released pow2_scale rule */
        float r = amax * (1.0f / 448.0f);
        int e;
        float m = frexpf(r, &e);  /* r = m * 2^e, m in [0.5,1) */
        if (m == 0.5f) e--;        /* ceil(log2(r)) = e - (m==0.5) */
        float scale = ldexpf(1.0f, e);
        float inv = 1.0f / scale;
        for (int i = 0; i < len; i++) {
            float q = x[base + i] * inv;
            if (q > 448.0f) q = 448.0f;
            else if (q < -448.0f) q = -448.0f;
            float dequant = dsv4_round_e4m3(q) * scale;
            if (mode == 2) x[base + i] = dequant;
            else if (mode == 1) x[base + i] = dsv4_round_e4m3(dequant);
            else x[base + i] = f32bf(dequant);
        }
    }
}

/* FP4 activation quant + in-place dequant to BF16 (indexer QAT simulation). */
void dsv4_fp4_act_quant_inplace(float *x, int n, int mode)
{
    static const float fp4_abs[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const int group = DSV4_FP4_GROUP;
    for (int g = 0; g * group < n; g++) {
        int base = g * group, len = n - base;
        if (len > group) len = group;
        float amax = 0.0f;
        for (int i = 0; i < len; i++) {
            float a = fabsf(x[base + i]);
            if (a > amax) amax = a;
        }
        if (amax < 6.0f * ldexpf(1.0f, -126)) amax = 6.0f * ldexpf(1.0f, -126);
        float r = amax * (1.0f / 6.0f);
        int e;
        float m = frexpf(r, &e);  /* r = m * 2^e, m in [0.5,1) */
        if (m == 0.5f) e--;        /* ceil(log2(r)) = e - (m==0.5) */
        float scale = ldexpf(1.0f, e);
        float inv = 1.0f / scale;
        for (int i = 0; i < len; i++) {
            float q = x[base + i] * inv;
            if (q > 6.0f) q = 6.0f;
            else if (q < -6.0f) q = -6.0f;
            /* round to nearest E2M1 value; ties choose the even code */
            int idx = 0;
            float best = fabsf(q);
            for (int k = 0; k < 8; k++) {
                float d = fabsf(fabsf(q) - fp4_abs[k]);
                if (d < best) { best = d; idx = k; }
                else if (d == best && (k & 1) == 0) { idx = k; }
            }
            float dequant = (q < 0 ? -fp4_abs[idx] : fp4_abs[idx]) * scale;
            x[base + i] = (mode == 1) ? dequant : f32bf(dequant);
        }
    }
}

void dsv4_hadamard_inplace(float *x, int n)
{
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
    for (int i = 0; i < n; i++) x[i] = f32bf(x[i] * inv);   /* round BF16 */
    free(t);
}

/* YaRN/rotary frequencies. Frequencies below the beta_fast correction dimension
 * remain unchanged; frequencies above beta_slow are divided by factor. */
static void rope_freqs_fill(DSV4RopeFreq *f, int rope_dim, float theta,
                            int yarn, float factor, float beta_fast,
                            float beta_slow, int position,
                            int original_position)
{
    for (int i = 0; i < f->n; i++) {
        float expo = (float)(2 * i) / (float)rope_dim;
        float inv = 1.0f / powf(theta, expo);
        if (yarn) {
            float low = floorf((float)rope_dim
                               * logf((float)original_position / (beta_fast * 2.0f * (float)M_PI))
                               / (2.0f * logf(theta)));
            float high = ceilf((float)rope_dim
                               * logf((float)original_position / (beta_slow * 2.0f * (float)M_PI))
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

void dsv4_rope_freqs(DSV4RopeFreq *f, int rope_dim, float theta, int yarn,
                     float factor, float beta_fast, float beta_slow, int position,
                     int original_position)
{
    f->n = rope_dim / 2;
    f->cosv = (float *)malloc((size_t)f->n * sizeof(float));
    f->sinv = (float *)malloc((size_t)f->n * sizeof(float));
    if (!f->cosv || !f->sinv) {
        fprintf(stderr, "dsv4: OOM in rope_freqs\n");
        exit(1);
    }
    rope_freqs_fill(f, rope_dim, theta, yarn, factor, beta_fast, beta_slow,
                    position, original_position);
}

static const DSV4RopeFreq *model_rope_freqs(DSV4Model *m, int kind,
                                             int position,
                                             DSV4RopeFreq *owned)
{
    const int compressed = kind != 0;
    if (getenv("DSV4_NO_ROPE_CACHE")) {
        dsv4_rope_freqs(owned, m->cfg.rope_dim,
                        compressed ? m->cfg.compress_rope_theta
                                   : m->cfg.rope_theta,
                        compressed, m->cfg.rope_factor, m->cfg.beta_fast,
                        m->cfg.beta_slow, position,
                        m->cfg.original_position);
        return owned;
    }
    DSV4RopeFreq *f = &m->rope_cache[kind];
    if (m->rope_cache_position[kind] != position) {
        rope_freqs_fill(f, m->cfg.rope_dim,
                        compressed ? m->cfg.compress_rope_theta
                                   : m->cfg.rope_theta,
                        compressed, m->cfg.rope_factor, m->cfg.beta_fast,
                        m->cfg.beta_slow, position,
                        m->cfg.original_position);
        m->rope_cache_position[kind] = position;
    }
    return f;
}

static void model_rope_freqs_release(const DSV4RopeFreq *f,
                                     DSV4RopeFreq *owned)
{
    if (f != owned) return;
    free(owned->cosv);
    free(owned->sinv);
}

void dsv4_rope_apply_buf(float *x, int heads, int head_dim, const DSV4RopeFreq *f)
{
    const int start = head_dim - f->n * 2;
    for (int h = 0; h < heads; h++) {
        float *p = x + (size_t)h * head_dim + start;
        for (int i = 0; i < f->n; i++) {
            float x0 = p[2 * i], x1 = p[2 * i + 1];
            float c = f->cosv[i], s = f->sinv[i];
            p[2 * i] = f32bf(x0 * c - x1 * s);
            p[2 * i + 1] = f32bf(x0 * s + x1 * c);
        }
    }
}

void dsv4_rope_apply_buf_inv(float *x, int heads, int head_dim, const DSV4RopeFreq *f)
{
    const int start = head_dim - f->n * 2;
    for (int h = 0; h < heads; h++) {
        float *p = x + (size_t)h * head_dim + start;
        for (int i = 0; i < f->n; i++) {
            float x0 = p[2 * i], x1 = p[2 * i + 1];
            float c = f->cosv[i], s = f->sinv[i];
            p[2 * i] = f32bf(x0 * c + x1 * s);
            p[2 * i + 1] = f32bf(-x0 * s + x1 * c);
        }
    }
}


/* ============================================================ binding === */

/* Look up a tensor and validate dtype/rank/shape; fatal on any mismatch. */
static const K3Tensor *bind_t(const K3St *st, const char *name, K3Dtype want,
                              int ndim, const int64_t *shape)
{
    const K3Tensor *t = k3_st_find(st, name);
    if (!t) {
        fprintf(stderr, "dsv4: missing tensor %s\n", name);
        exit(1);
    }
    if (t->dtype != want) {
        fprintf(stderr, "dsv4: %s has dtype %d, want %d\n", name, t->dtype, want);
        exit(1);
    }
    if (t->ndim != ndim) {
        fprintf(stderr, "dsv4: %s has rank %d, want %d\n", name, t->ndim, ndim);
        exit(1);
    }
    for (int i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) {
            fprintf(stderr, "dsv4: %s shape[%d]=%lld, want %lld\n", name, i,
                    (long long)t->shape[i], (long long)shape[i]);
            exit(1);
        }
    }
    return t;
}

/* Read a tensor body into freshly allocated memory; fatal on failure. */
static void *load_t(const K3St *st, const K3Tensor *t)
{
    void *buf = malloc((size_t)t->nbytes);
    if (!buf) { fprintf(stderr, "dsv4: OOM loading %s\n", t->name); exit(1); }
    if (k3_st_read(st, t, buf) != t->nbytes) {
        fprintf(stderr, "dsv4: short read on %s\n", t->name);
        exit(1);
    }
    return buf;
}

static void load_f32(const K3St *st, const char *name, const float **out,
                     int64_t n)
{
    const int64_t shape[1] = { n };
    const K3Tensor *t = bind_t(st, name, K3_DT_F32, 1, shape);
    *out = (const float *)load_t(st, t);
}

static int map_layer_shards(DSV4Model *m)
{
    if (getenv("DSV4_NO_LAYER_MMAP")) return 1;
    for (int i = 0; i < m->st.nt; i++) {
        const K3Tensor *t = &m->st.t[i];
        int alignment = tensor_natural_alignment(t->dtype);
        if (t->off < 0 || t->off % alignment != 0) {
            fprintf(stderr, "dsv4: layer mmap unavailable: %s is not %d-byte "
                    "aligned; using streamed pread\n", t->name, alignment);
            return 0;
        }
    }
    int n = m->st.nshard;
    m->layer_shard_map = (void **)calloc((size_t)n, sizeof(void *));
    m->layer_shard_map_len = (size_t *)calloc((size_t)n, sizeof(size_t));
    if (!m->layer_shard_map || !m->layer_shard_map_len) goto fail;
    for (int i = 0; i < n; i++) {
        struct stat sb;
        if (fstat(m->st.fd[i], &sb) != 0 || sb.st_size <= 0 ||
            (uint64_t)sb.st_size > SIZE_MAX) goto fail;
        size_t len = (size_t)sb.st_size;
        void *base = mmap(NULL, len, PROT_READ, MAP_PRIVATE, m->st.fd[i], 0);
        if (base == MAP_FAILED) goto fail;
        m->layer_shard_map[i] = base;
        m->layer_shard_map_len[i] = len;
    }
    return 1;

fail:
    if (m->layer_shard_map) {
        for (int i = 0; i < n; i++)
            if (m->layer_shard_map[i])
                munmap(m->layer_shard_map[i], m->layer_shard_map_len[i]);
    }
    free(m->layer_shard_map);
    free(m->layer_shard_map_len);
    m->layer_shard_map = NULL;
    m->layer_shard_map_len = NULL;
    fprintf(stderr, "dsv4: layer mmap unavailable; using streamed pread\n");
    return 0;
}

static void *load_layer_t(DSV4Model *m, const K3Tensor *t)
{
    if (m->layer_shard_map && t->shard >= 0 && t->shard < m->st.nshard &&
        t->off >= 0 && (uint64_t)t->off + (uint64_t)t->nbytes <=
                       m->layer_shard_map_len[t->shard])
        return (uint8_t *)m->layer_shard_map[t->shard] + t->off;
    return load_t(&m->st, t);
}

static void discard_cached_tensor_mapping(DSV4Model *m, const K3Tensor *t)
{
    if (!m->layer_shard_map || getenv("DSV4_KEEP_WO_A_MAP") ||
        t->shard < 0 || t->shard >= m->st.nshard) return;
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || t->off < 0 || t->nbytes <= 0) return;

    int64_t start = ((t->off + page_size - 1) / page_size) * page_size;
    int64_t finish = t->off + t->nbytes;
    int64_t end = (finish / page_size) * page_size;
    if (start >= end || (uint64_t)end > m->layer_shard_map_len[t->shard])
        return;
    (void)madvise((uint8_t *)m->layer_shard_map[t->shard] + start,
                  (size_t)(end - start), MADV_DONTNEED);
}

static void load_layer_f32(DSV4Model *m, const char *name, const float **out,
                           int64_t n)
{
    const int64_t shape[1] = { n };
    const K3Tensor *t = bind_t(&m->st, name, K3_DT_F32, 1, shape);
    *out = (const float *)load_layer_t(m, t);
}

static void load_layer_bf16(DSV4Model *m, const char *name,
                            const uint16_t **out, int64_t n)
{
    const int64_t shape[1] = { n };
    const K3Tensor *t = bind_t(&m->st, name, K3_DT_BF16, 1, shape);
    *out = (const uint16_t *)load_layer_t(m, t);
}

static int weight_layer_from_tensor_name(const DSV4Model *m, const char *name)
{
    const char *number = NULL;
    int base = 0, limit = 0;
    if (strncmp(name, "layers.", 7) == 0) {
        number = name + 7;
        limit = m->cfg.n_layers;
    } else if (strncmp(name, "mtp.", 4) == 0) {
        number = name + 4;
        base = m->cfg.n_layers;
        limit = m->cfg.dspark_stages;
    } else {
        return -1;
    }
    char *end = NULL;
    long layer = strtol(number, &end, 10);
    if (!end || *end != '.' || layer < 0 || layer >= limit) return -1;
    return base + (int)layer;
}

static int init_layer_prefetch(DSV4Model *m)
{
    const int nl = m->cfg.n_layers + m->cfg.dspark_stages;
    int *counts = (int *)calloc((size_t)nl, sizeof(*counts));
    int *fill = (int *)calloc((size_t)nl, sizeof(*fill));
    m->prefetch = (DSV4LayerPrefetch *)calloc((size_t)nl, sizeof(*m->prefetch));
    if (!counts || !fill || !m->prefetch) goto fail;

    for (int i = 0; i < m->st.nt; i++) {
        const K3Tensor *t = &m->st.t[i];
        int layer = weight_layer_from_tensor_name(m, t->name);
        if (layer >= 0 && layer < nl && !strstr(t->name, ".ffn.experts."))
            counts[layer]++;
    }
    for (int layer = 0; layer < nl; layer++) {
        m->prefetch[layer].count = counts[layer];
        m->prefetch[layer].tensor = (const K3Tensor **)malloc(
            (size_t)(counts[layer] ? counts[layer] : 1) * sizeof(const K3Tensor *));
        if (!m->prefetch[layer].tensor) goto fail;
    }
    for (int i = 0; i < m->st.nt; i++) {
        const K3Tensor *t = &m->st.t[i];
        int layer = weight_layer_from_tensor_name(m, t->name);
        if (layer >= 0 && layer < nl && !strstr(t->name, ".ffn.experts."))
            m->prefetch[layer].tensor[fill[layer]++] = t;
    }
    free(counts);
    free(fill);
    return 1;

fail:
    free(counts);
    free(fill);
    return 0;
}

static void prefetch_layer_weights(DSV4Model *m, int layer)
{
    const int nl = m->cfg.n_layers + m->cfg.dspark_stages;
    if (!m->prefetch || layer < 0 || layer >= nl ||
        getenv("DSV4_NO_PREFETCH")) return;
    const DSV4LayerPrefetch *p = &m->prefetch[layer];
    for (int i = 0; i < p->count; i++) {
        const K3Tensor *t = p->tensor[i];
        if (((m->wo_a_cache && m->wo_a_cache[layer]) ||
             (m->wo_a_code_cache && m->wo_a_code_cache[layer])) &&
            strstr(t->name, ".attn.wo_a."))
            continue;
        if (posix_fadvise(m->st.fd[t->shard], (off_t)t->off, (off_t)t->nbytes,
                          POSIX_FADV_WILLNEED) == 0) {
            m->prefetch_calls++;
            m->prefetch_bytes += (uint64_t)t->nbytes;
        }
    }
}

/* Load the per-layer weight bundle into a freshly allocated DSV4LayerW. */
static DSV4LayerW *load_layer(DSV4Model *m, int layer)
{
    K3St *st = &m->st;
    const DSV4Config *cfg = &m->cfg;
    DSV4LayerW *w = (DSV4LayerW *)calloc(1, sizeof *w);
    if (!w) { fprintf(stderr, "dsv4: OOM layer bundle\n"); exit(1); }
    const int is_draft = layer >= cfg->n_layers;
    const int stage = layer - cfg->n_layers;
    if (layer < 0 || (is_draft && stage >= cfg->dspark_stages)) {
        fprintf(stderr, "dsv4: invalid weight layer %d\n", layer);
        exit(1);
    }
    const char *scope = is_draft ? "mtp" : "layers";
    const int scope_layer = is_draft ? stage : layer;
    char name[512];
#define LAYER_NAME(suffix) \
    snprintf(name, sizeof name, "%s.%d.%s", scope, scope_layer, suffix)
    const int ratio = is_draft ? 0 : cfg->compress_ratio[layer];
    const int coff = (ratio == 4) ? 2 : 1;
    const int hid = cfg->hidden;

    w->compress_ratio = ratio;
    w->is_hash = !is_draft && layer < cfg->n_hash_layers;

    LAYER_NAME("attn_norm.weight");
    load_layer_bf16(m, name, &w->attn_norm, hid);
    LAYER_NAME("ffn_norm.weight");
    load_layer_bf16(m, name, &w->ffn_norm, hid);
    LAYER_NAME("attn.q_norm.weight");
    load_layer_bf16(m, name, &w->q_norm, cfg->q_lora);
    LAYER_NAME("attn.kv_norm.weight");
    load_layer_bf16(m, name, &w->kv_norm, cfg->head_dim);

    LAYER_NAME("hc_attn_fn");
    { int64_t sh[2] = { (2 + cfg->hc_mult) * cfg->hc_mult, cfg->hc_mult * hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_F32, 2, sh);
      w->hc_attn_fn = (const float *)load_layer_t(m, t); }
    LAYER_NAME("hc_ffn_fn");
    { int64_t sh[2] = { (2 + cfg->hc_mult) * cfg->hc_mult, cfg->hc_mult * hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_F32, 2, sh);
      w->hc_ffn_fn = (const float *)load_layer_t(m, t); }
    LAYER_NAME("hc_attn_scale");
    load_layer_f32(m, name, &w->hc_attn_scale, 3);
    LAYER_NAME("hc_ffn_scale");
    load_layer_f32(m, name, &w->hc_ffn_scale, 3);
    LAYER_NAME("hc_attn_base");
    load_layer_f32(m, name, &w->hc_attn_base, (2 + cfg->hc_mult) * cfg->hc_mult);
    LAYER_NAME("hc_ffn_base");
    load_layer_f32(m, name, &w->hc_ffn_base, (2 + cfg->hc_mult) * cfg->hc_mult);

    LAYER_NAME("attn.attn_sink");
    load_layer_f32(m, name, &w->attn_sink, cfg->n_heads);

    /* wq_a: BF16 [q_lora, hidden] */
    LAYER_NAME("attn.wq_a.weight");
    { int64_t sh[2] = { cfg->q_lora, hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->wq_a = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("attn.wq_a.scale");
    { int64_t sh[2] = { (cfg->q_lora + 127) / 128, (hid + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->wq_a_scale = (const uint8_t *)load_layer_t(m, t); }
    /* wq_b: FP8 [heads*head_dim, q_lora] */
    LAYER_NAME("attn.wq_b.weight");
    { int64_t sh[2] = { cfg->n_heads * cfg->head_dim, cfg->q_lora };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->wq_b = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("attn.wq_b.scale");
    { int64_t sh[2] = { (cfg->n_heads * cfg->head_dim + 127) / 128, (cfg->q_lora + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->wq_b_scale = (const uint8_t *)load_layer_t(m, t); }
    /* wkv: BF16 [head_dim, hidden] */
    LAYER_NAME("attn.wkv.weight");
    { int64_t sh[2] = { cfg->head_dim, hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->wkv = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("attn.wkv.scale");
    { int64_t sh[2] = { (cfg->head_dim + 127) / 128, (hid + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->wkv_scale = (const uint8_t *)load_layer_t(m, t); }
    /* wo_a: FP8 [o_lora*n_groups, heads*head_dim/groups] */
    /* wo_a: FP8 on disk, decoded per element with its scale, rounded to BF16;
     * the attention input is NOT activation-quantised for this projection */
    LAYER_NAME("attn.wo_a.weight");
    {
        int64_t sh[2] = { cfg->o_groups * cfg->o_lora, cfg->n_heads * cfg->head_dim / cfg->o_groups };
        const int cache_wo_a = (m->wo_a_cache || m->wo_a_code_cache) &&
            (is_draft || layer < m->wo_a_cache_layers);
        const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
        LAYER_NAME("attn.wo_a.scale");
        int64_t ssh[2] = { (cfg->o_groups * cfg->o_lora + 127) / 128, (cfg->n_heads * cfg->head_dim / cfg->o_groups + 127) / 128 };
        const K3Tensor *st_ = bind_t(st, name, K3_DT_F8_E8M0, 2, ssh);
        if (m->packed_wo_a && cache_wo_a && m->wo_a_code_cache[layer]) {
            w->wo_a_codes = m->wo_a_code_cache[layer];
            w->wo_a_scale = m->wo_a_scale_cache[layer];
            w->wo_a_cached = 1;
            m->wo_a_cache_hits++;
        } else if (!m->packed_wo_a && cache_wo_a && m->wo_a_cache[layer]) {
            w->wo_a = m->wo_a_cache[layer];
            w->wo_a_cached = 1;
            m->wo_a_cache_hits++;
        } else {
            const uint8_t *codes = (const uint8_t *)load_layer_t(m, t);
            const uint8_t *scales = (const uint8_t *)load_layer_t(m, st_);
            int rows = sh[0], cols = sh[1], scols = ssh[1];
            if (m->packed_wo_a) {
                if (cache_wo_a) {
                    if (m->layer_shard_map) {
                        /* The shard mappings live for the model lifetime and
                         * clean pages remain reclaimable by Linux. */
                        m->wo_a_code_cache[layer] = (uint8_t *)codes;
                        m->wo_a_scale_cache[layer] = (uint8_t *)scales;
                    } else {
                        uint8_t *owned_codes =
                            (uint8_t *)malloc((size_t)t->nbytes);
                        uint8_t *owned_scales =
                            (uint8_t *)malloc((size_t)st_->nbytes);
                        if (!owned_codes || !owned_scales) {
                            fprintf(stderr, "dsv4: OOM packed wo_a\n");
                            exit(1);
                        }
                        memcpy(owned_codes, codes, (size_t)t->nbytes);
                        memcpy(owned_scales, scales, (size_t)st_->nbytes);
                        free((void *)codes);
                        free((void *)scales);
                        m->wo_a_code_cache[layer] = owned_codes;
                        m->wo_a_scale_cache[layer] = owned_scales;
                    }
                    w->wo_a_codes = m->wo_a_code_cache[layer];
                    w->wo_a_scale = m->wo_a_scale_cache[layer];
                    w->wo_a_cached = 1;
                    m->wo_a_cache_misses++;
                } else {
                    w->wo_a_codes = codes;
                    w->wo_a_scale = scales;
                }
            } else {
                uint16_t *bf = (uint16_t *)malloc((size_t)rows * cols * 2);
                if (!bf) { fprintf(stderr, "dsv4: OOM wo_a bf16\n"); exit(1); }
                init_decode_tables();
                #pragma omp parallel for schedule(static) num_threads(m->threads)
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        float v = fp8_decode_table[codes[(size_t)i * cols + j]]
                            * e8m0_decode_table[scales[(size_t)(i / 128) * scols + j / 128]];
                        bf[(size_t)i * cols + j] = dsv4_f32_to_bf16(v);
                    }
                }
                if (cache_wo_a) {
                    discard_cached_tensor_mapping(m, t);
                    discard_cached_tensor_mapping(m, st_);
                }
                w->wo_a = bf;
                if (!m->layer_shard_map) {
                    free((void *)codes);
                    free((void *)scales);
                }
                if (cache_wo_a) {
                    m->wo_a_cache[layer] = bf;
                    w->wo_a_cached = 1;
                    m->wo_a_cache_misses++;
                }
            }
        }
    }
    /* wo_b: FP8 [hidden, o_lora*n_groups] */
    LAYER_NAME("attn.wo_b.weight");
    { int64_t sh[2] = { hid, cfg->o_groups * cfg->o_lora };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->wo_b = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("attn.wo_b.scale");
    { int64_t sh[2] = { (hid + 127) / 128, (cfg->o_groups * cfg->o_lora + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->wo_b_scale = (const uint8_t *)load_layer_t(m, t); }

    if (ratio) {
        LAYER_NAME("attn.compressor.ape");
        { int64_t sh[2] = { ratio, coff * cfg->head_dim };
          const K3Tensor *t = bind_t(st, name, K3_DT_F32, 2, sh);
          w->ape = (const float *)load_layer_t(m, t); }
        LAYER_NAME("attn.compressor.wkv.weight");
        { int64_t sh[2] = { coff * cfg->head_dim, hid };
          const K3Tensor *t = bind_t(st, name, K3_DT_BF16, 2, sh);
          w->comp_wkv = (const uint16_t *)load_layer_t(m, t); }
        LAYER_NAME("attn.compressor.wgate.weight");
        { int64_t sh[2] = { coff * cfg->head_dim, hid };
          const K3Tensor *t = bind_t(st, name, K3_DT_BF16, 2, sh);
          w->comp_wgate = (const uint16_t *)load_layer_t(m, t); }
        LAYER_NAME("attn.compressor.norm.weight");
        load_layer_bf16(m, name, &w->comp_norm, cfg->head_dim);

        if (ratio == 4) {
            LAYER_NAME("attn.indexer.wq_b.weight");
            { int64_t sh[2] = { cfg->index_heads * cfg->index_dim, cfg->q_lora };
              const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
              w->idx_wq_b = (const uint8_t *)load_layer_t(m, t); }
            LAYER_NAME("attn.indexer.wq_b.scale");
            { int64_t sh[2] = { (cfg->index_heads * cfg->index_dim + 127) / 128, (cfg->q_lora + 127) / 128 };
              const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
              w->idx_wq_b_scale = (const uint8_t *)load_layer_t(m, t); }
            LAYER_NAME("attn.indexer.weights_proj.weight");
            { int64_t sh[2] = { cfg->index_heads, hid };
              const K3Tensor *t = bind_t(st, name, K3_DT_BF16, 2, sh);
              w->idx_weights_proj = (const uint16_t *)load_layer_t(m, t); }
            LAYER_NAME("attn.indexer.compressor.ape");
            { int64_t sh[2] = { ratio, coff * cfg->index_dim };
              const K3Tensor *t = bind_t(st, name, K3_DT_F32, 2, sh);
              w->idx_ape = (const float *)load_layer_t(m, t); }
            LAYER_NAME("attn.indexer.compressor.wkv.weight");
            { int64_t sh[2] = { coff * cfg->index_dim, hid };
              const K3Tensor *t = bind_t(st, name, K3_DT_BF16, 2, sh);
              w->idx_comp_wkv = (const uint16_t *)load_layer_t(m, t); }
            LAYER_NAME("attn.indexer.compressor.wgate.weight");
            { int64_t sh[2] = { coff * cfg->index_dim, hid };
              const K3Tensor *t = bind_t(st, name, K3_DT_BF16, 2, sh);
              w->idx_comp_wgate = (const uint16_t *)load_layer_t(m, t); }
            LAYER_NAME("attn.indexer.compressor.norm.weight");
            load_layer_bf16(m, name, &w->idx_comp_norm, cfg->index_dim);
        }
    }

    LAYER_NAME("ffn.gate.weight");
    { int64_t sh[2] = { cfg->n_experts, hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_BF16, 2, sh);
      w->gate_w = (const uint16_t *)load_layer_t(m, t); }
    if (w->is_hash) {
        LAYER_NAME("ffn.gate.tid2eid");
        { int64_t sh[2] = { cfg->vocab, cfg->topk };
          const K3Tensor *t = bind_t(st, name, K3_DT_I64, 2, sh);
          w->tid2eid = (const int64_t *)load_layer_t(m, t); }
        w->bias = NULL;
    } else {
        LAYER_NAME("ffn.gate.bias");
        load_layer_f32(m, name, &w->bias, cfg->n_experts);
    }

    LAYER_NAME("ffn.shared_experts.w1.weight");
    { int64_t sh[2] = { cfg->moe_inter, hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->sh1 = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("ffn.shared_experts.w1.scale");
    { int64_t sh[2] = { (cfg->moe_inter + 127) / 128, (hid + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->sh1_s = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("ffn.shared_experts.w3.weight");
    { int64_t sh[2] = { cfg->moe_inter, hid };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->sh3 = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("ffn.shared_experts.w3.scale");
    { int64_t sh[2] = { (cfg->moe_inter + 127) / 128, (hid + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->sh3_s = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("ffn.shared_experts.w2.weight");
    { int64_t sh[2] = { hid, cfg->moe_inter };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E4M3, 2, sh);
      w->sh2 = (const uint8_t *)load_layer_t(m, t); }
    LAYER_NAME("ffn.shared_experts.w2.scale");
    { int64_t sh[2] = { (hid + 127) / 128, (cfg->moe_inter + 127) / 128 };
      const K3Tensor *t = bind_t(st, name, K3_DT_F8_E8M0, 2, sh);
      w->sh2_s = (const uint8_t *)load_layer_t(m, t); }

    #undef LAYER_NAME
    return w;
}


/* ========================================================= model open === */

static uint64_t ceil_div_u(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

/* Persistent context memory for `context` tokens, shared with the planner. */
static uint64_t ctx_for(const DSV4Config *c, int context)
{
    int n4 = 0, n128 = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->compress_ratio[i] == 4) n4++;
        else if (c->compress_ratio[i] == 128) n128++;
    }
    const uint64_t C = (uint64_t)context;
    uint64_t ctx = 0;
    ctx += (uint64_t)c->n_layers * (uint64_t)c->window * 512u * 4u;
    ctx += (uint64_t)n4 * ceil_div_u(C, 4) * 512u * 4u;
    ctx += (uint64_t)n4 * ceil_div_u(C, 4) * (uint64_t)c->index_dim * 4u;
    ctx += (uint64_t)n128 * ceil_div_u(C, 128) * 512u * 4u;
    return ctx;
}

static void free_layer(DSV4Model *m, DSV4LayerW *w)
{
    if (!w) return;
    if (m->layer_shard_map) {
        if (!w->wo_a_cached) free((void *)w->wo_a);
        free(w);
        return;
    }
    free((void *)w->attn_norm);
    free((void *)w->ffn_norm);
    free((void *)w->q_norm);
    free((void *)w->kv_norm);
    free((void *)w->attn_sink);
    free((void *)w->bias);
    free((void *)w->tid2eid);
    free((void *)w->hc_attn_fn);
    free((void *)w->hc_ffn_fn);
    free((void *)w->hc_attn_scale);
    free((void *)w->hc_ffn_scale);
    free((void *)w->hc_attn_base);
    free((void *)w->hc_ffn_base);
    free((void *)w->wq_a);
    free((void *)w->wq_a_scale);
    free((void *)w->wq_b);
    free((void *)w->wq_b_scale);
    free((void *)w->wkv);
    free((void *)w->wkv_scale);
    if (!w->wo_a_cached) free((void *)w->wo_a);
    if (!w->wo_a_cached) {
        free((void *)w->wo_a_codes);
        free((void *)w->wo_a_scale);
    }
    free((void *)w->wo_b);
    free((void *)w->wo_b_scale);
    free((void *)w->ape);
    free((void *)w->comp_wkv);
    free((void *)w->comp_wgate);
    free((void *)w->comp_norm);
    free((void *)w->idx_wq_b);
    free((void *)w->idx_wq_b_scale);
    free((void *)w->idx_weights_proj);
    free((void *)w->idx_ape);
    free((void *)w->idx_comp_wkv);
    free((void *)w->idx_comp_wgate);
    free((void *)w->idx_comp_norm);
    free((void *)w->gate_w);
    free((void *)w->sh1);
    free((void *)w->sh1_s);
    free((void *)w->sh3);
    free((void *)w->sh3_s);
    free((void *)w->sh2);
    free((void *)w->sh2_s);
    free(w);
}

struct DSV4DSpark {
    const uint8_t *main_proj, *main_proj_scale;
    const uint16_t *main_norm;
    const uint8_t *target_wkv[DSV4_MAX_DSPARK_STAGES];
    const uint8_t *target_wkv_scale[DSV4_MAX_DSPARK_STAGES];
    const uint16_t *target_kv_norm[DSV4_MAX_DSPARK_STAGES];
    const uint16_t *final_norm;
    const float *hc_head_fn, *hc_head_base, *hc_head_scale;
    const K3Tensor *markov_w1, *markov_w2, *confidence_proj;
    float *target_kv; /* [stage][window][head_dim] */
    int *target_position;
    int owns_loaded_weights;
};

static void dspark_close(DSV4Model *m)
{
    DSV4DSpark *d = m->dspark;
    if (!d) return;
    if (d->owns_loaded_weights) {
        free((void *)d->main_proj);
        free((void *)d->main_proj_scale);
        free((void *)d->main_norm);
        for (int s = 0; s < m->cfg.dspark_stages; s++) {
            free((void *)d->target_wkv[s]);
            free((void *)d->target_wkv_scale[s]);
            free((void *)d->target_kv_norm[s]);
        }
        free((void *)d->final_norm);
        free((void *)d->hc_head_fn);
        free((void *)d->hc_head_base);
        free((void *)d->hc_head_scale);
    }
    free(d->target_kv);
    free(d->target_position);
    free(d);
    m->dspark = NULL;
}

static int dspark_open(DSV4Model *m)
{
    const DSV4Config *c = &m->cfg;
    if (c->dspark_stages == 0 || m->draft_cache_slots == 0) return 1;
    DSV4DSpark *d = (DSV4DSpark *)calloc(1, sizeof(*d));
    if (!d) return 0;
    m->dspark = d;
    d->owns_loaded_weights = m->layer_shard_map == NULL;

    char name[256];
    int64_t sh_proj[2] = { c->hidden, (int64_t)c->hidden * c->dspark_stages };
    const K3Tensor *t = bind_t(&m->st, "mtp.0.main_proj.weight",
                               K3_DT_F8_E4M3, 2, sh_proj);
    d->main_proj = (const uint8_t *)load_layer_t(m, t);
    int64_t sh_proj_scale[2] = { (c->hidden + 127) / 128,
                                 ((int64_t)c->hidden * c->dspark_stages + 127) / 128 };
    t = bind_t(&m->st, "mtp.0.main_proj.scale", K3_DT_F8_E8M0, 2,
               sh_proj_scale);
    d->main_proj_scale = (const uint8_t *)load_layer_t(m, t);
    load_layer_bf16(m, "mtp.0.main_norm.weight", &d->main_norm, c->hidden);

    for (int s = 0; s < c->dspark_stages; s++) {
        snprintf(name, sizeof name, "mtp.%d.attn.wkv.weight", s);
        int64_t sh[2] = { c->head_dim, c->hidden };
        t = bind_t(&m->st, name, K3_DT_F8_E4M3, 2, sh);
        d->target_wkv[s] = (const uint8_t *)load_layer_t(m, t);
        snprintf(name, sizeof name, "mtp.%d.attn.wkv.scale", s);
        int64_t ss[2] = { (c->head_dim + 127) / 128,
                          (c->hidden + 127) / 128 };
        t = bind_t(&m->st, name, K3_DT_F8_E8M0, 2, ss);
        d->target_wkv_scale[s] = (const uint8_t *)load_layer_t(m, t);
        snprintf(name, sizeof name, "mtp.%d.attn.kv_norm.weight", s);
        load_layer_bf16(m, name, &d->target_kv_norm[s], c->head_dim);
    }

    const int last = c->dspark_stages - 1;
    snprintf(name, sizeof name, "mtp.%d.norm.weight", last);
    load_layer_bf16(m, name, &d->final_norm, c->hidden);
    snprintf(name, sizeof name, "mtp.%d.hc_head_fn", last);
    int64_t sh_hc[2] = { c->hc_mult, (int64_t)c->hc_mult * c->hidden };
    t = bind_t(&m->st, name, K3_DT_F32, 2, sh_hc);
    d->hc_head_fn = (const float *)load_layer_t(m, t);
    snprintf(name, sizeof name, "mtp.%d.hc_head_base", last);
    load_layer_f32(m, name, &d->hc_head_base, c->hc_mult);
    snprintf(name, sizeof name, "mtp.%d.hc_head_scale", last);
    load_layer_f32(m, name, &d->hc_head_scale, 1);

    snprintf(name, sizeof name, "mtp.%d.markov_head.markov_w1.weight", last);
    int64_t sh_markov[2] = { c->vocab, c->dspark_markov_rank };
    d->markov_w1 = bind_t(&m->st, name, K3_DT_BF16, 2, sh_markov);
    snprintf(name, sizeof name, "mtp.%d.markov_head.markov_w2.weight", last);
    d->markov_w2 = bind_t(&m->st, name, K3_DT_BF16, 2, sh_markov);
    snprintf(name, sizeof name, "mtp.%d.confidence_head.proj.weight", last);
    int64_t sh_conf[2] = { 1, (int64_t)c->hidden + c->dspark_markov_rank };
    d->confidence_proj = bind_t(&m->st, name, K3_DT_BF16, 2, sh_conf);

    d->target_kv = (float *)calloc((size_t)c->dspark_stages * c->window *
                                    c->head_dim, sizeof(float));
    d->target_position = (int *)malloc((size_t)c->window * sizeof(int));
    if (!d->target_kv || !d->target_position) {
        dspark_close(m);
        return 0;
    }
    for (int i = 0; i < c->window; i++) d->target_position[i] = -1;
    return 1;
}

DSV4Model *dsv4_model_open(const char *dir, const DSV4Options *opt)
{
    DSV4Model *m = (DSV4Model *)calloc(1, sizeof *m);
    if (!m) return NULL;
    m->opt = *opt;
#ifdef _OPENMP
    int auto_threads = omp_get_num_procs();
    if (auto_threads > 12) auto_threads = 12;
    if (auto_threads < 1) auto_threads = 1;
    m->threads = opt->threads > 0 ? opt->threads : auto_threads;
    omp_set_dynamic(0);
    omp_set_num_threads(m->threads);
    omp_set_max_active_levels(1);
#else
    m->threads = 1;
#endif
    m->profiling = getenv("DSV4_PROFILE") != NULL;

    char path[4096];
    snprintf(path, sizeof path, "%s/config.json", dir);
    if (!dsv4_config_load_file(&m->cfg, path)) { free(m); return NULL; }
    const int enable_dspark = getenv("DSV4_EXPERIMENTAL_DSPARK") != NULL;

    /* The CLI runs dsv4_memory_plan() itself so it can print the plan BEFORE
     * opening; it then passes the expert cache budget through opt.
     * expert_cache_bytes == 0 means "default 2 GiB memory plan". */
    int context = opt->max_context > 0 ? opt->max_context : 4096;
    uint64_t cache_bytes = opt->expert_cache_bytes;
    if (cache_bytes == 0) {
        uint64_t ctx = ctx_for(&m->cfg, context);
        uint64_t budget = (2ull << 30);
        uint64_t wo_a_values = (uint64_t)m->cfg.o_lora * m->cfg.n_heads *
                               m->cfg.head_dim;
        uint64_t wo_a_layer = wo_a_values * 2u;
        if (getenv("DSV4_PACKED_WO_A")) {
            uint64_t rows = (uint64_t)m->cfg.o_groups * m->cfg.o_lora;
            uint64_t cols = (uint64_t)m->cfg.n_heads * m->cfg.head_dim /
                            m->cfg.o_groups;
            wo_a_layer = wo_a_values + (rows + 127) / 128 *
                         ((cols + 127) / 128);
        }
        uint64_t dspark = enable_dspark
            ? (uint64_t)m->cfg.dspark_stages * wo_a_layer
              + (uint64_t)m->cfg.dspark_stages * m->cfg.window *
                m->cfg.head_dim * sizeof(float)
            : 0;
        uint64_t fixed = 1258291200ull + ctx + dspark;
        uint64_t slots = budget > fixed ? (budget - fixed) / EXPERT_SLOT : 1;
        if (slots < 1) slots = 1;
        cache_bytes = slots * EXPERT_SLOT;
    }
    m->context = context;
    m->wo_a_cache_layers = opt->wo_a_cache_layers;
    m->packed_wo_a = getenv("DSV4_PACKED_WO_A") != NULL;
    if (m->wo_a_cache_layers < 0) m->wo_a_cache_layers = 0;
    if (m->wo_a_cache_layers > m->cfg.n_layers)
        m->wo_a_cache_layers = m->cfg.n_layers;
    if (m->wo_a_cache_layers > 0 ||
        (enable_dspark && m->cfg.dspark_stages > 0)) {
        const int weight_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        if (m->packed_wo_a) {
            m->wo_a_code_cache = (uint8_t **)calloc(
                (size_t)weight_layers, sizeof(*m->wo_a_code_cache));
            m->wo_a_scale_cache = (uint8_t **)calloc(
                (size_t)weight_layers, sizeof(*m->wo_a_scale_cache));
        } else {
            m->wo_a_cache = (uint16_t **)calloc((size_t)weight_layers,
                                                sizeof(*m->wo_a_cache));
        }
        if ((!m->packed_wo_a && !m->wo_a_cache) ||
            (m->packed_wo_a && (!m->wo_a_code_cache ||
                                !m->wo_a_scale_cache))) {
            fprintf(stderr, "dsv4: OOM wo_a cache index\n");
            dsv4_model_close(m);
            return NULL;
        }
    }
    /* safetensors index */
    if (k3_st_open(&m->st, dir) != 0) { dsv4_model_close(m); return NULL; }
    (void)map_layer_shards(m);
    if (!init_layer_prefetch(m)) {
        fprintf(stderr, "dsv4: OOM layer prefetch index\n");
        dsv4_model_close(m);
        return NULL;
    }

    /* global weights */
    {
        int64_t sh1[1] = { m->cfg.hidden };
        const K3Tensor *t = bind_t(&m->st, "norm.weight", K3_DT_BF16, 1, sh1);
        m->norm = (const uint16_t *)load_t(&m->st, t);
    }
    {
        int64_t sh2[2] = { m->cfg.hc_mult, m->cfg.hc_mult * m->cfg.hidden };
        const K3Tensor *t = bind_t(&m->st, "hc_head_fn", K3_DT_F32, 2, sh2);
        m->hc_head_fn = (const float *)load_t(&m->st, t);
    }
    load_f32(&m->st, "hc_head_base", &m->hc_head_base, m->cfg.hc_mult);
    load_f32(&m->st, "hc_head_scale", &m->hc_head_scale, 1);
    {
        int64_t sh3[2] = { m->cfg.vocab, m->cfg.hidden };
        m->embed_t = bind_t(&m->st, "embed.weight", K3_DT_BF16, 2, sh3);
        m->head_t  = bind_t(&m->st, "head.weight",  K3_DT_BF16, 2, sh3);
        (void)map_vocab_head(m);
    }

    /* runtime buffers */
    const int mult = m->cfg.hc_mult;
    const int hid = m->cfg.hidden;
    const int mi = m->cfg.moe_inter;
    const int ne = m->cfg.n_experts;
    const int topk = m->cfg.topk;
    const size_t scratch_floats = (size_t)hid + (hid + 127) / 128 +
                                  (size_t)2 * ne + topk +
                                  (size_t)(topk + 2) * hid;
    const size_t expert_work_stride = (size_t)4 * mi + (mi + 127) / 128;
    const size_t expert_work_floats = (size_t)(topk + 1) * expert_work_stride;
    size_t max_compressed = 0;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        const int ratio = m->cfg.compress_ratio[L];
        size_t n = ratio == 4 ? (size_t)m->cfg.index_topk
                              : ratio > 0 ? (size_t)(context + ratio - 1) / ratio
                                          : 0;
        if (n > max_compressed) max_compressed = n;
    }
    const size_t max_attn_keys = (size_t)m->cfg.window + max_compressed;
    const size_t attention_floats =
        (size_t)2 * m->cfg.q_lora +
        (size_t)2 * m->cfg.n_heads * m->cfg.head_dim +
        (size_t)32 * m->cfg.head_dim +
        (size_t)m->cfg.index_heads * m->cfg.index_dim +
        (size_t)32 * m->cfg.index_dim +
        (size_t)(context + 3) / 4 +
        (size_t)2 * m->cfg.n_heads * max_attn_keys +
        (size_t)m->cfg.o_groups * m->cfg.o_lora + 4096;
    size_t attention_bytes = attention_floats * sizeof(float) +
        ((size_t)2 * max_attn_keys + max_compressed + 4096) * sizeof(int);
    if (attention_bytes < (1u << 20)) attention_bytes = 1u << 20;
    m->state = (float *)calloc((size_t)mult * hid, sizeof(float));
    m->mixes_buf = (float *)malloc((size_t)((2 + mult) * mult) * sizeof(float));
    m->hc_state_buf = (float *)malloc((size_t)mult * hid * sizeof(float));
    m->head_hidden = (float *)malloc((size_t)hid * sizeof(float));
    if (!m->head_map_rows)
        m->head_buf = (uint8_t *)malloc((size_t)1024 * hid * sizeof(uint16_t));
    m->scratch = (float *)malloc(scratch_floats * sizeof(float));
    m->expert_work = (float *)malloc(expert_work_floats * sizeof(float));
    m->route_scratch = (int *)malloc((size_t)(ne + topk) * sizeof(int));
    m->attention_scratch = (unsigned char *)malloc(attention_bytes);
    m->attention_scratch_cap = attention_bytes;
    for (int k = 0; k < 3; k++) {
        m->rope_cache[k].n = m->cfg.rope_dim / 2;
        m->rope_cache[k].cosv = (float *)malloc(
            (size_t)m->rope_cache[k].n * sizeof(float));
        m->rope_cache[k].sinv = (float *)malloc(
            (size_t)m->rope_cache[k].n * sizeof(float));
        m->rope_cache_position[k] = INT_MIN;
    }
    if (!m->state || !m->mixes_buf || !m->hc_state_buf || !m->head_hidden ||
        (!m->head_map_rows && !m->head_buf) || !m->scratch || !m->expert_work ||
        !m->route_scratch || !m->attention_scratch ||
        !m->rope_cache[0].cosv || !m->rope_cache[0].sinv ||
        !m->rope_cache[1].cosv || !m->rope_cache[1].sinv ||
        !m->rope_cache[2].cosv || !m->rope_cache[2].sinv) {
        fprintf(stderr, "dsv4: OOM state buffers\n");
        dsv4_model_close(m);
        return NULL;
    }

    /* per-layer runtime */
    m->run = (DSV4LayerRun *)calloc((size_t)m->cfg.n_layers, sizeof(DSV4LayerRun));
    if (!m->run) { fprintf(stderr, "dsv4: OOM run state\n"); dsv4_model_close(m); return NULL; }
    for (int L = 0; L < m->cfg.n_layers; L++) {
        DSV4LayerRun *r = &m->run[L];
        int ratio = m->cfg.compress_ratio[L];
        int coff = (ratio == 4) ? 2 : 1;
        int max_comp = ratio ? 1 + (context + ratio - 1) / ratio : 0;   /* +1 slack */
        r->kv_cache = (float *)calloc((size_t)(m->cfg.window + max_comp) * m->cfg.head_dim, sizeof(float));
        r->kv_cap = m->cfg.window + max_comp;
        if (ratio) {
            /* the main compressor projects to coff*head_dim; the indexer (ratio
             * 4) projects to coff*index_dim. Separate buffers so the two
             * compressors' running state never clobbers each other. */
            size_t nstate = (size_t)coff * ratio * coff * m->cfg.head_dim;
            r->kv_state = (float *)calloc(nstate, sizeof(float));
            r->score_state = (float *)malloc(nstate * sizeof(float));
            if (r->score_state)
                for (size_t i = 0; i < nstate; i++) r->score_state[i] = -INFINITY;
            if (ratio == 4) {
                size_t nidx = (size_t)coff * ratio * coff * m->cfg.index_dim;
                r->idx_kv_state = (float *)calloc(nidx, sizeof(float));
                r->idx_score_state = (float *)malloc(nidx * sizeof(float));
                if (r->idx_score_state)
                    for (size_t i = 0; i < nidx; i++) r->idx_score_state[i] = -INFINITY;
                r->idx_cache = (float *)calloc((size_t)max_comp * m->cfg.index_dim, sizeof(float));
            }
        }
        if (!r->kv_cache || (ratio && (!r->kv_state || !r->score_state)) ||
            (ratio == 4 && (!r->idx_kv_state || !r->idx_score_state || !r->idx_cache))) {
            fprintf(stderr, "dsv4: OOM layer %d run state\n", L);
            dsv4_model_close(m);
            return NULL;
        }
    }

    /* expert cache: bounded LRU, slots preallocated */
    m->cache_slots = (int)(cache_bytes / EXPERT_SLOT);
    if (m->cache_slots < 1) m->cache_slots = 1;
    m->draft_cache_slots = 0;
    if (enable_dspark && m->cfg.dspark_stages > 0) {
        int verify_positions = m->cfg.dspark_block_size;
        const char *block_env = getenv("DSV4_DSPARK_VERIFY_DRAFTS");
        if (block_env) {
            char *end = NULL;
            long requested = strtol(block_env, &end, 10);
            if (end != block_env && *end == '\0' && requested >= 1 &&
                requested <= m->cfg.dspark_block_size)
                verify_positions = (int)requested;
        }
        int draft_positions = (m->cfg.dspark_block_size + 1) / 2;
        if (draft_positions > verify_positions)
            draft_positions = verify_positions;
        const char *cache_positions_env =
            getenv("DSV4_DSPARK_DRAFT_CACHE_POSITIONS");
        if (cache_positions_env) {
            char *end = NULL;
            long requested = strtol(cache_positions_env, &end, 10);
            if (end != cache_positions_env && *end == '\0' && requested >= 1 &&
                requested <= m->cfg.dspark_block_size)
                draft_positions = (int)requested;
        }
        int ideal = m->cfg.dspark_stages * draft_positions * m->cfg.topk;
        int cap = m->cache_slots / 4;
        if (ideal > cap) ideal = cap;
        ideal -= ideal % m->cfg.dspark_stages;
        if (ideal >= m->cfg.dspark_stages * m->cfg.topk)
            m->draft_cache_slots = ideal;
    }
    m->main_cache_slots = m->cache_slots - m->draft_cache_slots;
    m->cache = (DSV4ExpertSlot *)calloc((size_t)m->cache_slots, sizeof(DSV4ExpertSlot));
    if (!m->cache) { fprintf(stderr, "dsv4: OOM expert cache\n"); dsv4_model_close(m); return NULL; }
    for (int i = 0; i < m->cache_slots; i++) {
        m->cache[i].layer = -1;
        m->cache[i].expert = -1;
        m->cache[i].scales = NULL;
        m->cache[i].weights = NULL;
    }
    m->cache_per_layer = getenv("DSV4_EXPERT_CACHE_GLOBAL") == NULL &&
                         m->main_cache_slots >= m->cfg.n_layers * m->cfg.topk;
    m->cache_slru = m->cache_per_layer &&
                    getenv("DSV4_EXPERT_CACHE_LRU") == NULL;
    m->cache_arc = m->cache_per_layer &&
                   getenv("DSV4_EXPERT_CACHE_ARC") != NULL;
    if (m->cache_arc) m->cache_slru = 0;
    m->cache_lrfu = m->cache_per_layer &&
                     getenv("DSV4_EXPERT_CACHE_LRFU") != NULL;
    if (m->cache_lrfu) {
        m->cache_slru = 0;
        m->cache_arc = 0;
    }
    m->cache_hash_min = m->cache_per_layer && m->cfg.n_hash_layers > 0 &&
                        m->cfg.n_hash_layers < m->cfg.n_layers &&
                        getenv("DSV4_EXPERT_CACHE_EQUAL") == NULL;
    if (m->cache_hash_min) {
        const int learned_layers = m->cfg.n_layers - m->cfg.n_hash_layers;
        const int max_hash_slots =
            (m->main_cache_slots - learned_layers * m->cfg.topk) /
            m->cfg.n_hash_layers;
        int hash_slots = m->cfg.topk;
        const char *hash_slots_env = getenv("DSV4_HASH_CACHE_SLOTS");
        if (hash_slots_env) {
            char *end = NULL;
            long requested = strtol(hash_slots_env, &end, 10);
            if (end != hash_slots_env && *end == '\0' && requested >= m->cfg.topk &&
                requested <= m->cfg.n_experts)
                hash_slots = (int)requested;
        }
        if (hash_slots > max_hash_slots) hash_slots = max_hash_slots;
        if (hash_slots < m->cfg.topk) hash_slots = m->cfg.topk;
        m->cache_hash_slots = hash_slots;
    }
    if (m->cache_slru || m->cache_arc) {
        const int policy_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        m->cache_protected = (int *)calloc((size_t)policy_layers,
                                           sizeof(*m->cache_protected));
        if (!m->cache_protected) {
            fprintf(stderr, "dsv4: OOM expert cache policy state\n");
            dsv4_model_close(m);
            return NULL;
        }
    }
    if (m->cache_arc) {
        const int policy_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        const size_t policy_keys = (size_t)policy_layers * m->cfg.n_experts;
        m->cache_arc_target = (int *)calloc(
            (size_t)policy_layers, sizeof(*m->cache_arc_target));
        m->cache_ghost_b1 = (int *)calloc(
            (size_t)policy_layers, sizeof(*m->cache_ghost_b1));
        m->cache_ghost_b2 = (int *)calloc(
            (size_t)policy_layers, sizeof(*m->cache_ghost_b2));
        m->cache_ghost = (unsigned char *)calloc(
            policy_keys, sizeof(*m->cache_ghost));
        m->cache_ghost_last = (int64_t *)calloc(
            policy_keys, sizeof(*m->cache_ghost_last));
        if (!m->cache_arc_target || !m->cache_ghost_b1 ||
            !m->cache_ghost_b2 || !m->cache_ghost ||
            !m->cache_ghost_last) {
            fprintf(stderr, "dsv4: OOM ARC expert cache policy state\n");
            dsv4_model_close(m);
            return NULL;
        }
    }
    if (m->cache_lrfu) {
        const int policy_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        const size_t policy_keys = (size_t)policy_layers * m->cfg.n_experts;
        m->cache_frequency = (float *)calloc(policy_keys,
                                              sizeof(*m->cache_frequency));
        m->cache_frequency_at = (int64_t *)calloc(
            policy_keys, sizeof(*m->cache_frequency_at));
        m->cache_layer_clock = (int64_t *)calloc(
            (size_t)policy_layers, sizeof(*m->cache_layer_clock));
        if (!m->cache_frequency || !m->cache_frequency_at ||
            !m->cache_layer_clock) {
            fprintf(stderr, "dsv4: OOM expert cache frequency state\n");
            dsv4_model_close(m);
            return NULL;
        }
    }
    {
        const char *trace_path = getenv("DSV4_EXPERT_TRACE");
        if (trace_path && trace_path[0]) {
            m->expert_trace = fopen(trace_path, "wx");
            if (m->expert_trace) {
                fputs("# layer\texpert\tposition\n", m->expert_trace);
            } else {
                fprintf(stderr, "dsv4: cannot create expert trace %s "
                        "(the path must not already exist)\n", trace_path);
            }
        }
    }
    (void)expert_ring_init(m);
    (void)expert_pool_init(m);
    if (!dspark_open(m)) {
        fprintf(stderr, "dsv4: OOM DSpark runtime\n");
        dsv4_model_close(m);
        return NULL;
    }
    return m;
}

void dsv4_model_close(DSV4Model *m)
{
    if (!m) return;
    const int packed_wo_a_mapped =
        m->packed_wo_a && m->layer_shard_map != NULL;
    expert_pool_close(m);
    expert_ring_close(m);
    if (m->expert_trace) fclose(m->expert_trace);
    dspark_close(m);
    if (m->head_map_base) munmap(m->head_map_base, m->head_map_len);
    if (m->layer_shard_map) {
        for (int i = 0; i < m->st.nshard; i++)
            if (m->layer_shard_map[i])
                munmap(m->layer_shard_map[i], m->layer_shard_map_len[i]);
        free(m->layer_shard_map);
        free(m->layer_shard_map_len);
    }
    if (m->st.fd) k3_st_close(&m->st);
    free((void *)m->norm);
    free((void *)m->hc_head_fn);
    free((void *)m->hc_head_base);
    free((void *)m->hc_head_scale);
    if (m->run) {
        for (int L = 0; L < m->cfg.n_layers; L++) {
            free(m->run[L].kv_cache);
            free(m->run[L].kv_state);
            free(m->run[L].score_state);
            free(m->run[L].idx_kv_state);
            free(m->run[L].idx_score_state);
            free(m->run[L].idx_cache);
        }
        free(m->run);
    }
    free(m->meta);
    if (m->wo_a_cache) {
        const int weight_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        for (int L = 0; L < weight_layers; L++) free(m->wo_a_cache[L]);
        free(m->wo_a_cache);
    }
    if (m->wo_a_code_cache) {
        const int weight_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        if (!packed_wo_a_mapped)
            for (int L = 0; L < weight_layers; L++) {
                free(m->wo_a_code_cache[L]);
                free(m->wo_a_scale_cache[L]);
            }
        free(m->wo_a_code_cache);
        free(m->wo_a_scale_cache);
    }
    if (m->prefetch) {
        const int weight_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        for (int L = 0; L < weight_layers; L++) free(m->prefetch[L].tensor);
        free(m->prefetch);
    }
    if (m->cache) {
        for (int i = 0; i < m->cache_slots; i++) {
            if (m->cache[i].scales_base) free(m->cache[i].scales_base);
            if (m->cache[i].weights_base) free(m->cache[i].weights_base);
        }
        free(m->cache);
    }
    for (int i = 0; i < DSV4_MAX_TOPK; i++) {
        free(m->route_prefetch_temp[i].scales_base);
        free(m->route_prefetch_temp[i].weights_base);
    }
    free(m->cache_protected);
    free(m->cache_arc_target);
    free(m->cache_ghost_b1);
    free(m->cache_ghost_b2);
    free(m->cache_ghost);
    free(m->cache_ghost_last);
    free(m->cache_frequency);
    free(m->cache_frequency_at);
    free(m->cache_layer_clock);
    free(m->state);
    free(m->mixes_buf);
    free(m->hc_state_buf);
    free(m->head_hidden);
    free(m->head_buf);
    free(m->scratch);
    free(m->expert_work);
    free(m->route_scratch);
    free(m->attention_scratch);
    for (int k = 0; k < 3; k++) {
        free(m->rope_cache[k].cosv);
        free(m->rope_cache[k].sinv);
    }
    free(m);
}

void dsv4_model_reset_context(DSV4Model *m)
{
    if (!m || !m->run) return;
    memset(m->state, 0, (size_t)m->cfg.hc_mult * m->cfg.hidden * sizeof(float));
    for (int L = 0; L < m->cfg.n_layers; L++) {
        DSV4LayerRun *r = &m->run[L];
        int ratio = m->cfg.compress_ratio[L];
        int coff = ratio == 4 ? 2 : 1;
        memset(r->kv_cache, 0,
               (size_t)r->kv_cap * m->cfg.head_dim * sizeof(float));
        r->comp_count = 0;
        r->idx_count = 0;
        if (ratio) {
            size_t nstate = (size_t)coff * ratio * coff * m->cfg.head_dim;
            memset(r->kv_state, 0, nstate * sizeof(float));
            for (size_t i = 0; i < nstate; i++) r->score_state[i] = -INFINITY;
        }
        if (ratio == 4) {
            size_t nidx = (size_t)coff * ratio * coff * m->cfg.index_dim;
            size_t max_comp = (size_t)(r->kv_cap - m->cfg.window);
            memset(r->idx_kv_state, 0, nidx * sizeof(float));
            for (size_t i = 0; i < nidx; i++) r->idx_score_state[i] = -INFINITY;
            memset(r->idx_cache, 0,
                   max_comp * (size_t)m->cfg.index_dim * sizeof(float));
        }
    }
    if (m->dspark) {
        memset(m->dspark->target_kv, 0,
               (size_t)m->cfg.dspark_stages * m->cfg.window *
               m->cfg.head_dim * sizeof(float));
        for (int i = 0; i < m->cfg.window; i++)
            m->dspark->target_position[i] = -1;
    }
}

typedef struct {
    float *window_rows;
    float *post_window_rows;
    int *window_slots;
    int n_window_slots;
    float *comp_rows;
    int comp_row_start, n_comp_rows;
    float *idx_rows;
    int idx_row_start, n_idx_rows;
    float *kv_state, *score_state;
    float *idx_kv_state, *idx_score_state;
    float *post_kv_state, *post_score_state;
    float *post_idx_kv_state, *post_idx_score_state;
    int *post_comp_count, *post_idx_count;
    size_t state_elems, idx_state_elems;
    int comp_count, idx_count;
} DSV4SnapshotLayer;

struct DSV4ContextSnapshot {
    DSV4Model *model;
    DSV4SnapshotLayer *layer;
    float *model_state;
    float *post_model_state;
    int max_positions;
    int start_position, n_positions;
    int captured_positions;
    int ready;
};

void dsv4_context_snapshot_free(DSV4ContextSnapshot *s)
{
    if (!s) return;
    if (s->model && s->model->active_snapshot == s)
        s->model->active_snapshot = NULL;
    if (s->layer && s->model) {
        for (int L = 0; L < s->model->cfg.n_layers; L++) {
            DSV4SnapshotLayer *sl = &s->layer[L];
            free(sl->window_rows);
            free(sl->post_window_rows);
            free(sl->window_slots);
            free(sl->comp_rows);
            free(sl->idx_rows);
            free(sl->kv_state);
            free(sl->score_state);
            free(sl->idx_kv_state);
            free(sl->idx_score_state);
            free(sl->post_kv_state);
            free(sl->post_score_state);
            free(sl->post_idx_kv_state);
            free(sl->post_idx_score_state);
            free(sl->post_comp_count);
            free(sl->post_idx_count);
        }
    }
    free(s->layer);
    free(s->model_state);
    free(s->post_model_state);
    free(s);
}

DSV4ContextSnapshot *dsv4_context_snapshot_create(DSV4Model *m,
                                                   int max_positions)
{
    if (!m || !m->run || max_positions <= 0 || max_positions > m->context)
        return NULL;
    DSV4ContextSnapshot *s = (DSV4ContextSnapshot *)calloc(1, sizeof(*s));
    if (!s) return NULL;
    s->model = m;
    s->max_positions = max_positions;
    s->layer = (DSV4SnapshotLayer *)calloc((size_t)m->cfg.n_layers,
                                            sizeof(*s->layer));
    s->model_state = (float *)malloc((size_t)m->cfg.hc_mult * m->cfg.hidden *
                                      sizeof(float));
    s->post_model_state = (float *)malloc((size_t)max_positions *
        m->cfg.hc_mult * m->cfg.hidden * sizeof(float));
    if (!s->layer || !s->model_state || !s->post_model_state) {
        dsv4_context_snapshot_free(s);
        return NULL;
    }

    const int d = m->cfg.head_dim;
    for (int L = 0; L < m->cfg.n_layers; L++) {
        DSV4SnapshotLayer *sl = &s->layer[L];
        const int ratio = m->cfg.compress_ratio[L];
        const int coff = ratio == 4 ? 2 : 1;
        sl->window_rows = (float *)malloc((size_t)max_positions * d * sizeof(float));
        sl->post_window_rows =
            (float *)malloc((size_t)max_positions * d * sizeof(float));
        sl->window_slots = (int *)malloc((size_t)max_positions * sizeof(int));
        if (ratio) {
            sl->state_elems = (size_t)coff * ratio * coff * d;
            sl->comp_rows = (float *)malloc((size_t)max_positions * d * sizeof(float));
            sl->kv_state = (float *)malloc(sl->state_elems * sizeof(float));
            sl->score_state = (float *)malloc(sl->state_elems * sizeof(float));
            sl->post_kv_state = (float *)malloc((size_t)max_positions *
                                                 sl->state_elems * sizeof(float));
            sl->post_score_state = (float *)malloc((size_t)max_positions *
                                                    sl->state_elems * sizeof(float));
            sl->post_comp_count = (int *)malloc((size_t)max_positions *
                                                 sizeof(int));
        }
        if (ratio == 4) {
            sl->idx_state_elems = (size_t)coff * ratio * coff * m->cfg.index_dim;
            sl->idx_rows = (float *)malloc((size_t)max_positions *
                                           m->cfg.index_dim * sizeof(float));
            sl->idx_kv_state = (float *)malloc(sl->idx_state_elems * sizeof(float));
            sl->idx_score_state = (float *)malloc(sl->idx_state_elems * sizeof(float));
            sl->post_idx_kv_state = (float *)malloc((size_t)max_positions *
                sl->idx_state_elems * sizeof(float));
            sl->post_idx_score_state = (float *)malloc((size_t)max_positions *
                sl->idx_state_elems * sizeof(float));
            sl->post_idx_count = (int *)malloc((size_t)max_positions *
                                                sizeof(int));
        }
        if (!sl->window_rows || !sl->post_window_rows || !sl->window_slots ||
            (ratio && (!sl->comp_rows || !sl->kv_state || !sl->score_state ||
                       !sl->post_kv_state || !sl->post_score_state ||
                       !sl->post_comp_count)) ||
            (ratio == 4 && (!sl->idx_rows || !sl->idx_kv_state ||
                            !sl->idx_score_state || !sl->post_idx_kv_state ||
                            !sl->post_idx_score_state || !sl->post_idx_count))) {
            dsv4_context_snapshot_free(s);
            return NULL;
        }
    }
    return s;
}

int dsv4_context_snapshot_take(DSV4ContextSnapshot *s, int start_position,
                               int n_positions)
{
    if (!s || !s->model || start_position < 0 || n_positions <= 0 ||
        n_positions > s->max_positions ||
        start_position > s->model->context - n_positions) return -1;
    DSV4Model *m = s->model;
    const int win = m->cfg.window;
    const int d = m->cfg.head_dim;
    memcpy(s->model_state, m->state,
           (size_t)m->cfg.hc_mult * m->cfg.hidden * sizeof(float));

    for (int L = 0; L < m->cfg.n_layers; L++) {
        DSV4LayerRun *r = &m->run[L];
        DSV4SnapshotLayer *sl = &s->layer[L];
        sl->n_window_slots = 0;
        for (int i = 0; i < n_positions; i++) {
            int slot = (start_position + i) % win;
            int seen = 0;
            for (int j = 0; j < sl->n_window_slots; j++)
                seen |= sl->window_slots[j] == slot;
            if (!seen) sl->window_slots[sl->n_window_slots++] = slot;
        }
        for (int i = 0; i < sl->n_window_slots; i++)
            memcpy(sl->window_rows + (size_t)i * d,
                   r->kv_cache + (size_t)sl->window_slots[i] * d,
                   (size_t)d * sizeof(float));

        sl->comp_count = r->comp_count;
        sl->idx_count = r->idx_count;
        if (sl->state_elems) {
            memcpy(sl->kv_state, r->kv_state, sl->state_elems * sizeof(float));
            memcpy(sl->score_state, r->score_state,
                   sl->state_elems * sizeof(float));
            sl->comp_row_start = r->comp_count;
            sl->n_comp_rows = r->kv_cap - win - sl->comp_row_start;
            if (sl->n_comp_rows > s->max_positions)
                sl->n_comp_rows = s->max_positions;
            if (sl->n_comp_rows > 0)
                memcpy(sl->comp_rows,
                       r->kv_cache + (size_t)(win + sl->comp_row_start) * d,
                       (size_t)sl->n_comp_rows * d * sizeof(float));
        }
        if (sl->idx_state_elems) {
            memcpy(sl->idx_kv_state, r->idx_kv_state,
                   sl->idx_state_elems * sizeof(float));
            memcpy(sl->idx_score_state, r->idx_score_state,
                   sl->idx_state_elems * sizeof(float));
            sl->idx_row_start = r->idx_count;
            sl->n_idx_rows = r->kv_cap - win - sl->idx_row_start;
            if (sl->n_idx_rows > s->max_positions)
                sl->n_idx_rows = s->max_positions;
            if (sl->n_idx_rows > 0)
                memcpy(sl->idx_rows,
                       r->idx_cache + (size_t)sl->idx_row_start * m->cfg.index_dim,
                       (size_t)sl->n_idx_rows * m->cfg.index_dim * sizeof(float));
        }
    }
    s->start_position = start_position;
    s->n_positions = n_positions;
    s->captured_positions = 0;
    s->ready = 1;
    s->model->active_snapshot = s;
    return 0;
}

int dsv4_context_snapshot_restore(DSV4ContextSnapshot *s)
{
    if (!s || !s->model || !s->ready) return -1;
    DSV4Model *m = s->model;
    if (m->active_snapshot == s) m->active_snapshot = NULL;
    const int win = m->cfg.window;
    const int d = m->cfg.head_dim;
    memcpy(m->state, s->model_state,
           (size_t)m->cfg.hc_mult * m->cfg.hidden * sizeof(float));
    for (int L = 0; L < m->cfg.n_layers; L++) {
        DSV4LayerRun *r = &m->run[L];
        DSV4SnapshotLayer *sl = &s->layer[L];
        for (int i = 0; i < sl->n_window_slots; i++)
            memcpy(r->kv_cache + (size_t)sl->window_slots[i] * d,
                   sl->window_rows + (size_t)i * d,
                   (size_t)d * sizeof(float));
        if (sl->state_elems) {
            memcpy(r->kv_state, sl->kv_state, sl->state_elems * sizeof(float));
            memcpy(r->score_state, sl->score_state,
                   sl->state_elems * sizeof(float));
            if (sl->n_comp_rows > 0)
                memcpy(r->kv_cache + (size_t)(win + sl->comp_row_start) * d,
                       sl->comp_rows,
                       (size_t)sl->n_comp_rows * d * sizeof(float));
        }
        if (sl->idx_state_elems) {
            memcpy(r->idx_kv_state, sl->idx_kv_state,
                   sl->idx_state_elems * sizeof(float));
            memcpy(r->idx_score_state, sl->idx_score_state,
                   sl->idx_state_elems * sizeof(float));
            if (sl->n_idx_rows > 0)
                memcpy(r->idx_cache + (size_t)sl->idx_row_start * m->cfg.index_dim,
                       sl->idx_rows,
                       (size_t)sl->n_idx_rows * m->cfg.index_dim * sizeof(float));
        }
        r->comp_count = sl->comp_count;
        r->idx_count = sl->idx_count;
    }
    return 0;
}

static void context_snapshot_capture_layer(DSV4Model *m, int layer,
                                           int position)
{
    DSV4ContextSnapshot *s = m->active_snapshot;
    if (!s || !s->ready || layer < 0 || layer >= m->cfg.n_layers ||
        position < s->start_position ||
        position >= s->start_position + s->n_positions) return;
    const int step = position - s->start_position;
    const int slot = position % m->cfg.window;
    DSV4LayerRun *r = &m->run[layer];
    DSV4SnapshotLayer *sl = &s->layer[layer];
    memcpy(sl->post_window_rows + (size_t)step * m->cfg.head_dim,
           r->kv_cache + (size_t)slot * m->cfg.head_dim,
           (size_t)m->cfg.head_dim * sizeof(float));
    if (sl->state_elems) {
        memcpy(sl->post_kv_state + (size_t)step * sl->state_elems,
               r->kv_state, sl->state_elems * sizeof(float));
        memcpy(sl->post_score_state + (size_t)step * sl->state_elems,
               r->score_state, sl->state_elems * sizeof(float));
        sl->post_comp_count[step] = r->comp_count;
    }
    if (sl->idx_state_elems) {
        memcpy(sl->post_idx_kv_state + (size_t)step * sl->idx_state_elems,
               r->idx_kv_state, sl->idx_state_elems * sizeof(float));
        memcpy(sl->post_idx_score_state + (size_t)step * sl->idx_state_elems,
               r->idx_score_state, sl->idx_state_elems * sizeof(float));
        sl->post_idx_count[step] = r->idx_count;
    }
}

static void context_snapshot_capture_model_state(DSV4Model *m,
                                                 const float *states,
                                                 int n_tokens,
                                                 int start_position)
{
    DSV4ContextSnapshot *s = m->active_snapshot;
    if (!s || !s->ready || start_position != s->start_position ||
        n_tokens != s->n_positions) return;
    const size_t stride = (size_t)m->cfg.hc_mult * m->cfg.hidden;
    memcpy(s->post_model_state, states,
           (size_t)n_tokens * stride * sizeof(float));
    s->captured_positions = n_tokens;
}

int dsv4_context_snapshot_commit_prefix(DSV4ContextSnapshot *s,
                                         int n_positions)
{
    if (!s || !s->model || !s->ready || n_positions <= 0 ||
        n_positions > s->n_positions ||
        s->captured_positions < n_positions) return -1;
    DSV4Model *m = s->model;
    const int step = n_positions - 1;
    const int win = m->cfg.window;
    const int d = m->cfg.head_dim;
    const size_t model_stride = (size_t)m->cfg.hc_mult * m->cfg.hidden;

    memcpy(m->state, s->post_model_state + (size_t)step * model_stride,
           model_stride * sizeof(float));
    for (int L = 0; L < m->cfg.n_layers; L++) {
        DSV4LayerRun *r = &m->run[L];
        DSV4SnapshotLayer *sl = &s->layer[L];
        for (int i = 0; i < sl->n_window_slots; i++)
            memcpy(r->kv_cache + (size_t)sl->window_slots[i] * d,
                   sl->window_rows + (size_t)i * d,
                   (size_t)d * sizeof(float));
        for (int p = 0; p < n_positions; p++)
            memcpy(r->kv_cache +
                       (size_t)((s->start_position + p) % win) * d,
                   sl->post_window_rows + (size_t)p * d,
                   (size_t)d * sizeof(float));
        if (sl->state_elems) {
            memcpy(r->kv_state,
                   sl->post_kv_state + (size_t)step * sl->state_elems,
                   sl->state_elems * sizeof(float));
            memcpy(r->score_state,
                   sl->post_score_state + (size_t)step * sl->state_elems,
                   sl->state_elems * sizeof(float));
            r->comp_count = sl->post_comp_count[step];
            int restore_start = r->comp_count - sl->comp_row_start;
            if (restore_start < 0) restore_start = 0;
            if (restore_start < sl->n_comp_rows)
                memcpy(r->kv_cache + (size_t)(win + r->comp_count) * d,
                       sl->comp_rows + (size_t)restore_start * d,
                       (size_t)(sl->n_comp_rows - restore_start) * d *
                           sizeof(float));
        }
        if (sl->idx_state_elems) {
            memcpy(r->idx_kv_state,
                   sl->post_idx_kv_state + (size_t)step * sl->idx_state_elems,
                   sl->idx_state_elems * sizeof(float));
            memcpy(r->idx_score_state,
                   sl->post_idx_score_state + (size_t)step * sl->idx_state_elems,
                   sl->idx_state_elems * sizeof(float));
            r->idx_count = sl->post_idx_count[step];
            int restore_start = r->idx_count - sl->idx_row_start;
            if (restore_start < 0) restore_start = 0;
            if (restore_start < sl->n_idx_rows)
                memcpy(r->idx_cache +
                           (size_t)r->idx_count * m->cfg.index_dim,
                       sl->idx_rows +
                           (size_t)restore_start * m->cfg.index_dim,
                       (size_t)(sl->n_idx_rows - restore_start) *
                           m->cfg.index_dim * sizeof(float));
        }
    }
    if (m->active_snapshot == s) m->active_snapshot = NULL;
    return 0;
}

const DSV4Config *dsv4_model_config(const DSV4Model *m) { return &m->cfg; }

/* ====================================================== expert cache === */

/* Lazily discovered per-expert shard geometry (DSV4ExpertMeta in the header). */
static DSV4ExpertMeta *expert_meta(DSV4Model *m, int layer, int expert)
{
    const int weight_layers = m->cfg.n_layers + m->cfg.dspark_stages;
    if (layer < 0 || layer >= weight_layers || expert < 0 ||
        expert >= m->cfg.n_experts) {
        fprintf(stderr, "dsv4: invalid expert key %d.%d\n", layer, expert);
        exit(1);
    }
    /* metadata lives in a side table allocated per model: main layers followed
     * by the virtual mtp stages, each with n_experts entries. */
    if (!m->meta) {
        m->meta = (DSV4ExpertMeta *)calloc((size_t)weight_layers * m->cfg.n_experts,
                                           sizeof(DSV4ExpertMeta));
        if (!m->meta) { fprintf(stderr, "dsv4: OOM expert meta\n"); exit(1); }
    }
    DSV4ExpertMeta *em = &m->meta[(size_t)layer * m->cfg.n_experts + expert];
    if (!em->valid) {
        char name[512];
        const int is_draft = layer >= m->cfg.n_layers;
        const char *scope = is_draft ? "mtp" : "layers";
        const int scope_layer = is_draft ? layer - m->cfg.n_layers : layer;
        const K3Tensor *ts, *tw;
        int64_t s0 = INT64_MAX, s1 = 0, w0 = INT64_MAX, w1 = 0;
        for (int i = 0; i < 3; i++) {
            const char *suffix = (i == 0) ? "w1" : (i == 1) ? "w2" : "w3";
            snprintf(name, sizeof name, "%s.%d.ffn.experts.%d.%s.scale",
                     scope, scope_layer, expert, suffix);
            ts = k3_st_find(&m->st, name);
            if (!ts) { fprintf(stderr, "dsv4: missing %s\n", name); exit(1); }
            snprintf(name, sizeof name, "%s.%d.ffn.experts.%d.%s.weight",
                     scope, scope_layer, expert, suffix);
            tw = k3_st_find(&m->st, name);
            if (!tw) { fprintf(stderr, "dsv4: missing %s\n", name); exit(1); }
            if (i == 0) em->shard = ts->shard;
            if (ts->shard != em->shard || tw->shard != em->shard) {
                fprintf(stderr, "dsv4: expert %d.%d tensors span shards\n", layer, expert);
                exit(1);
            }
            if (ts->off < s0) s0 = ts->off;
            if (ts->off + ts->nbytes > s1) s1 = ts->off + ts->nbytes;
            if (tw->off < w0) w0 = tw->off;
            if (tw->off + tw->nbytes > w1) w1 = tw->off + tw->nbytes;
        }
        em->scale_off = s0;
        em->weight_off = w0;
        em->scale_run = s1 - s0;
        em->weight_run = w1 - w0;
        em->valid = 1;
        /* contiguity check: the three scale tensors must form one run and the
         * three weight tensors another (their sizes come from the shapes, so
         * this holds for any checkpoint geometry, not just the 166.9 GB one) */
        {
            int64_t want_s = 0, want_w = 0;
            for (int i = 0; i < 3; i++) {
                const char *suffix = (i == 0) ? "w1" : (i == 1) ? "w2" : "w3";
                snprintf(name, sizeof name, "%s.%d.ffn.experts.%d.%s.scale",
                         scope, scope_layer, expert, suffix);
                ts = k3_st_find(&m->st, name);
                snprintf(name, sizeof name, "%s.%d.ffn.experts.%d.%s.weight",
                         scope, scope_layer, expert, suffix);
                tw = k3_st_find(&m->st, name);
                want_s += ts->nbytes;
                want_w += tw->nbytes;
            }
            if (s1 - s0 != want_s || w1 - w0 != want_w) {
                fprintf(stderr, "dsv4: expert %d.%d runs are not contiguous (scale %lld/%lld, weight %lld/%lld)\n",
                        layer, expert, (long long)(s1 - s0), (long long)want_s,
                        (long long)(w1 - w0), (long long)want_w);
                exit(1);
            }
        }
    }
    return em;
}

/* Return the slot range owned by a layer. The default partition keeps one
 * layer's hot experts from displacing another's; undersized caches retain the
 * global policy so every top-k batch can still pin all of its inputs. */
static void cache_bounds(const DSV4Model *m, int layer, int *begin, int *end)
{
    if (layer >= m->cfg.n_layers) {
        const int stage = layer - m->cfg.n_layers;
        if (stage < 0 || stage >= m->cfg.dspark_stages ||
            m->draft_cache_slots == 0) {
            *begin = 0;
            *end = m->main_cache_slots;
            return;
        }
        int base = m->draft_cache_slots / m->cfg.dspark_stages;
        int extra = m->draft_cache_slots % m->cfg.dspark_stages;
        *begin = m->main_cache_slots + stage * base +
                 (stage < extra ? stage : extra);
        *end = *begin + base + (stage < extra);
        return;
    }
    if (!m->cache_per_layer) {
        *begin = 0;
        *end = m->main_cache_slots;
        return;
    }
    if (m->cache_hash_min) {
        const int hash_layers = m->cfg.n_hash_layers;
        const int hash_slots = m->cache_hash_slots;
        const int hash_total = hash_layers * hash_slots;
        if (layer < hash_layers) {
            *begin = layer * hash_slots;
            *end = *begin + hash_slots;
            return;
        }
        const int learned_layers = m->cfg.n_layers - hash_layers;
        const int learned_slots = m->main_cache_slots - hash_total;
        const int base = learned_slots / learned_layers;
        const int extra = learned_slots % learned_layers;
        const int local = layer - hash_layers;
        *begin = hash_total + local * base +
                 (local < extra ? local : extra);
        *end = *begin + base + (local < extra);
        return;
    }
    int base = m->main_cache_slots / m->cfg.n_layers;
    int extra = m->main_cache_slots % m->cfg.n_layers;
    *begin = layer * base + (layer < extra ? layer : extra);
    *end = *begin + base + (layer < extra);
}

int dsv4_expert_cache_set_hash_slots(DSV4Model *m, int hash_slots)
{
    if (!m || !m->cache_hash_min || !m->cache_slru ||
        hash_slots < m->cfg.topk || hash_slots > m->cfg.n_experts)
        return -1;
    const int learned_layers = m->cfg.n_layers - m->cfg.n_hash_layers;
    const int max_hash_slots =
        (m->main_cache_slots - learned_layers * m->cfg.topk) /
        m->cfg.n_hash_layers;
    if (hash_slots > max_hash_slots) hash_slots = max_hash_slots;
    if (hash_slots == m->cache_hash_slots) return 0;

    DSV4ExpertSlot *old = (DSV4ExpertSlot *)malloc(
        (size_t)m->cache_slots * sizeof(*old));
    DSV4ExpertSlot *reordered = (DSV4ExpertSlot *)malloc(
        (size_t)m->cache_slots * sizeof(*reordered));
    unsigned char *used = (unsigned char *)calloc(
        (size_t)m->main_cache_slots, 1);
    unsigned char *filled = (unsigned char *)calloc(
        (size_t)m->main_cache_slots, 1);
    if (!old || !reordered || !used || !filled) {
        free(old); free(reordered); free(used); free(filled);
        return -1;
    }
    memcpy(old, m->cache, (size_t)m->cache_slots * sizeof(*old));
    m->cache_hash_slots = hash_slots;

    for (int layer = 0; layer < m->cfg.n_layers; layer++) {
        int begin, end;
        cache_bounds(m, layer, &begin, &end);
        for (int dst = begin; dst < end; dst++) {
            int best = -1;
            for (int src = 0; src < m->main_cache_slots; src++) {
                if (used[src] || old[src].layer != layer) continue;
                if (best < 0 || old[src].last_use > old[best].last_use)
                    best = src;
            }
            if (best < 0) break;
            reordered[dst] = old[best];
            used[best] = 1;
            filled[dst] = 1;
        }
    }

    int source = 0;
    for (int dst = 0; dst < m->main_cache_slots; dst++) {
        if (filled[dst]) continue;
        while (source < m->main_cache_slots && used[source]) source++;
        if (source >= m->main_cache_slots) {
            free(old); free(reordered); free(used); free(filled);
            return -1;
        }
        reordered[dst] = old[source];
        reordered[dst].scales = NULL;
        reordered[dst].weights = NULL;
        reordered[dst].layer = -1;
        reordered[dst].expert = -1;
        reordered[dst].last_use = 0;
        reordered[dst].segment = 0;
        used[source++] = 1;
    }
    for (int i = m->main_cache_slots; i < m->cache_slots; i++)
        reordered[i] = old[i];
    memcpy(m->cache, reordered,
           (size_t)m->cache_slots * sizeof(*m->cache));

    if (m->cache_protected) {
        const int policy_layers = m->cfg.n_layers + m->cfg.dspark_stages;
        memset(m->cache_protected, 0,
               (size_t)policy_layers * sizeof(*m->cache_protected));
        for (int i = 0; i < m->cache_slots; i++)
            if (m->cache[i].layer >= 0 && m->cache[i].segment == 2)
                m->cache_protected[m->cache[i].layer]++;
    }
    free(old); free(reordered); free(used); free(filled);
    return 0;
}

static void cache_arc_prepare_miss(DSV4Model *m, int layer, int expert)
{
    if (!m->cache_arc) return;
    size_t key = (size_t)layer * m->cfg.n_experts + expert;
    const int ghost = m->cache_ghost[key];
    if (ghost != 3 && ghost != 4) return;
    int b1 = m->cache_ghost_b1[layer];
    int b2 = m->cache_ghost_b2[layer];
    int delta;
    if (ghost == 3) {
        delta = b1 > 0 ? b2 / b1 : 1;
        if (delta < 1) delta = 1;
        m->cache_arc_target[layer] += delta;
        m->cache_ghost_b1[layer]--;
    } else {
        delta = b2 > 0 ? b1 / b2 : 1;
        if (delta < 1) delta = 1;
        m->cache_arc_target[layer] -= delta;
        m->cache_ghost_b2[layer]--;
    }
    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    int cap = end - begin;
    if (m->cache_arc_target[layer] < 0) m->cache_arc_target[layer] = 0;
    if (m->cache_arc_target[layer] > cap) m->cache_arc_target[layer] = cap;
    /* Five/six retain the B1/B2 origin until cache_publish() installs the
     * already-prepared miss in T2. */
    m->cache_ghost[key] = (unsigned char)(ghost + 2);
}

static int cache_victim_better(const DSV4Model *m, int layer,
                               int incoming_expert, int recent_reserved,
                               int candidate, int current, int64_t oldest)
{
    if (current < 0) return 1;
    const int candidate_empty = m->cache[candidate].layer < 0;
    const int current_empty = m->cache[current].layer < 0;
    if (candidate_empty != current_empty) return candidate_empty;
    if (m->cache_arc && !candidate_empty) {
        int begin, end;
        cache_bounds(m, layer, &begin, &end);
        int recent = 0;
        for (int i = begin; i < end; i++)
            recent += m->cache[i].layer >= 0 && m->cache[i].segment == 1;
        recent -= recent_reserved;
        size_t key = (size_t)layer * m->cfg.n_experts + incoming_expert;
        const int incoming_b2 = m->cache_ghost[key] == 4 ||
                                m->cache_ghost[key] == 6;
        const int prefer_recent =
            recent > m->cache_arc_target[layer] ||
            (incoming_b2 && recent == m->cache_arc_target[layer]);
        const int preferred_segment = prefer_recent ? 1 : 2;
        const int candidate_preferred =
            m->cache[candidate].segment == preferred_segment;
        const int current_preferred =
            m->cache[current].segment == preferred_segment;
        if (candidate_preferred != current_preferred)
            return candidate_preferred;
    }
    if (m->cache_lrfu) {
        const DSV4ExpertSlot *candidate_slot = &m->cache[candidate];
        const DSV4ExpertSlot *current_slot = &m->cache[current];
        float candidate_value = 0.0f;
        float current_value = 0.0f;
        if (candidate_slot->layer >= 0) {
            size_t key = (size_t)candidate_slot->layer * m->cfg.n_experts +
                         candidate_slot->expert;
            int64_t age = m->cache_layer_clock[candidate_slot->layer] -
                          m->cache_frequency_at[key];
            candidate_value = m->cache_frequency[key] *
                              exp2f(-(float)age / 96.0f);
        }
        if (current_slot->layer >= 0) {
            size_t key = (size_t)current_slot->layer * m->cfg.n_experts +
                         current_slot->expert;
            int64_t age = m->cache_layer_clock[current_slot->layer] -
                          m->cache_frequency_at[key];
            current_value = m->cache_frequency[key] *
                            exp2f(-(float)age / 96.0f);
        }
        if (candidate_value != current_value)
            return candidate_value < current_value;
    }
    if (m->cache_slru) {
        int candidate_protected = m->cache[candidate].segment == 2;
        int current_protected = m->cache[current].segment == 2;
        if (candidate_protected != current_protected)
            return candidate_protected < current_protected;
    }
    return m->cache[candidate].last_use < oldest;
}

static void cache_record_frequency(DSV4Model *m, int layer, int expert)
{
    if (!m->cache_lrfu || layer < 0 || expert < 0) return;
    size_t key = (size_t)layer * m->cfg.n_experts + expert;
    int64_t now = ++m->cache_layer_clock[layer];
    int64_t age = now - m->cache_frequency_at[key];
    m->cache_frequency[key] = m->cache_frequency[key] *
                              exp2f(-(float)age / 96.0f) + 1.0f;
    m->cache_frequency_at[key] = now;
}

static void cache_touch(DSV4Model *m, int layer, int slot)
{
    DSV4ExpertSlot *s = &m->cache[slot];
    s->last_use = ++m->lru_clock;
    cache_record_frequency(m, layer, s->expert);
    if (m->cache_arc) {
        if (s->segment == 1) {
            s->segment = 2;
            m->cache_protected[layer]++;
        }
        return;
    }
    if (!m->cache_slru || s->segment != 1) return;

    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    int protected_cap = (end - begin) * 7 / 10;
    if (protected_cap < 1) protected_cap = 1;
    if (protected_cap >= end - begin) protected_cap = end - begin - 1;
    if (protected_cap <= 0) return;
    if (m->cache_protected[layer] >= protected_cap) {
        int victim = -1;
        for (int i = begin; i < end; i++) {
            if (i == slot || m->cache[i].segment != 2) continue;
            if (victim < 0 ||
                m->cache[i].last_use < m->cache[victim].last_use)
                victim = i;
        }
        if (victim >= 0) m->cache[victim].segment = 1;
    } else {
        m->cache_protected[layer]++;
    }
    s->segment = 2;
}

static void cache_arc_trim_ghosts(DSV4Model *m, int layer)
{
    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    const int cap = end - begin;
    while (m->cache_ghost_b1[layer] + m->cache_ghost_b2[layer] > cap) {
        int victim = -1;
        int64_t oldest = INT64_MAX;
        for (int expert = 0; expert < m->cfg.n_experts; expert++) {
            size_t key = (size_t)layer * m->cfg.n_experts + expert;
            if (m->cache_ghost[key] != 3 && m->cache_ghost[key] != 4)
                continue;
            if (m->cache_ghost_last[key] < oldest) {
                oldest = m->cache_ghost_last[key];
                victim = expert;
            }
        }
        if (victim < 0) break;
        size_t key = (size_t)layer * m->cfg.n_experts + victim;
        if (m->cache_ghost[key] == 3) m->cache_ghost_b1[layer]--;
        else m->cache_ghost_b2[layer]--;
        m->cache_ghost[key] = 0;
        m->cache_ghost_last[key] = 0;
    }
}

static void cache_publish(DSV4Model *m, int layer, int expert, int slot)
{
    DSV4ExpertSlot *s = &m->cache[slot];
    if ((m->cache_slru || m->cache_arc) && s->segment == 2)
        m->cache_protected[s->layer]--;
    int incoming_ghost = 0;
    if (m->cache_arc) {
        size_t incoming_key = (size_t)layer * m->cfg.n_experts + expert;
        cache_arc_prepare_miss(m, layer, expert);
        incoming_ghost = m->cache_ghost[incoming_key] == 5 ||
                         m->cache_ghost[incoming_key] == 6;
        m->cache_ghost[incoming_key] = 0;
        m->cache_ghost_last[incoming_key] = 0;
        if (s->layer >= 0 && s->expert >= 0) {
            size_t evicted_key = (size_t)s->layer * m->cfg.n_experts +
                                 s->expert;
            const int ghost = s->segment == 2 ? 4 : 3;
            m->cache_ghost[evicted_key] = (unsigned char)ghost;
            m->cache_ghost_last[evicted_key] = ++m->cache_ghost_clock;
            if (ghost == 3) m->cache_ghost_b1[s->layer]++;
            else m->cache_ghost_b2[s->layer]++;
        }
    }
    s->layer = layer;
    s->expert = expert;
    s->segment = m->cache_arc && incoming_ghost ? 2 : 1;
    if (m->cache_arc && incoming_ghost) m->cache_protected[layer]++;
    s->last_use = ++m->lru_clock;
    cache_record_frequency(m, layer, expert);
    if (m->cache_arc) cache_arc_trim_ghosts(m, layer);
}

/* Find a cached slot for (layer, expert); -1 if absent. */
static int cache_find(DSV4Model *m, int layer, int expert)
{
    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    for (int i = begin; i < end; i++)
        if (m->cache[i].layer == layer && m->cache[i].expert == expert) return i;
    return -1;
}

static int expert_mmap_enabled(const DSV4Model *m)
{
    return m->layer_shard_map && getenv("DSV4_EXPERT_MMAP") != NULL;
}

static void expert_mmap_advise(const DSV4Model *m, const DSV4ExpertMeta *em)
{
    long page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) return;

    const int64_t offsets[2] = { em->scale_off, em->weight_off };
    const int64_t lengths[2] = { em->scale_run, em->weight_run };
    for (int i = 0; i < 2; i++) {
        int64_t start = (offsets[i] / page_size) * page_size;
        int64_t finish = offsets[i] + lengths[i];
        int64_t end = ((finish + page_size - 1) / page_size) * page_size;
        if (start < 0 || end < start || (uint64_t)end >
            m->layer_shard_map_len[em->shard]) continue;
        (void)posix_madvise((uint8_t *)m->layer_shard_map[em->shard] + start,
                            (size_t)(end - start), POSIX_MADV_WILLNEED);
    }
}

static void expert_mmap_bind(DSV4Model *m, DSV4ExpertSlot *s,
                             const DSV4ExpertMeta *em)
{
    if (em->shard < 0 || em->shard >= m->st.nshard || em->scale_off < 0 ||
        em->weight_off < 0 || (uint64_t)em->scale_off + em->scale_run >
        m->layer_shard_map_len[em->shard] ||
        (uint64_t)em->weight_off + em->weight_run >
        m->layer_shard_map_len[em->shard]) {
        fprintf(stderr, "dsv4: expert mmap range is outside shard %d\n", em->shard);
        exit(1);
    }
    s->scales = (uint8_t *)m->layer_shard_map[em->shard] + em->scale_off;
    s->weights = (uint8_t *)m->layer_shard_map[em->shard] + em->weight_off;
    expert_mmap_advise(m, em);
}

static void expert_slot_ensure_buffers(DSV4ExpertSlot *s)
{
    if (s->scales_base) return;
    if (posix_memalign((void **)&s->scales_base, 4096,
                       EXPERT_SCALE_RUN + 2 * 4096) != 0) {
        fprintf(stderr, "dsv4: OOM expert slot\n");
        exit(1);
    }
    const int huge = getenv("DSV4_EXPERT_HUGEPAGE") != NULL;
    const size_t alignment = huge ? (2u << 20) : 4096u;
    size_t weight_bytes = EXPERT_WEIGHT_RUN + 2 * 4096;
    if (huge)
        weight_bytes = (weight_bytes + alignment - 1) & ~(alignment - 1);
    if (posix_memalign((void **)&s->weights_base, alignment,
                       weight_bytes) != 0) {
        fprintf(stderr, "dsv4: OOM expert slot\n");
        exit(1);
    }
#if defined(MADV_HUGEPAGE)
    if (huge)
        (void)madvise(s->weights_base, weight_bytes, MADV_HUGEPAGE);
#endif
}

/* Safetensors stores expert names lexicographically, so numeric expert order
 * can jump across large regions of a shard (for example 1 -> 2 skips over
 * 10..199). Issue independent misses in physical order while leaving cache
 * publication and all floating-point work in their original order. */
static int expert_read_order(DSV4ExpertMeta *const *meta, const int *is_miss,
                             int n, int *order)
{
    int count = 0;
    for (int i = 0; i < n; i++)
        if (is_miss[i]) order[count++] = i;
    if (getenv("DSV4_EXPERT_IO_ID_ORDER")) return count;

    for (int i = 1; i < count; i++) {
        int item = order[i];
        int j = i;
        while (j > 0) {
            const DSV4ExpertMeta *a = meta[order[j - 1]];
            const DSV4ExpertMeta *b = meta[item];
            if (a->shard < b->shard ||
                (a->shard == b->shard && a->weight_off <= b->weight_off))
                break;
            order[j] = order[j - 1];
            j--;
        }
        order[j] = item;
    }
    return count;
}

/* Read one expert into a cache slot. Two page-aligned coalesced preads: the
 * scale run then the weight run; each gets its own page-aligned buffer with
 * room for the O_DIRECT widening window. Falls back to per-tensor reads if a
 * run is not contiguous (checked once in expert_meta). */
static void expert_load(DSV4Model *m, int layer, int expert, int slot)
{
    DSV4ExpertMeta *em = expert_meta(m, layer, expert);
    DSV4ExpertSlot *s = &m->cache[slot];
    if (expert_mmap_enabled(m)) {
        expert_mmap_bind(m, s, em);
        cache_publish(m, layer, expert, slot);
        m->cache_misses++;
        m->coalesced_loads += 1;
        m->expert_read_ops += 2;
        m->expert_bytes_read += em->scale_run + em->weight_run;
        return;
    }
    expert_slot_ensure_buffers(s);
    int64_t po;
    int64_t got = k3_st_read_aligned(&m->st, em->shard, em->scale_off, em->scale_run,
                                     s->scales_base, EXPERT_SCALE_RUN + 2 * 4096, &po);
    if (got != em->scale_run) {
        fprintf(stderr, "dsv4: short read expert scales %d.%d\n", layer, expert);
        exit(1);
    }
    s->scales = s->scales_base + po;
    got = k3_st_read_aligned(&m->st, em->shard, em->weight_off, em->weight_run,
                             s->weights_base, EXPERT_WEIGHT_RUN + 2 * 4096, &po);
    if (got != em->weight_run) {
        fprintf(stderr, "dsv4: short read expert weights %d.%d\n", layer, expert);
        exit(1);
    }
    s->weights = s->weights_base + po;
    cache_publish(m, layer, expert, slot);
    m->cache_misses++;
    m->coalesced_loads += 1;
    m->expert_read_ops += 2;
    m->expert_bytes_read += em->scale_run + em->weight_run;
}

/* Ensure (layer, expert) is resident. Returns a pointer to its slot. */
static DSV4ExpertSlot *expert_get(DSV4Model *m, int layer, int expert,
                                  const int *pinned, int npin)
{
    int idx = cache_find(m, layer, expert);
    if (idx >= 0) {
        if (expert_mmap_enabled(m))
            expert_mmap_advise(m, expert_meta(m, layer, expert));
        cache_touch(m, layer, idx);
        m->cache_hits++;
        return &m->cache[idx];
    }
    /* victim: least recently used slot not in the pinned set */
    int victim = -1;
    int64_t oldest = INT64_MAX;
    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    cache_arc_prepare_miss(m, layer, expert);
    for (int i = begin; i < end; i++) {
        int ispin = 0;
        for (int j = 0; j < npin; j++) if (pinned[j] == i) { ispin = 1; break; }
        if (ispin) continue;
        if (cache_victim_better(m, layer, expert, 0,
                                i, victim, oldest)) {
            oldest = m->cache[i].last_use;
            victim = i;
        }
    }
    if (victim < 0) victim = 0;   /* all pinned: overwrite slot 0 (only reachable
                                     when npin >= slots, impossible for topk=6) */
    expert_load(m, layer, expert, victim);
    return &m->cache[victim];
}

/* Batch fetch: parse hits, reserve victims, then load misses concurrently with
 * OpenMP dynamic scheduling; the LRU and counters are only touched by the
 * main thread after the reads finish. Falls back to serial expert_get when the
 * cache cannot hold the working set. */
static int expert_get_many(DSV4Model *m, int layer, const int *experts, int n,
                           int *slot_ids, int *is_miss)
{
    int miss_count = 0;
    for (int i = 0; i < n; i++) {
        int idx = cache_find(m, layer, experts[i]);
        if (idx >= 0) {
            slot_ids[i] = idx;
            is_miss[i] = 0;
        } else {
            slot_ids[i] = -1;
            is_miss[i] = 1;
            miss_count++;
        }
    }
    if (miss_count == 0) {
        for (int i = 0; i < n; i++) {
            if (expert_mmap_enabled(m))
                expert_mmap_advise(m, expert_meta(m, layer, experts[i]));
            cache_touch(m, layer, slot_ids[i]);
            m->cache_hits++;
        }
        return 0;
    }

    /* Find victims for the misses. Every hit is pinned until the whole batch has
     * been evaluated; overwriting one here makes slot_ids for that hit silently
     * refer to a different expert. */
    int victim[DSV4_MAX_TOPK];
    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    for (int i = 0; i < n; i++) victim[i] = -1;
    for (int mi = 0; mi < n; mi++) {
        if (!is_miss[mi]) continue;
        cache_arc_prepare_miss(m, layer, experts[mi]);
        int v = -1;
        int64_t oldest = INT64_MAX;
        int recent_reserved = 0;
        for (int j = 0; j < n; j++)
            if (victim[j] >= 0 && m->cache[victim[j]].segment == 1)
                recent_reserved++;
        for (int si = begin; si < end; si++) {
            int taken = 0;
            for (int j = 0; j < n; j++) {
                if (!is_miss[j] && slot_ids[j] == si) { taken = 1; break; }
            }
            for (int j = 0; j < n; j++) if (victim[j] == si) { taken = 1; break; }
            if (taken) continue;
            if (cache_victim_better(m, layer, experts[mi], recent_reserved,
                                    si, v, oldest)) {
                oldest = m->cache[si].last_use;
                v = si;
            }
        }
        if (v < 0) {
            /* cache too small: fall back to serial loads */
            for (int i = 0; i < n; i++) {
                if (!is_miss[i]) continue;
                DSV4ExpertSlot *s = expert_get(m, layer, experts[i], slot_ids, i);
                slot_ids[i] = (int)(s - m->cache);
            }
            return 0;
        }
        victim[mi] = v;
    }

    /* Resolve the lazy metadata on the main thread. expert_meta() populates a
     * shared table and is deliberately not called from the parallel region. */
    DSV4ExpertMeta *batch_meta[DSV4_MAX_TOPK];
    for (int i = 0; i < n; i++)
        batch_meta[i] = is_miss[i] ? expert_meta(m, layer, experts[i]) : NULL;
    int read_order[DSV4_MAX_TOPK];
    int read_count = expert_read_order(batch_meta, is_miss, n, read_order);

    if (expert_mmap_enabled(m)) {
        for (int i = 0; i < n; i++) {
            if (!is_miss[i]) continue;
            DSV4ExpertMeta *em = batch_meta[i];
            DSV4ExpertSlot *s = &m->cache[victim[i]];
            expert_mmap_bind(m, s, em);
            cache_publish(m, layer, experts[i], victim[i]);
            slot_ids[i] = victim[i];
            m->cache_misses++;
            m->coalesced_loads += 1;
            m->expert_read_ops += 2;
            m->expert_bytes_read += em->scale_run + em->weight_run;
        }
        for (int i = 0; i < n; i++) {
            if (is_miss[i]) continue;
            expert_mmap_advise(m, expert_meta(m, layer, experts[i]));
            cache_touch(m, layer, slot_ids[i]);
            m->cache_hits++;
        }
        m->batch_prefetched += miss_count;
        return 0;
    }

    /* concurrent reads: every thread writes its own slot only */
    double read_start = m->profiling ? now_s() : 0.0;
#ifdef _OPENMP
    int io_threads = m->threads;
    if (io_threads > read_count) io_threads = read_count;
    #pragma omp parallel for schedule(dynamic, 1) num_threads(io_threads)
#endif
    for (int ri = 0; ri < read_count; ri++) {
        int i = read_order[ri];
        DSV4ExpertMeta *em = batch_meta[i];
        DSV4ExpertSlot *s = &m->cache[victim[i]];
        expert_slot_ensure_buffers(s);
        int64_t po;
        (void)k3_st_read_aligned(&m->st, em->shard, em->scale_off, em->scale_run,
                                 s->scales_base, EXPERT_SCALE_RUN + 2 * 4096, &po);
        s->scales = s->scales_base + po;
        (void)k3_st_read_aligned(&m->st, em->shard, em->weight_off, em->weight_run,
                                 s->weights_base, EXPERT_WEIGHT_RUN + 2 * 4096, &po);
        s->weights = s->weights_base + po;
    }
    if (m->profiling) {
        double elapsed = now_s() - read_start;
        m->time_expert_read += elapsed;
        m->time_expert_io += elapsed;
    }
    /* publish serially */
    for (int i = 0; i < n; i++) {
        if (!is_miss[i]) continue;
        DSV4ExpertMeta *em = batch_meta[i];
        cache_publish(m, layer, experts[i], victim[i]);
        slot_ids[i] = victim[i];
        m->cache_misses++;
        m->coalesced_loads += 1;
        m->expert_read_ops += 2;
        m->expert_bytes_read += em->scale_run + em->weight_run;
    }
    for (int i = 0; i < n; i++) {
        if (is_miss[i]) continue;
        cache_touch(m, layer, slot_ids[i]);
        m->cache_hits++;
    }
    m->batch_prefetched += miss_count;
    return miss_count;
}

typedef struct {
    void *owner;
    int phase;
    int expected;
} DSV4RingRead;

struct DSV4IoRing {
#ifdef __linux__
    int fd;
    void *sq_map;
    void *cq_map;
    size_t sq_map_len;
    size_t cq_map_len;
    struct io_uring_sqe *sqes;
    size_t sqes_len;
    unsigned *sq_head;
    unsigned *sq_tail;
    unsigned *sq_mask;
    unsigned *sq_entries;
    unsigned *sq_array;
    unsigned *cq_head;
    unsigned *cq_tail;
    unsigned *cq_mask;
    struct io_uring_cqe *cqes;
#else
    int unavailable;
#endif
};

#ifdef __linux__
static DSV4IoRing *io_ring_open(unsigned entries)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    int fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (fd < 0) return NULL;

    DSV4IoRing *ring = (DSV4IoRing *)calloc(1, sizeof(*ring));
    if (!ring) {
        close(fd);
        return NULL;
    }
    ring->fd = fd;
    ring->sq_map_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    ring->cq_map_len = p.cq_off.cqes +
        p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        if (ring->cq_map_len > ring->sq_map_len)
            ring->sq_map_len = ring->cq_map_len;
        ring->cq_map_len = ring->sq_map_len;
    }
    ring->sq_map = mmap(NULL, ring->sq_map_len, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    if (ring->sq_map == MAP_FAILED) goto fail;
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        ring->cq_map = ring->sq_map;
    } else {
        ring->cq_map = mmap(NULL, ring->cq_map_len, PROT_READ | PROT_WRITE,
                            MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (ring->cq_map == MAP_FAILED) goto fail;
    }
    ring->sqes_len = p.sq_entries * sizeof(struct io_uring_sqe);
    ring->sqes = (struct io_uring_sqe *)mmap(
        NULL, ring->sqes_len, PROT_READ | PROT_WRITE,
        MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (ring->sqes == MAP_FAILED) goto fail;

    ring->sq_head = (unsigned *)((char *)ring->sq_map + p.sq_off.head);
    ring->sq_tail = (unsigned *)((char *)ring->sq_map + p.sq_off.tail);
    ring->sq_mask = (unsigned *)((char *)ring->sq_map + p.sq_off.ring_mask);
    ring->sq_entries =
        (unsigned *)((char *)ring->sq_map + p.sq_off.ring_entries);
    ring->sq_array = (unsigned *)((char *)ring->sq_map + p.sq_off.array);
    ring->cq_head = (unsigned *)((char *)ring->cq_map + p.cq_off.head);
    ring->cq_tail = (unsigned *)((char *)ring->cq_map + p.cq_off.tail);
    ring->cq_mask = (unsigned *)((char *)ring->cq_map + p.cq_off.ring_mask);
    ring->cqes =
        (struct io_uring_cqe *)((char *)ring->cq_map + p.cq_off.cqes);
    return ring;

fail:
    if (ring->sqes && ring->sqes != MAP_FAILED)
        munmap(ring->sqes, ring->sqes_len);
    if (ring->cq_map && ring->cq_map != MAP_FAILED &&
        ring->cq_map != ring->sq_map)
        munmap(ring->cq_map, ring->cq_map_len);
    if (ring->sq_map && ring->sq_map != MAP_FAILED)
        munmap(ring->sq_map, ring->sq_map_len);
    close(fd);
    free(ring);
    return NULL;
}

static void io_ring_free(DSV4IoRing *ring)
{
    if (!ring) return;
    munmap(ring->sqes, ring->sqes_len);
    if (ring->cq_map != ring->sq_map)
        munmap(ring->cq_map, ring->cq_map_len);
    munmap(ring->sq_map, ring->sq_map_len);
    close(ring->fd);
    free(ring);
}

static int io_ring_submit(DSV4IoRing *ring, DSV4RingRead *read,
                          int fd, void *buf, unsigned len, int64_t offset)
{
    unsigned head = __atomic_load_n(ring->sq_head, __ATOMIC_ACQUIRE);
    unsigned tail = *ring->sq_tail;
    if (tail - head >= *ring->sq_entries) return 0;
    unsigned index = tail & *ring->sq_mask;
    struct io_uring_sqe *sqe = &ring->sqes[index];
    memset(sqe, 0, sizeof(*sqe));
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->off = (uint64_t)offset;
    sqe->addr = (uint64_t)(uintptr_t)buf;
    sqe->len = len;
    sqe->user_data = (uint64_t)(uintptr_t)read;
    ring->sq_array[index] = index;
    __atomic_store_n(ring->sq_tail, tail + 1, __ATOMIC_RELEASE);
    int rc;
    do {
        rc = (int)syscall(__NR_io_uring_enter, ring->fd, 1, 0, 0, NULL, 0);
    } while (rc < 0 && errno == EINTR);
    return rc == 1;
}

static int io_ring_wait(DSV4IoRing *ring, DSV4RingRead **read, int *result)
{
    for (;;) {
        unsigned head = __atomic_load_n(ring->cq_head, __ATOMIC_ACQUIRE);
        unsigned tail = __atomic_load_n(ring->cq_tail, __ATOMIC_ACQUIRE);
        if (head != tail) {
            struct io_uring_cqe *cqe = &ring->cqes[head & *ring->cq_mask];
            *read = (DSV4RingRead *)(uintptr_t)cqe->user_data;
            *result = cqe->res;
            __atomic_store_n(ring->cq_head, head + 1, __ATOMIC_RELEASE);
            return 1;
        }
        int rc;
        do {
            rc = (int)syscall(__NR_io_uring_enter, ring->fd, 0, 1,
                              IORING_ENTER_GETEVENTS, NULL, 0);
        } while (rc < 0 && errno == EINTR);
        if (rc < 0) return 0;
    }
}
#else
static DSV4IoRing *io_ring_open(unsigned entries)
{
    (void)entries;
    return NULL;
}
static void io_ring_free(DSV4IoRing *ring) { (void)ring; }
#endif

typedef struct DSV4ExpertReadJob DSV4ExpertReadJob;
struct DSV4ExpertReadJob {
    DSV4Model *model;
    DSV4ExpertMeta *meta;
    DSV4ExpertSlot *slot;
    DSV4ExpertPool *pool;
    int phased;
    int phase;
    int ok;
    int l1_ok;
    int l1_done;
    int done;
    int ring_pending;
    int ring_failed;
    int64_t ring_payload_off[2];
    DSV4RingRead ring_read[2];
    double done_at;
};

typedef struct {
    DSV4Model *model;
    const int *experts;
    int layer, n, miss_count;
    int *slot_ids, *is_miss;
    int victim[DSV4_MAX_TOPK];
    DSV4ExpertMeta *meta[DSV4_MAX_TOPK];
    DSV4ExpertReadJob job[DSV4_MAX_TOPK];
    pthread_t thread[DSV4_MAX_TOPK];
    unsigned char started[DSV4_MAX_TOPK];
    unsigned char pooled[DSV4_MAX_TOPK];
    unsigned char ringed[DSV4_MAX_TOPK];
    unsigned char joined[DSV4_MAX_TOPK];
    double started_at;
} DSV4ExpertBatch;

typedef struct {
    int experts[DSV4_MAX_TOPK];
    int slot_ids[DSV4_MAX_TOPK];
    int is_miss[DSV4_MAX_TOPK];
    DSV4ExpertBatch batch;
    int count;
    int active;
    int temporary;
} DSV4RoutePrefetch;

struct DSV4ExpertPool {
    pthread_mutex_t mutex;
    pthread_cond_t work_ready;
    pthread_cond_t state_changed;
    DSV4ExpertReadJob *queue[2 * DSV4_MAX_TOPK];
    pthread_t worker[DSV4_MAX_TOPK];
    int head, count, workers, stop;
};

static int expert_ring_init(DSV4Model *m)
{
    if (!getenv("DSV4_EXPERIMENTAL_IO_URING") ||
        getenv("K3_ST_BUFFERED") || getenv("DSV4_EXPERT_L1_PIPELINE"))
        return 0;
    m->expert_ring = io_ring_open(64);
    if (!m->expert_ring)
        fprintf(stderr, "dsv4: io_uring unavailable; using expert read workers\n");
    return m->expert_ring != NULL;
}

static void expert_ring_close(DSV4Model *m)
{
    io_ring_free(m->expert_ring);
    m->expert_ring = NULL;
}

static int expert_ring_prepare(DSV4ExpertReadJob *job, int phase,
                               int64_t off, int64_t nbytes, void *buf,
                               int64_t bufcap)
{
#ifdef __linux__
    DSV4Model *m = job->model;
    const int shard = job->meta->shard;
    if (!m->expert_ring || !m->st.dfd || shard < 0 || shard >= m->st.nshard ||
        m->st.dfd[shard] < 0) return 0;
    const int64_t lo = off & ~(int64_t)(K3_ST_ALIGN - 1);
    const int64_t hi = (off + nbytes + K3_ST_ALIGN - 1) &
                       ~(int64_t)(K3_ST_ALIGN - 1);
    const int64_t len = hi - lo;
    const int64_t pad = off - lo;
    if (len <= 0 || len > bufcap || len > UINT_MAX) return 0;
    DSV4RingRead *read = &job->ring_read[phase];
    read->owner = job;
    read->phase = phase;
    /* The final aligned window may end beyond EOF. Only the bytes through the
     * requested payload must be present, matching k3_st_read_aligned(). */
    read->expected = (int)(pad + nbytes);
    job->ring_payload_off[phase] = pad;
    return io_ring_submit(m->expert_ring, read, m->st.dfd[shard], buf,
                          (unsigned)len, lo);
#else
    (void)job; (void)phase; (void)off; (void)nbytes; (void)buf; (void)bufcap;
    return 0;
#endif
}

static int expert_ring_submit_job(DSV4ExpertReadJob *job)
{
    DSV4ExpertMeta *em = job->meta;
    DSV4ExpertSlot *slot = job->slot;
    job->ring_pending = 0;
    if (!expert_ring_prepare(job, 0, em->scale_off, em->scale_run,
                             slot->scales_base,
                             EXPERT_SCALE_RUN + 2 * K3_ST_ALIGN))
        return 0;
    job->ring_pending = 1;
    if (!expert_ring_prepare(job, 1, em->weight_off, em->weight_run,
                             slot->weights_base,
                             EXPERT_WEIGHT_RUN + 2 * K3_ST_ALIGN)) {
        job->ring_failed = 1;
        return 1;
    }
    job->ring_pending = 2;
    return 1;
}

static int expert_ring_reap_one(DSV4Model *m)
{
#ifdef __linux__
    DSV4RingRead *read = NULL;
    int result = 0;
    if (!io_ring_wait(m->expert_ring, &read, &result) || !read || !read->owner)
        return 0;
    DSV4ExpertReadJob *job = (DSV4ExpertReadJob *)read->owner;
    if (result < read->expected) job->ring_failed = 1;
    if (read->phase == 0)
        job->slot->scales = job->slot->scales_base + job->ring_payload_off[0];
    else
        job->slot->weights = job->slot->weights_base + job->ring_payload_off[1];
    job->ring_pending--;
    if (job->ring_pending == 0) {
        job->ok = !job->ring_failed;
        job->done = 1;
        job->done_at = now_s();
    }
    return 1;
#else
    (void)m;
    return 0;
#endif
}

static void expert_read_publish_l1(DSV4ExpertReadJob *job)
{
    if (!job->pool) {
        job->l1_done = 1;
        return;
    }
    pthread_mutex_lock(&job->pool->mutex);
    job->l1_done = 1;
    pthread_cond_broadcast(&job->pool->state_changed);
    pthread_mutex_unlock(&job->pool->mutex);
}

static void expert_read_job_run(DSV4ExpertReadJob *job)
{
    DSV4Model *m = job->model;
    DSV4ExpertMeta *em = job->meta;
    DSV4ExpertSlot *s = job->slot;
    int64_t po;
    if (getenv("DSV4_EXPERT_L1_PIPELINE")) {
        const int64_t projection_scale = EXPERT_SCALE_RUN / 3;
        const int64_t projection_weight = EXPERT_WEIGHT_RUN / 3;
        int64_t got = k3_st_read_aligned(
            &m->st, em->shard, em->scale_off, em->scale_run,
            s->scales_base, EXPERT_SCALE_RUN + 2 * 4096, &po);
        if (got != em->scale_run) {
            job->l1_ok = 0;
            expert_read_publish_l1(job);
            job->ok = 0;
            job->done_at = now_s();
            return;
        }
        s->scales = s->scales_base + po;

        int64_t weight_po = -1;
        got = k3_st_read_aligned(
            &m->st, em->shard, em->weight_off, projection_weight,
            s->weights_base, EXPERT_WEIGHT_RUN + 2 * 4096, &weight_po);
        if (got != projection_weight) {
            job->l1_ok = 0;
            expert_read_publish_l1(job);
            job->ok = 0;
            job->done_at = now_s();
            return;
        }
        s->weights = s->weights_base + weight_po;
        got = k3_st_read_aligned(
            &m->st, em->shard, em->weight_off + 2 * projection_weight,
            projection_weight, s->weights_base + 2 * projection_weight,
            EXPERT_WEIGHT_RUN + 2 * 4096 - 2 * projection_weight, &po);
        if (got != projection_weight || po != weight_po) {
            job->l1_ok = 0;
            expert_read_publish_l1(job);
            job->ok = 0;
            job->done_at = now_s();
            return;
        }
        (void)projection_scale;
        job->l1_ok = 1;
        expert_read_publish_l1(job);

        got = k3_st_read_aligned(
            &m->st, em->shard, em->weight_off + projection_weight,
            projection_weight, s->weights_base + projection_weight,
            EXPERT_WEIGHT_RUN + 2 * 4096 - projection_weight, &po);
        if (got != projection_weight || po != weight_po) {
            job->ok = 0;
            job->done_at = now_s();
            return;
        }
        job->ok = 1;
        job->done_at = now_s();
        return;
    }
    if (getenv("DSV4_EXPERT_WEIGHT_FIRST")) {
        int64_t got = k3_st_read_aligned(&m->st, em->shard, em->weight_off,
                                         em->weight_run, s->weights_base,
                                         EXPERT_WEIGHT_RUN + 2 * 4096, &po);
        if (got != em->weight_run) {
            job->ok = 0;
            job->done_at = now_s();
            return;
        }
        s->weights = s->weights_base + po;
        got = k3_st_read_aligned(&m->st, em->shard, em->scale_off,
                                 em->scale_run, s->scales_base,
                                 EXPERT_SCALE_RUN + 2 * 4096, &po);
        if (got != em->scale_run) {
            job->ok = 0;
            job->done_at = now_s();
            return;
        }
        s->scales = s->scales_base + po;
        job->ok = 1;
        job->done_at = now_s();
        return;
    }
    int64_t got = k3_st_read_aligned(&m->st, em->shard, em->scale_off,
                                     em->scale_run, s->scales_base,
                                     EXPERT_SCALE_RUN + 2 * 4096, &po);
    if (got != em->scale_run) {
        job->ok = 0;
        job->done_at = now_s();
        return;
    }
    s->scales = s->scales_base + po;
    got = k3_st_read_aligned(&m->st, em->shard, em->weight_off,
                             em->weight_run, s->weights_base,
                             EXPERT_WEIGHT_RUN + 2 * 4096, &po);
    if (got != em->weight_run) {
        job->ok = 0;
        job->done_at = now_s();
        return;
    }
    s->weights = s->weights_base + po;
    job->ok = 1;
    job->done_at = now_s();
}

static int expert_read_scale_phase(DSV4ExpertReadJob *job)
{
    DSV4Model *m = job->model;
    DSV4ExpertMeta *em = job->meta;
    DSV4ExpertSlot *s = job->slot;
    int64_t po;
    int64_t got = k3_st_read_aligned(
        &m->st, em->shard, em->scale_off, em->scale_run,
        s->scales_base, EXPERT_SCALE_RUN + 2 * 4096, &po);
    if (got != em->scale_run) return 0;
    s->scales = s->scales_base + po;
    return 1;
}

static int expert_read_weight_phase(DSV4ExpertReadJob *job)
{
    DSV4Model *m = job->model;
    DSV4ExpertMeta *em = job->meta;
    DSV4ExpertSlot *s = job->slot;
    int64_t po;
    int64_t got = k3_st_read_aligned(
        &m->st, em->shard, em->weight_off, em->weight_run,
        s->weights_base, EXPERT_WEIGHT_RUN + 2 * 4096, &po);
    if (got != em->weight_run) return 0;
    s->weights = s->weights_base + po;
    return 1;
}

static void *expert_read_thread(void *arg)
{
    expert_read_job_run((DSV4ExpertReadJob *)arg);
    return NULL;
}

static void *expert_pool_worker(void *arg)
{
    DSV4ExpertPool *pool = (DSV4ExpertPool *)arg;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->stop && pool->count == 0)
            pthread_cond_wait(&pool->work_ready, &pool->mutex);
        if (pool->stop && pool->count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        DSV4ExpertReadJob *job = pool->queue[pool->head];
        pool->head = (pool->head + 1) % (2 * DSV4_MAX_TOPK);
        pool->count--;
        pthread_cond_broadcast(&pool->state_changed);
        pthread_mutex_unlock(&pool->mutex);

        int requeue = 0;
        if (job->phased && job->phase == 0) {
            if (expert_read_scale_phase(job)) {
                job->phase = 1;
                requeue = 1;
            } else {
                job->ok = 0;
                job->done_at = now_s();
            }
        } else if (job->phased) {
            job->ok = expert_read_weight_phase(job);
            job->done_at = now_s();
        } else {
            expert_read_job_run(job);
        }
        pthread_mutex_lock(&pool->mutex);
        if (requeue) {
            int tail = (pool->head + pool->count) % (2 * DSV4_MAX_TOPK);
            pool->queue[tail] = job;
            pool->count++;
            pthread_cond_signal(&pool->work_ready);
        } else {
            job->done = 1;
        }
        pthread_cond_broadcast(&pool->state_changed);
        pthread_mutex_unlock(&pool->mutex);
    }
}

static int expert_pool_init(DSV4Model *m)
{
    if (m->expert_ring) return 0;
    if (expert_mmap_enabled(m)) return 0;
    if (getenv("DSV4_NO_EXPERT_POOL")) return 0;
    DSV4ExpertPool *pool = (DSV4ExpertPool *)calloc(1, sizeof(*pool));
    if (!pool) return 0;
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool);
        return 0;
    }
    if (pthread_cond_init(&pool->work_ready, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return 0;
    }
    if (pthread_cond_init(&pool->state_changed, NULL) != 0) {
        pthread_cond_destroy(&pool->work_ready);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return 0;
    }

    int wanted = m->cfg.topk;
    /* Four readers won the controlled laptop sweep. More readers reduced raw
     * I/O wait slightly but contended with the attention and MoE kernels. */
    if (wanted > 4) wanted = 4;
    if (wanted > m->threads) wanted = m->threads;
    if (wanted > DSV4_MAX_TOPK) wanted = DSV4_MAX_TOPK;
    if (wanted < 1) wanted = 1;
    const char *workers_env = getenv("DSV4_EXPERT_IO_WORKERS");
    if (workers_env) {
        char *end = NULL;
        long requested = strtol(workers_env, &end, 10);
        if (end != workers_env && *end == '\0' && requested >= 1 &&
            requested <= m->cfg.topk && requested <= DSV4_MAX_TOPK) {
            wanted = (int)requested;
        } else {
            fprintf(stderr, "dsv4: ignoring invalid DSV4_EXPERT_IO_WORKERS=%s "
                    "(expected 1..%d)\n", workers_env, m->cfg.topk);
        }
    }
    for (int i = 0; i < wanted; i++) {
        if (pthread_create(&pool->worker[i], NULL, expert_pool_worker, pool) != 0)
            break;
        pool->workers++;
    }
    if (pool->workers == 0) {
        pthread_cond_destroy(&pool->state_changed);
        pthread_cond_destroy(&pool->work_ready);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return 0;
    }
    m->expert_pool = pool;
    return 1;
}

static void expert_pool_close(DSV4Model *m)
{
    DSV4ExpertPool *pool = m->expert_pool;
    if (!pool) return;
    pthread_mutex_lock(&pool->mutex);
    pool->stop = 1;
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
    for (int i = 0; i < pool->workers; i++)
        (void)pthread_join(pool->worker[i], NULL);
    pthread_cond_destroy(&pool->state_changed);
    pthread_cond_destroy(&pool->work_ready);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
    m->expert_pool = NULL;
}

static void expert_pool_submit(DSV4ExpertPool *pool, DSV4ExpertReadJob *job)
{
    const int cap = 2 * DSV4_MAX_TOPK;
    pthread_mutex_lock(&pool->mutex);
    while (pool->count == cap)
        pthread_cond_wait(&pool->state_changed, &pool->mutex);
    int tail = (pool->head + pool->count) % cap;
    pool->queue[tail] = job;
    pool->count++;
    pthread_cond_signal(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
}

static void expert_pool_submit_many(DSV4ExpertPool *pool,
                                    DSV4ExpertReadJob **jobs, int count)
{
    const int cap = 2 * DSV4_MAX_TOPK;
    pthread_mutex_lock(&pool->mutex);
    while (pool->count + count > cap)
        pthread_cond_wait(&pool->state_changed, &pool->mutex);
    for (int i = 0; i < count; i++) {
        int tail = (pool->head + pool->count) % cap;
        pool->queue[tail] = jobs[i];
        pool->count++;
    }
    pthread_cond_broadcast(&pool->work_ready);
    pthread_mutex_unlock(&pool->mutex);
}

static void expert_pool_wait(DSV4ExpertReadJob *job)
{
    DSV4ExpertPool *pool = job->pool;
    pthread_mutex_lock(&pool->mutex);
    while (!job->done)
        pthread_cond_wait(&pool->state_changed, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

/* Reserve miss slots and start their O_DIRECT reads. Cache hits remain pinned,
 * so the caller may compute them while these jobs fill distinct victim slots.
 * Returns 1 when asynchronous reads were started, or 0 when all experts were
 * made resident by the established synchronous path. */
static int expert_batch_begin_pinned(DSV4Model *m, int layer,
                                     const int *experts, int n,
                                     int *slot_ids, int *is_miss,
                                     DSV4ExpertBatch *batch, int allow_overlap,
                                     const int *pinned, int npin)
{
    memset(batch, 0, sizeof(*batch));
    if (expert_mmap_enabled(m)) {
        expert_get_many(m, layer, experts, n, slot_ids, is_miss);
        return 0;
    }
    if (!allow_overlap || getenv("DSV4_NO_EXPERT_OVERLAP")) {
        expert_get_many(m, layer, experts, n, slot_ids, is_miss);
        return 0;
    }

    int miss_count = 0;
    for (int i = 0; i < n; i++) {
        int idx = cache_find(m, layer, experts[i]);
        /* A preceding double-buffer group may already be filling a victim
         * whose old cache tag is not published until that group finishes.
         * Such a slot is pinned for both reading and computation and cannot be
         * consumed as a hit by the next group. */
        if (idx >= 0) {
            for (int j = 0; j < npin; j++) {
                if (pinned[j] == idx) {
                    idx = -1;
                    break;
                }
            }
        }
        if (idx >= 0) {
            slot_ids[i] = idx;
            is_miss[i] = 0;
        } else {
            slot_ids[i] = -1;
            is_miss[i] = 1;
            miss_count++;
        }
    }
    if (miss_count == 0) {
        expert_get_many(m, layer, experts, n, slot_ids, is_miss);
        return 0;
    }

    for (int i = 0; i < n; i++) batch->victim[i] = -1;
    int begin, end;
    cache_bounds(m, layer, &begin, &end);
    for (int mi = 0; mi < n; mi++) {
        if (!is_miss[mi]) continue;
        cache_arc_prepare_miss(m, layer, experts[mi]);
        int v = -1;
        int64_t oldest = INT64_MAX;
        int recent_reserved = 0;
        for (int j = 0; j < n; j++)
            if (batch->victim[j] >= 0 &&
                m->cache[batch->victim[j]].segment == 1)
                recent_reserved++;
        for (int si = begin; si < end; si++) {
            int taken = 0;
            for (int j = 0; j < npin; j++) {
                if (pinned[j] == si) { taken = 1; break; }
            }
            for (int j = 0; j < n; j++) {
                if (!is_miss[j] && slot_ids[j] == si) { taken = 1; break; }
            }
            for (int j = 0; j < n; j++) {
                if (batch->victim[j] == si) { taken = 1; break; }
            }
            if (taken) continue;
            if (cache_victim_better(m, layer, experts[mi], recent_reserved,
                                    si, v, oldest)) {
                oldest = m->cache[si].last_use;
                v = si;
            }
        }
        if (v < 0) {
            expert_get_many(m, layer, experts, n, slot_ids, is_miss);
            return 0;
        }
        batch->victim[mi] = v;
    }

    batch->model = m;
    batch->experts = experts;
    batch->layer = layer;
    batch->n = n;
    batch->miss_count = miss_count;
    batch->slot_ids = slot_ids;
    batch->is_miss = is_miss;
    batch->started_at = now_s();

    for (int i = 0; i < n; i++) {
        if (!is_miss[i]) continue;
        batch->meta[i] = expert_meta(m, layer, experts[i]);
    }
    int read_order[DSV4_MAX_TOPK];
    int read_count = expert_read_order(batch->meta, is_miss, n, read_order);
    DSV4ExpertReadJob *phased_jobs[DSV4_MAX_TOPK];
    int phased_count = 0;
    const int phased = m->expert_pool &&
        getenv("DSV4_EXPERT_PHASED_READS") != NULL;
    for (int ri = 0; ri < read_count; ri++) {
        int i = read_order[ri];
        DSV4ExpertSlot *slot = &m->cache[batch->victim[i]];
        expert_slot_ensure_buffers(slot);
        batch->job[i].model = m;
        batch->job[i].meta = batch->meta[i];
        batch->job[i].slot = slot;
        batch->job[i].phased = phased;
        if (m->expert_ring && expert_ring_submit_job(&batch->job[i])) {
            batch->ringed[i] = 1;
        } else if (m->expert_pool) {
            batch->job[i].pool = m->expert_pool;
            batch->pooled[i] = 1;
            if (phased) phased_jobs[phased_count++] = &batch->job[i];
            else expert_pool_submit(m->expert_pool, &batch->job[i]);
        } else if (pthread_create(&batch->thread[i], NULL, expert_read_thread,
                                  &batch->job[i]) == 0) {
            batch->started[i] = 1;
        } else {
            expert_read_job_run(&batch->job[i]);
        }
    }
    if (phased_count > 0)
        expert_pool_submit_many(m->expert_pool, phased_jobs, phased_count);
    return 1;
}

static int expert_batch_begin(DSV4Model *m, int layer, const int *experts, int n,
                              int *slot_ids, int *is_miss,
                              DSV4ExpertBatch *batch, int allow_overlap)
{
    return expert_batch_begin_pinned(m, layer, experts, n, slot_ids, is_miss,
                                     batch, allow_overlap, NULL, 0);
}

static int expert_temp_batch_begin(DSV4Model *m, int layer,
                                   const int *experts, int n,
                                   int *slot_ids, int *is_miss,
                                   DSV4ExpertBatch *batch)
{
    memset(batch, 0, sizeof(*batch));
    batch->model = m;
    batch->experts = experts;
    batch->layer = layer;
    batch->n = n;
    batch->slot_ids = slot_ids;
    batch->is_miss = is_miss;
    batch->started_at = now_s();

    for (int i = 0; i < n; i++) {
        const int cached = cache_find(m, layer, experts[i]);
        if (cached >= 0) {
            slot_ids[i] = cached;
            is_miss[i] = 0;
            continue;
        }
        slot_ids[i] = -1;
        is_miss[i] = 1;
        batch->miss_count++;
        batch->meta[i] = expert_meta(m, layer, experts[i]);
    }
    if (batch->miss_count == 0) return 0;

    int read_order[DSV4_MAX_TOPK];
    int read_count = expert_read_order(batch->meta, is_miss, n, read_order);
    DSV4ExpertReadJob *phased_jobs[DSV4_MAX_TOPK];
    int phased_count = 0;
    const int phased = m->expert_pool &&
        getenv("DSV4_EXPERT_PHASED_READS") != NULL;
    for (int ri = 0; ri < read_count; ri++) {
        const int i = read_order[ri];
        DSV4ExpertSlot *slot = &m->route_prefetch_temp[i];
        expert_slot_ensure_buffers(slot);
        batch->job[i].model = m;
        batch->job[i].meta = batch->meta[i];
        batch->job[i].slot = slot;
        batch->job[i].phased = phased;
        if (m->expert_ring && expert_ring_submit_job(&batch->job[i])) {
            batch->ringed[i] = 1;
        } else if (m->expert_pool) {
            batch->job[i].pool = m->expert_pool;
            batch->pooled[i] = 1;
            if (phased) phased_jobs[phased_count++] = &batch->job[i];
            else expert_pool_submit(m->expert_pool, &batch->job[i]);
        } else if (pthread_create(&batch->thread[i], NULL, expert_read_thread,
                                  &batch->job[i]) == 0) {
            batch->started[i] = 1;
        } else {
            expert_read_job_run(&batch->job[i]);
        }
    }
    if (phased_count > 0)
        expert_pool_submit_many(m->expert_pool, phased_jobs, phased_count);
    return 1;
}

static DSV4ExpertSlot *expert_batch_wait_one(DSV4ExpertBatch *batch, int i)
{
    DSV4Model *m = batch->model;
    if (!batch->is_miss[i]) return &m->cache[batch->slot_ids[i]];

    double wait_start = m->profiling ? now_s() : 0.0;
    if (!batch->joined[i]) {
        if (batch->ringed[i]) {
            while (!batch->job[i].done) {
                if (!expert_ring_reap_one(m)) {
                    fprintf(stderr, "dsv4: io_uring completion failed\n");
                    exit(1);
                }
            }
        } else if (batch->pooled[i]) {
            expert_pool_wait(&batch->job[i]);
        } else if (batch->started[i] && pthread_join(batch->thread[i], NULL) != 0) {
            fprintf(stderr, "dsv4: failed to join expert read worker\n");
            exit(1);
        }
        batch->joined[i] = 1;
    }
    if (m->profiling) {
        double elapsed = now_s() - wait_start;
#ifdef _OPENMP
        #pragma omp atomic update
#endif
        m->time_expert_read += elapsed;
    }
    if (!batch->job[i].ok) {
        fprintf(stderr, "dsv4: short asynchronous read of expert %d.%d\n",
                batch->layer, batch->experts[i]);
        exit(1);
    }
    return batch->job[i].slot;
}

static DSV4ExpertSlot *expert_batch_wait_l1(DSV4ExpertBatch *batch, int i)
{
    DSV4Model *m = batch->model;
    DSV4ExpertReadJob *job = &batch->job[i];
    if (!batch->is_miss[i]) return &m->cache[batch->slot_ids[i]];

    double wait_start = m->profiling ? now_s() : 0.0;
    DSV4ExpertPool *pool = m->expert_pool;
    pthread_mutex_lock(&pool->mutex);
    while (!job->l1_done && !job->done)
        pthread_cond_wait(&pool->state_changed, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
    if (m->profiling) m->time_expert_read += now_s() - wait_start;
    if (!job->l1_done || !job->l1_ok) {
        fprintf(stderr, "dsv4: short asynchronous L1 read of expert %d.%d\n",
                batch->layer, batch->experts[i]);
        exit(1);
    }
    return job->slot;
}

static int expert_batch_next_l1_ready(DSV4ExpertBatch *batch,
                                      const unsigned char *computed)
{
    DSV4Model *m = batch->model;
    DSV4ExpertPool *pool = m->expert_pool;
    double wait_start = m->profiling ? now_s() : 0.0;
    pthread_mutex_lock(&pool->mutex);
    int ready = -1;
    while (ready < 0) {
        for (int i = 0; i < batch->n; i++) {
            if (batch->is_miss[i] && !computed[i] &&
                (batch->job[i].l1_done || batch->job[i].done)) {
                ready = i;
                break;
            }
        }
        if (ready < 0) pthread_cond_wait(&pool->state_changed, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
    if (m->profiling) m->time_expert_read += now_s() - wait_start;
    return ready;
}

static int expert_batch_next_ready(DSV4ExpertBatch *batch,
                                   const unsigned char *computed)
{
    DSV4Model *m = batch->model;
    DSV4ExpertPool *pool = m->expert_pool;
    int has_ring = 0;
    for (int i = 0; i < batch->n; i++) {
        if (!batch->is_miss[i] || computed[i] || !batch->ringed[i]) continue;
        has_ring = 1;
        if (batch->job[i].done) return i;
    }
    if (has_ring) {
        double wait_start = m->profiling ? now_s() : 0.0;
        for (;;) {
            if (!expert_ring_reap_one(m)) return -1;
            for (int i = 0; i < batch->n; i++) {
                if (batch->is_miss[i] && !computed[i] && batch->ringed[i] &&
                    batch->job[i].done) {
                    if (m->profiling)
                        m->time_expert_read += now_s() - wait_start;
                    return i;
                }
            }
        }
    }
    for (int i = 0; i < batch->n; i++) {
        if (batch->is_miss[i] && !computed[i] && !batch->pooled[i] &&
            !batch->ringed[i]) return i;
    }
    if (!pool) return -1;

    double wait_start = m->profiling ? now_s() : 0.0;
    pthread_mutex_lock(&pool->mutex);
    int ready = -1;
    while (ready < 0) {
        for (int i = 0; i < batch->n; i++) {
            if (batch->is_miss[i] && !computed[i] && batch->job[i].done) {
                ready = i;
                break;
            }
        }
        if (ready < 0) pthread_cond_wait(&pool->state_changed, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
    if (m->profiling) m->time_expert_read += now_s() - wait_start;
    return ready;
}

static void expert_batch_finish(DSV4ExpertBatch *batch)
{
    DSV4Model *m = batch->model;
    double last_done = batch->started_at;
    for (int i = 0; i < batch->n; i++) {
        if (!batch->is_miss[i]) continue;
        (void)expert_batch_wait_one(batch, i);
        if (batch->job[i].done_at > last_done) last_done = batch->job[i].done_at;
    }
    if (m->profiling)
        m->time_expert_io += last_done - batch->started_at;

    for (int i = 0; i < batch->n; i++) {
        if (!batch->is_miss[i]) continue;
        DSV4ExpertMeta *em = batch->meta[i];
        cache_publish(m, batch->layer, batch->experts[i], batch->victim[i]);
        batch->slot_ids[i] = batch->victim[i];
        m->cache_misses++;
        m->coalesced_loads += 1;
        m->expert_read_ops += getenv("DSV4_EXPERT_L1_PIPELINE") ? 4 : 2;
        m->expert_bytes_read += em->scale_run + em->weight_run;
    }
    for (int i = 0; i < batch->n; i++) {
        if (batch->is_miss[i]) continue;
        cache_touch(m, batch->layer, batch->slot_ids[i]);
        m->cache_hits++;
    }
    m->batch_prefetched += batch->miss_count;
}

static void expert_temp_batch_finish(DSV4ExpertBatch *batch,
                                     const int *actual_experts, int n_actual)
{
    DSV4Model *m = batch->model;
    double last_done = batch->started_at;
    for (int i = 0; i < batch->n; i++) {
        if (!batch->is_miss[i]) continue;
        (void)expert_batch_wait_one(batch, i);
        if (batch->job[i].done_at > last_done)
            last_done = batch->job[i].done_at;
    }
    if (m->profiling)
        m->time_expert_io += last_done - batch->started_at;

    for (int i = 0; i < batch->n; i++) {
        if (!batch->is_miss[i]) continue;
        DSV4ExpertMeta *em = batch->meta[i];
        m->route_prefetch_reads++;
        m->cache_misses++;
        m->coalesced_loads++;
        m->expert_read_ops += getenv("DSV4_EXPERT_L1_PIPELINE") ? 4 : 2;
        m->expert_bytes_read += em->scale_run + em->weight_run;
    }
    m->batch_prefetched += batch->miss_count;

    int pinned[256];
    int npin = 0;
    for (int u = 0; u < n_actual; u++) {
        const int slot = cache_find(m, batch->layer, actual_experts[u]);
        if (slot >= 0) pinned[npin++] = slot;
    }
    int adopted[DSV4_MAX_TOPK];
    int nadopted = 0;
    int begin, end;
    cache_bounds(m, batch->layer, &begin, &end);
    for (int i = 0; i < batch->n; i++) {
        if (!batch->is_miss[i]) continue;
        int needed = 0;
        for (int u = 0; u < n_actual; u++)
            if (actual_experts[u] == batch->experts[i]) {
                needed = 1;
                break;
            }
        if (!needed) {
            m->route_prefetch_wasted++;
            continue;
        }
        if (cache_find(m, batch->layer, batch->experts[i]) >= 0) {
            m->route_prefetch_adopted++;
            continue;
        }

        int victim = -1;
        int64_t oldest = INT64_MAX;
        cache_arc_prepare_miss(m, batch->layer, batch->experts[i]);
        int recent_reserved = 0;
        for (int p = 0; p < nadopted; p++)
            if (m->cache[adopted[p]].segment == 1) recent_reserved++;
        for (int slot = begin; slot < end; slot++) {
            int taken = 0;
            for (int p = 0; p < npin; p++)
                if (pinned[p] == slot) { taken = 1; break; }
            for (int p = 0; p < nadopted; p++)
                if (adopted[p] == slot) { taken = 1; break; }
            if (taken) continue;
            if (cache_victim_better(m, batch->layer, batch->experts[i],
                                    recent_reserved, slot, victim, oldest)) {
                victim = slot;
                oldest = m->cache[slot].last_use;
            }
        }
        if (victim < 0) {
            m->route_prefetch_unplaced++;
            continue;
        }

        DSV4ExpertSlot *temp = batch->job[i].slot;
        DSV4ExpertSlot *resident = &m->cache[victim];
        uint8_t *base = resident->scales_base;
        uint8_t *data = resident->scales;
        resident->scales_base = temp->scales_base;
        resident->scales = temp->scales;
        temp->scales_base = base;
        temp->scales = data;
        base = resident->weights_base;
        data = resident->weights;
        resident->weights_base = temp->weights_base;
        resident->weights = temp->weights;
        temp->weights_base = base;
        temp->weights = data;
        cache_publish(m, batch->layer, batch->experts[i], victim);
        adopted[nadopted++] = victim;
        m->route_prefetch_adopted++;
    }
}

/* ============================================================ helpers === */

/* One layer weight bundle. Loaded per forward step from disk, then released. */
static void load_layer_weights(DSV4Model *m, int layer, DSV4LayerW **out)
{
    *out = load_layer(m, layer);
}

static void release_layer_weights(DSV4Model *m, DSV4LayerW *w)
{
    free_layer(m, w);
}

typedef struct {
    DSV4Model *model;
    int layer;
    DSV4LayerW *weights;
    pthread_t thread;
    double started_at, done_at;
} DSV4LayerLoadJob;

static DSV4LayerW *load_layer_profiled(DSV4Model *m, int layer)
{
    double start = m->profiling ? now_s() : 0.0;
    DSV4LayerW *w = NULL;
    load_layer_weights(m, layer, &w);
    if (m->profiling) {
        double elapsed = now_s() - start;
        m->time_load += elapsed;
        m->time_layer_io += elapsed;
    }
    return w;
}

static void *layer_load_thread(void *arg)
{
    DSV4LayerLoadJob *job = (DSV4LayerLoadJob *)arg;
    job->weights = load_layer(job->model, job->layer);
    job->done_at = now_s();
    return NULL;
}

/* Loading a layer whose decoded wo_a is not resident can open an OpenMP region.
 * Keep that work on the main thread; background loads are pure allocation and I/O. */
static int layer_load_begin(DSV4Model *m, int layer, DSV4LayerLoadJob *job)
{
    if (m->layer_shard_map || getenv("DSV4_NO_LAYER_OVERLAP") || !m->wo_a_cache ||
        layer < 0 || layer >= m->wo_a_cache_layers || !m->wo_a_cache[layer])
        return 0;
    memset(job, 0, sizeof(*job));
    job->model = m;
    job->layer = layer;
    job->started_at = now_s();
    return pthread_create(&job->thread, NULL, layer_load_thread, job) == 0;
}

static DSV4LayerW *layer_load_finish(DSV4LayerLoadJob *job)
{
    DSV4Model *m = job->model;
    double wait_start = m->profiling ? now_s() : 0.0;
    if (pthread_join(job->thread, NULL) != 0) {
        fprintf(stderr, "dsv4: failed to join layer load worker\n");
        exit(1);
    }
    if (m->profiling) {
        m->time_load += now_s() - wait_start;
        m->time_layer_io += job->done_at - job->started_at;
    }
    return job->weights;
}

/* BF16 round of a float vector (in place). */
static void round_bf16_vec(float *x, int n)
{
    for (int i = 0; i < n; i++) x[i] = f32bf(x[i]);
}

/* RMSNorm with an optional BF16 gain; output rounded to BF16 in place. */
static void rmsnorm_bf16(float *y, const float *x, const uint16_t *w, int n,
                         float eps, int round_out)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float r = 1.0f / sqrtf((float)(ss / n) + eps);
    if (w) {
        for (int i = 0; i < n; i++) {
            float v = x[i] * r * bf16f(w[i]);
            y[i] = round_out ? f32bf(v) : v;
        }
    } else {
        for (int i = 0; i < n; i++) {
            float v = x[i] * r;
            y[i] = round_out ? f32bf(v) : v;
        }
    }
}

int dsv4_dspark_ready(const DSV4Model *m)
{
    return m && m->dspark != NULL;
}

int dsv4_dspark_commit_target_hidden(DSV4Model *m, const float *capture_hidden,
                                     int n_tokens, int start_position)
{
    if (!m || !m->dspark || !capture_hidden || n_tokens <= 0 ||
        start_position < 0 || start_position > m->context - n_tokens)
        return -1;
    const DSV4Config *c = &m->cfg;
    DSV4DSpark *d = m->dspark;
    const int hid = c->hidden;
    const int stages = c->dspark_stages;
    const int input = hid * stages;
    const int head_dim = c->head_dim;
    const int rope_dim = c->rope_dim;
    const int in_scales = (input + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
    const int hid_scales = (hid + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;

    float *input_q = (float *)malloc((size_t)n_tokens * input * sizeof(float));
    float *input_s = (float *)malloc((size_t)n_tokens * in_scales * sizeof(float));
    float *projected = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *projected_s = (float *)malloc((size_t)n_tokens * hid_scales * sizeof(float));
    float *kv = (float *)malloc((size_t)head_dim * sizeof(float));
    float *kvn = (float *)malloc((size_t)head_dim * sizeof(float));
    if (!input_q || !input_s || !projected || !projected_s || !kv || !kvn) {
        free(input_q); free(input_s); free(projected); free(projected_s);
        free(kv); free(kvn);
        return -1;
    }

    for (int t = 0; t < n_tokens; t++) {
        const float *src = capture_hidden + (size_t)t * stages * hid;
        float *qx = input_q + (size_t)t * input;
        float *qs = input_s + (size_t)t * in_scales;
        float *out = projected + (size_t)t * hid;
        dsv4_quant_fp8_codes(qx, qs, src, input);
        dsv4_gemv_fp8_q(out, qx, qs, d->main_proj, d->main_proj_scale,
                        input, hid, 1);
        rmsnorm_bf16(out, out, d->main_norm, hid, c->rms_eps, 1);
        dsv4_quant_fp8_codes(out, projected_s + (size_t)t * hid_scales,
                            out, hid);
    }

    for (int s = 0; s < stages; s++) {
        float *stage_cache = d->target_kv +
            (size_t)s * c->window * head_dim;
        for (int t = 0; t < n_tokens; t++) {
            const int pos = start_position + t;
            float *qx = projected + (size_t)t * hid;
            const float *qs = projected_s + (size_t)t * hid_scales;
            dsv4_gemv_fp8_q(kv, qx, qs, d->target_wkv[s],
                            d->target_wkv_scale[s], hid, head_dim, 1);
            rmsnorm_bf16(kvn, kv, d->target_kv_norm[s], head_dim,
                         c->rms_eps, 1);
            DSV4RopeFreq rope;
            dsv4_rope_freqs(&rope, rope_dim, c->rope_theta, 0,
                            c->rope_factor, c->beta_fast, c->beta_slow, pos,
                            c->original_position);
            dsv4_rope_apply_buf(kvn, 1, head_dim, &rope);
            free(rope.cosv);
            free(rope.sinv);
            dsv4_act_quant_inplace(kvn, head_dim - rope_dim,
                                   DSV4_ACT_GROUP, 0);
            memcpy(stage_cache + (size_t)(pos % c->window) * head_dim,
                   kvn, (size_t)head_dim * sizeof(float));
        }
    }
    for (int t = 0; t < n_tokens; t++) {
        const int pos = start_position + t;
        d->target_position[pos % c->window] = pos;
    }

    free(input_q); free(input_s); free(projected); free(projected_s);
    free(kv); free(kvn);
    return 0;
}

/* ========================================================= attention ==== */

static int indexer_forward(DSV4Model *m, const DSV4LayerW *w, DSV4LayerRun *r,
                           const float *x, const float *qr_qx,
                           const float *qr_qscale, int pos, int *comp_topk);
static void sparse_attn(DSV4Model *m, float *o, const float *q, const float *kv,
                        const float *attn_sink, const int *topk_idx,
                        int topk_total, int H, int d, float scale);

static void attention_scratch_reset(DSV4Model *m)
{
    m->attention_scratch_used = 0;
}

static void *attention_scratch_alloc(DSV4Model *m, size_t bytes)
{
    if (getenv("DSV4_NO_ATTN_SCRATCH")) return malloc(bytes);
    const size_t align = sizeof(double);
    size_t at = (m->attention_scratch_used + align - 1) & ~(align - 1);
    if (at <= m->attention_scratch_cap &&
        bytes <= m->attention_scratch_cap - at) {
        void *p = m->attention_scratch + at;
        m->attention_scratch_used = at + bytes;
        return p;
    }
    return malloc(bytes);
}

static void attention_scratch_free(DSV4Model *m, void *p)
{
    if (!p) return;
    uintptr_t address = (uintptr_t)p;
    uintptr_t begin = (uintptr_t)m->attention_scratch;
    uintptr_t end = begin + m->attention_scratch_cap;
    if (address < begin || address >= end) free(p);
}

/* Decode-phase window indices: the released get_window_topk_idxs for
 * seqlen=1. Ring order when the window has wrapped; ascending order before
 * that. Before the window fills, omit padding that sparse_attn would skip. */
static int window_topk(DSV4Model *m, int pos, int *win_idx, int keep_padding)
{
    const int win = m->cfg.window;
    if (pos >= win - 1) {
        int s = pos % win;
        for (int i = 0; i < win; i++) win_idx[i] = (s + 1 + i) % win;
        return win;
    }

    const int count = pos + 1;
    for (int i = 0; i < count; i++) win_idx[i] = i;
    if (keep_padding) {
        for (int i = count; i < win; i++) win_idx[i] = -1;
        return win;
    }
    return count;
}

/* One update step of a Compressor (decode, one token). x is the incoming
 * hidden state (float, the BF16-rounded bin). wkv/wgate are BF16 weight
 * matrices [coff*head_dim][hidden] but the projection is computed in fp32
 * (the released module stores its own parameters fp32); kv_state and
 * score_state are fp32. On a block boundary the compressed vector is written
 * to out_buf when non-NULL, else to kv_cache[win + comp_count] (fp32 storage
 * of BF16-rounded values). rotate selects the indexer path (Hadamard + FP4
 * act quant instead of FP8 act quant on the non-rope dims). Returns 1 when a
 * block boundary fired, 0 otherwise. */
static int compressor_update(DSV4Model *m, const DSV4LayerW *w, DSV4LayerRun *r,
                             const float *x, int pos, int head_dim,
                             const uint16_t *comp_wkv, const uint16_t *comp_wgate,
                             const float *ape, const uint16_t *comp_norm,
                             int ratio, int coff, int rotate, float *out_buf)
{
    const int d = head_dim;
    const int cd = coff * d;                 /* projection width                 */
    /* the indexer keeps its own state buffers so its memmove never clobbers the
     * main compressor's running state (the two share a layer) */
    float *kvs = rotate ? r->idx_kv_state : r->kv_state;
    float *scs = rotate ? r->idx_score_state : r->score_state;

    /* kv = comp_wkv(x)  [cd] (fp32); score = comp_wgate(x) [cd] (fp32) */
    float *kv = (float *)attention_scratch_alloc(
        m, (size_t)cd * sizeof(float));
    float *score = (float *)attention_scratch_alloc(
        m, (size_t)cd * sizeof(float));
    if (!kv || !score) { fprintf(stderr, "dsv4: OOM compressor\n"); exit(1); }
    dsv4_gemv_bf16(kv, x, comp_wkv, m->cfg.hidden, cd, 0);
    dsv4_gemv_bf16(score, x, comp_wgate, m->cfg.hidden, cd, 0);

    /* score += ape[pos % ratio] */
    const float *ape_row = ape + (size_t)(pos % ratio) * cd;
    for (int i = 0; i < cd; i++) score[i] += ape_row[i];

    /* write into state slots (fp32) */
    int slot = (pos % ratio) + (coff == 2 ? ratio : 0);
    memcpy(kvs + (size_t)slot * cd, kv, (size_t)cd * sizeof(float));
    memcpy(scs + (size_t)slot * cd, score, (size_t)cd * sizeof(float));

    const int should_compress = ((pos + 1) % ratio == 0);
    if (!should_compress) {
        attention_scratch_free(m, kv);
        attention_scratch_free(m, score);
        return 0;
    }

    /* gather the block rows: overlap -> [2*ratio rows, d]; else [ratio rows, d] */
    int rows = ratio * coff;
    float *bkv = (float *)attention_scratch_alloc(
        m, (size_t)rows * d * sizeof(float));
    float *bsc = (float *)attention_scratch_alloc(
        m, (size_t)rows * d * sizeof(float));
    if (!bkv || !bsc) { fprintf(stderr, "dsv4: OOM compressor block\n"); exit(1); }
    if (coff == 2) {
        /* Previous block contributes its first half; the current block, which
         * was written into slots ratio..2*ratio, contributes its second half. */
        for (int i = 0; i < ratio; i++) {
            memcpy(bkv + (size_t)i * d, kvs + (size_t)i * cd, (size_t)d * sizeof(float));
            memcpy(bsc + (size_t)i * d, scs + (size_t)i * cd, (size_t)d * sizeof(float));
            memcpy(bkv + (size_t)(ratio + i) * d,
                   kvs + (size_t)(ratio + i) * cd + d, (size_t)d * sizeof(float));
            memcpy(bsc + (size_t)(ratio + i) * d,
                   scs + (size_t)(ratio + i) * cd + d, (size_t)d * sizeof(float));
        }
        /* shift: kv_state[:ratio] = kv_state[ratio:] */
        memmove(kvs, kvs + (size_t)ratio * cd, (size_t)ratio * cd * sizeof(float));
        memmove(scs, scs + (size_t)ratio * cd, (size_t)ratio * cd * sizeof(float));
    } else {
        for (int i = 0; i < ratio; i++) {
            memcpy(bkv + (size_t)i * d, kvs + (size_t)i * cd, (size_t)d * sizeof(float));
            memcpy(bsc + (size_t)i * d, scs + (size_t)i * cd, (size_t)d * sizeof(float));
        }
    }

    /* softmax over rows (per channel d) */
    for (int ch = 0; ch < d; ch++) {
        float mx = bsc[ch];
        for (int i = 1; i < rows; i++) if (bsc[(size_t)i * d + ch] > mx) mx = bsc[(size_t)i * d + ch];
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) sum += expf(bsc[(size_t)i * d + ch] - mx);
        float inv = 1.0f / sum;
        for (int i = 0; i < rows; i++) bsc[(size_t)i * d + ch] = expf(bsc[(size_t)i * d + ch] - mx) * inv;
    }
    /* weighted sum -> [d] (fp32) */
    float *comp = (float *)attention_scratch_alloc(
        m, (size_t)d * sizeof(float));
    if (!comp) { fprintf(stderr, "dsv4: OOM compressor out\n"); exit(1); }
    for (int ch = 0; ch < d; ch++) {
        float acc = 0.0f;
        for (int i = 0; i < rows; i++)
            acc = fmaf(bsc[(size_t)i * d + ch], bkv[(size_t)i * d + ch], acc);
        comp[ch] = acc;
    }

    /* kv = norm(kv.to(bf16)): round to bf16, rmsnorm with gain, round to bf16 */
    round_bf16_vec(comp, d);
    float *compb = (float *)attention_scratch_alloc(
        m, (size_t)d * sizeof(float));
    if (!compb) { fprintf(stderr, "dsv4: OOM compressor norm\n"); exit(1); }
    rmsnorm_bf16(compb, comp, comp_norm, d, m->cfg.rms_eps, 1);

    /* RoPE on the last 64 dims at the block-start position */
    const int blk_start = pos + 1 - ratio;
    float *ropebuf = (float *)attention_scratch_alloc(
        m, (size_t)d * sizeof(float));
    if (!ropebuf) { fprintf(stderr, "dsv4: OOM compressor rope\n"); exit(1); }
    memcpy(ropebuf, compb, (size_t)d * sizeof(float));
    if (d >= m->cfg.rope_dim) {
        DSV4RopeFreq owned;
        const DSV4RopeFreq *f = model_rope_freqs(m, 2, blk_start, &owned);
        /* rotate the last 64 dims: treat as 1 head of width 64 */
        float *tail = ropebuf + (d - m->cfg.rope_dim);
        dsv4_rope_apply_buf(tail, 1, m->cfg.rope_dim, f);
        model_rope_freqs_release(f, &owned);
    }

    /* act quant (or rotate+fp4 for the indexer) */
    if (rotate) {
        dsv4_hadamard_inplace(ropebuf, d);
        dsv4_fp4_act_quant_inplace(ropebuf, d, 0);   /* BF16 */
    } else {
        dsv4_act_quant_inplace(ropebuf, d - m->cfg.rope_dim, DSV4_ACT_GROUP, 0);   /* BF16 */
    }

    /* store into out_buf or kv_cache[win + comp_count] (fp32 of bf16 values) */
    if (out_buf) {
        memcpy(out_buf, ropebuf, (size_t)d * sizeof(float));
    } else {
        if (r->comp_count + m->cfg.window + 1 >= r->kv_cap) {
            fprintf(stderr, "dsv4: compressor overflow at pos %d\n", pos);
            exit(1);
        }
        float *dst = r->kv_cache + (size_t)(m->cfg.window + r->comp_count) * d;
        memcpy(dst, ropebuf, (size_t)d * sizeof(float));
        r->comp_count++;
    }
    attention_scratch_free(m, kv);
    attention_scratch_free(m, score);
    attention_scratch_free(m, bkv);
    attention_scratch_free(m, bsc);
    attention_scratch_free(m, comp);
    attention_scratch_free(m, compb);
    attention_scratch_free(m, ropebuf);
    return 1;
}

/* Attention: q/kv projections, KV write, compressor/indexer updates, sparse
 * attention, output projection. x is the bf16-rounded, attn_norm'ed input
 * (float array holding bf16 values). Returns the attention output (bf16). */
static void attention_forward_impl(DSV4Model *m, const DSV4LayerW *w,
                                   DSV4LayerRun *r, const float *x, int pos,
                                   float *out, float *qr, float *qr_n,
                                   float *q, float *kv, float *kvn,
                                   float *raw_attention)
{
    attention_scratch_reset(m);
    const int hid = m->cfg.hidden;
    const int H = m->cfg.n_heads;
    const int d = m->cfg.head_dim;
    const int rd = m->cfg.rope_dim;
    const int ql = m->cfg.q_lora;
    const int ratio = w->compress_ratio;
    const int coff = (ratio == 4) ? 2 : 1;
    const float scale = 1.0f / sqrtf((float)d);
    const int win = m->cfg.window;

    float *qscale = qr;  /* wq_a output is dead after RMSNorm. */
    DSV4RopeFreq owned_attn_rope;
    const DSV4RopeFreq *attn_rope =
        model_rope_freqs(m, ratio ? 1 : 0, pos, &owned_attn_rope);
    if (!q) {
        /* ---- q ---- */
        qr = (float *)attention_scratch_alloc(m, (size_t)ql * sizeof(float));
        qr_n = (float *)attention_scratch_alloc(
            m, (size_t)ql * sizeof(float));
        float *xqx = m->scratch;
        float *xqsc = xqx + hid;
        if (!qr || !qr_n) { fprintf(stderr, "dsv4: OOM q\n"); exit(1); }
        dsv4_quant_fp8_codes(xqx, xqsc, x, hid);
        dsv4_gemv_fp8_q(qr, xqx, xqsc, w->wq_a, w->wq_a_scale,
                        hid, ql, 1);
        rmsnorm_bf16(qr_n, qr, w->q_norm, ql, m->cfg.rms_eps, 1);

        q = (float *)attention_scratch_alloc(
            m, (size_t)H * d * sizeof(float));
        qscale = qr;
        if (!q) { fprintf(stderr, "dsv4: OOM q2\n"); exit(1); }
        dsv4_quant_fp8_codes(qr_n, qscale, qr_n, ql);
        dsv4_gemv_fp8_q(q, qr_n, qscale, w->wq_b, w->wq_b_scale,
                        ql, H * d, 1);
        for (int h = 0; h < H; h++) {
            float *qh = q + (size_t)h * d;
            double ss = 0.0;
            for (int i = 0; i < d; i++)
                ss += (double)qh[i] * (double)qh[i];
            float rr = 1.0f / sqrtf((float)(ss / d) + m->cfg.rms_eps);
            for (int i = 0; i < d; i++) qh[i] = f32bf(qh[i] * rr);
        }
        dsv4_rope_apply_buf(q, H, d, attn_rope);

        /* ---- kv ---- */
        kv = (float *)attention_scratch_alloc(m, (size_t)d * sizeof(float));
        kvn = (float *)attention_scratch_alloc(m, (size_t)d * sizeof(float));
        if (!kv || !kvn) { fprintf(stderr, "dsv4: OOM kv\n"); exit(1); }
        dsv4_gemv_fp8_q(kv, xqx, xqsc, w->wkv, w->wkv_scale,
                        hid, d, 1);
        rmsnorm_bf16(kvn, kv, w->kv_norm, d, m->cfg.rms_eps, 1);
        dsv4_rope_apply_buf(kvn, 1, d, attn_rope);
        dsv4_act_quant_inplace(kvn, d - rd, DSV4_ACT_GROUP, 0);
    }

    /* write kv into the window slot */
    memcpy(r->kv_cache + (size_t)(pos % win) * d, kvn, (size_t)d * sizeof(float));
    if (getenv("DSV4_DEBUG_ATTN")) {
        double qs = 0; for (int _i = 0; _i < H*d; _i++) qs += (double)q[_i]*(double)q[_i];
        fprintf(stderr, "ATNK pos=%d qL2=%.7f", pos, sqrt(qs));
        for (int _s = 0; _s <= pos; _s++) {
            double ks = 0; for (int _i = 0; _i < d; _i++) ks += (double)r->kv_cache[(size_t)_s*d+_i]*(double)r->kv_cache[(size_t)_s*d+_i];
            fprintf(stderr, " kv%d=%.7f", _s, sqrt(ks));
        }
        fprintf(stderr, "\n");
    }

    /* ---- compressor / indexer ---- */
    int *comp_topk = NULL;
    int ncomp = 0;
    if (ratio) {
        compressor_update(m, w, r, x, pos, d, w->comp_wkv, w->comp_wgate, w->ape,
                          w->comp_norm, ratio, coff, 0, NULL);
        if (ratio == 4) {
            comp_topk = (int *)attention_scratch_alloc(
                m, (size_t)m->cfg.index_topk * sizeof(int));
            if (!comp_topk) { fprintf(stderr, "dsv4: OOM index top-k\n"); exit(1); }
            ncomp = indexer_forward(m, w, r, x, qr_n, qscale, pos, comp_topk);
        } else {
            /* all compressed positions */
            int cnt = r->comp_count;
            comp_topk = (int *)attention_scratch_alloc(
                m, (size_t)(cnt > 0 ? cnt : 1) * sizeof(int));
            if (!comp_topk) { fprintf(stderr, "dsv4: OOM compressed indices\n"); exit(1); }
            for (int i = 0; i < cnt; i++) comp_topk[i] = win + i;
            ncomp = cnt;
        }
        {
            const char *limit_env = getenv("DSV4_DEBUG_COMP_MAX_INDEX");
            const char *pos_env = getenv("DSV4_DEBUG_COMP_POS");
            if (limit_env && (!pos_env || pos == (int)strtol(pos_env, NULL, 10))) {
                int limit = (int)strtol(limit_env, NULL, 10);
                int kept = 0;
                for (int i = 0; i < ncomp; i++)
                    if (comp_topk[i] < win + limit) comp_topk[kept++] = comp_topk[i];
                ncomp = kept;
            }
        }
    }

    /* ---- sparse attention ---- */
    const int keep_window_padding = getenv("DSV4_ATTN_PAD_WINDOW") != NULL;
    int *topk_idx = (int *)attention_scratch_alloc(
        m, (size_t)(win + ncomp) * sizeof(int));
    if (!topk_idx) { fprintf(stderr, "dsv4: OOM topk\n"); exit(1); }
    const int window_count = window_topk(m, pos, topk_idx,
                                         keep_window_padding);
    if (ncomp) {
        /* Compressed indices already carry +win into the combined KV cache. */
        memcpy(topk_idx + window_count, comp_topk,
               (size_t)ncomp * sizeof(int));
    }
    const int topk_total = window_count + ncomp;

    float *o = (float *)attention_scratch_alloc(
        m, (size_t)H * d * sizeof(float));
    if (!o) { fprintf(stderr, "dsv4: OOM attn out\n"); exit(1); }
    sparse_attn(m, o, q, r->kv_cache, w->attn_sink, topk_idx, topk_total,
                H, d, scale);
    if (getenv("DSV4_DEBUG_ATTN")) {
        double _s = 0; for (int _i = 0; _i < H*d; _i++) _s += (double)o[_i]*(double)o[_i];
        fprintf(stderr, "ATN pos=%d oL2=%.7f\n", pos, sqrt(_s));
    }

    /* inverse rope on the last rd dims */
    dsv4_rope_apply_buf_inv(o, H, d, attn_rope);

    /* ---- output projection ---- */
    /* o: [H][d] -> groups [G][gdim]; wo_a is BF16 (decoded at load) with no
     * activation quantisation, then wo_b is an FP8 GEMV */
    float *oall = NULL;
    if (raw_attention) {
        memcpy(raw_attention, o, (size_t)H * d * sizeof(float));
    } else {
        const int G = m->cfg.o_groups;
        const int ol = m->cfg.o_lora;
        const int gdim = H * d / G;
        oall = (float *)attention_scratch_alloc(
            m, (size_t)G * ol * sizeof(float));
        if (!oall) { fprintf(stderr, "dsv4: OOM wo_a\n"); exit(1); }
        if (w->wo_a_codes) {
            float *outputs[1] = { oall };
            const float *inputs[1] = { o };
            gemv_packed_wo_a_grouped_batch(
                outputs, inputs, 1, w->wo_a_codes, w->wo_a_scale,
                G, gdim, ol);
        } else if (!getenv("DSV4_NO_GROUPED_WOA")) {
            gemv_bf16_grouped(oall, o, w->wo_a, G, gdim, ol);
        } else {
            for (int g = 0; g < G; g++) {
                dsv4_gemv_bf16(oall + (size_t)g * ol,
                               o + (size_t)g * gdim,
                               w->wo_a + (size_t)g * ol * gdim,
                               gdim, ol, 1);
            }
        }
        if (getenv("DSV4_DEBUG_ATTN")) {
            double _s = 0;
            for (int _i = 0; _i < G * ol; _i++)
                _s += (double)oall[_i] * (double)oall[_i];
            fprintf(stderr, "ATN pos=%d oallL2=%.7f\n", pos, sqrt(_s));
        }
        float *obqsc = (float *)attention_scratch_alloc(
            m, (size_t)((G * ol + 127) / 128) * sizeof(float));
        if (!obqsc) { fprintf(stderr, "dsv4: OOM wo_b\n"); exit(1); }
        dsv4_quant_fp8_codes(oall, obqsc, oall, G * ol);
        dsv4_gemv_fp8_q(out, oall, obqsc, w->wo_b, w->wo_b_scale,
                        G * ol, hid, 1);
        if (getenv("DSV4_DEBUG_ATTN")) {
            double _s = 0;
            for (int _i = 0; _i < hid; _i++)
                _s += (double)out[_i] * (double)out[_i];
            fprintf(stderr, "ATN pos=%d ooutL2=%.7f\n", pos, sqrt(_s));
        }
        attention_scratch_free(m, obqsc);
    }

    model_rope_freqs_release(attn_rope, &owned_attn_rope);
    attention_scratch_free(m, qr);
    attention_scratch_free(m, qr_n);
    attention_scratch_free(m, q);
    attention_scratch_free(m, kv);
    attention_scratch_free(m, kvn);
    attention_scratch_free(m, comp_topk);
    attention_scratch_free(m, topk_idx);
    attention_scratch_free(m, o);
    attention_scratch_free(m, oall);
}

static void attention_forward(DSV4Model *m, const DSV4LayerW *w,
                              DSV4LayerRun *r, const float *x, int pos,
                              float *out)
{
    attention_forward_impl(m, w, r, x, pos, out, NULL, NULL, NULL, NULL,
                           NULL, NULL);
}

/* Precompute the position-independent Q/K/V projections for a short token
 * block, then execute KV, compressor, indexer and sparse-attention updates in
 * their original position order. Projection buffers are individually owned so
 * attention_forward_impl() can release them through the normal scratch API. */
static void attention_forward_batch_projected(DSV4Model *m,
                                               const DSV4LayerW *w,
                                               DSV4LayerRun *r,
                                               const float *x,
                                               int start_position,
                                               int n_tokens, float *out)
{
    const int hid = m->cfg.hidden;
    const int ql = m->cfg.q_lora;
    const int qwidth = m->cfg.n_heads * m->cfg.head_dim;
    const int d = m->cfg.head_dim;
    const int rd = m->cfg.rope_dim;
    const int xsc_n = (hid + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;
    const int groups = m->cfg.o_groups;
    const int ol = m->cfg.o_lora;
    const int oall_n = groups * ol;
    const int osc_n = (oall_n + DSV4_FP8_BLOCK - 1) / DSV4_FP8_BLOCK;

    for (int base = 0; base < n_tokens; base += DSV4_MAX_TOPK + 1) {
        int count = n_tokens - base;
        if (count > DSV4_MAX_TOPK + 1) count = DSV4_MAX_TOPK + 1;
        float *xq[DSV4_MAX_TOPK + 1], *xsc[DSV4_MAX_TOPK + 1];
        float *qr[DSV4_MAX_TOPK + 1], *qrn[DSV4_MAX_TOPK + 1];
        float *q[DSV4_MAX_TOPK + 1], *kv[DSV4_MAX_TOPK + 1];
        float *kvn[DSV4_MAX_TOPK + 1];
        float *raw[DSV4_MAX_TOPK + 1], *oall[DSV4_MAX_TOPK + 1];
        float *osc[DSV4_MAX_TOPK + 1], *out_ptr[DSV4_MAX_TOPK + 1];
        const float *xq_in[DSV4_MAX_TOPK + 1];
        const float *xsc_in[DSV4_MAX_TOPK + 1];
        const float *qrn_in[DSV4_MAX_TOPK + 1];
        const float *qrsc_in[DSV4_MAX_TOPK + 1];
        const float *raw_in[DSV4_MAX_TOPK + 1];
        const float *oall_in[DSV4_MAX_TOPK + 1];
        const float *osc_in[DSV4_MAX_TOPK + 1];

        for (int t = 0; t < count; t++) {
            xq[t] = (float *)malloc((size_t)hid * sizeof(float));
            xsc[t] = (float *)malloc((size_t)xsc_n * sizeof(float));
            qr[t] = (float *)malloc((size_t)ql * sizeof(float));
            qrn[t] = (float *)malloc((size_t)ql * sizeof(float));
            q[t] = (float *)malloc((size_t)qwidth * sizeof(float));
            kv[t] = (float *)malloc((size_t)d * sizeof(float));
            kvn[t] = (float *)malloc((size_t)d * sizeof(float));
            raw[t] = (float *)malloc((size_t)qwidth * sizeof(float));
            oall[t] = (float *)malloc((size_t)oall_n * sizeof(float));
            osc[t] = (float *)malloc((size_t)osc_n * sizeof(float));
            if (!xq[t] || !xsc[t] || !qr[t] || !qrn[t] || !q[t] ||
                !kv[t] || !kvn[t] || !raw[t] || !oall[t] || !osc[t]) {
                fprintf(stderr, "dsv4: OOM batched attention projection\n");
                exit(1);
            }
            dsv4_quant_fp8_codes(
                xq[t], xsc[t], x + (size_t)(base + t) * hid, hid);
            xq_in[t] = xq[t];
            xsc_in[t] = xsc[t];
        }
        dsv4_gemv_fp8_batch_q(qr, xq_in, xsc_in, count, w->wq_a,
                              w->wq_a_scale, hid, ql, 1);
        for (int t = 0; t < count; t++) {
            rmsnorm_bf16(qrn[t], qr[t], w->q_norm, ql,
                         m->cfg.rms_eps, 1);
            dsv4_quant_fp8_codes(qrn[t], qr[t], qrn[t], ql);
            qrn_in[t] = qrn[t];
            qrsc_in[t] = qr[t];
        }
        dsv4_gemv_fp8_batch_q(q, qrn_in, qrsc_in, count, w->wq_b,
                              w->wq_b_scale, ql, qwidth, 1);
        dsv4_gemv_fp8_batch_q(kv, xq_in, xsc_in, count, w->wkv,
                              w->wkv_scale, hid, d, 1);

        for (int t = 0; t < count; t++) {
            const int position = start_position + base + t;
            for (int h = 0; h < m->cfg.n_heads; h++) {
                float *qh = q[t] + (size_t)h * d;
                double ss = 0.0;
                for (int i = 0; i < d; i++)
                    ss += (double)qh[i] * (double)qh[i];
                float rr = 1.0f / sqrtf(
                    (float)(ss / d) + m->cfg.rms_eps);
                for (int i = 0; i < d; i++) qh[i] = f32bf(qh[i] * rr);
            }
            DSV4RopeFreq owned;
            const DSV4RopeFreq *rope = model_rope_freqs(
                m, w->compress_ratio ? 1 : 0, position, &owned);
            dsv4_rope_apply_buf(q[t], m->cfg.n_heads, d, rope);
            rmsnorm_bf16(kvn[t], kv[t], w->kv_norm, d,
                         m->cfg.rms_eps, 1);
            dsv4_rope_apply_buf(kvn[t], 1, d, rope);
            dsv4_act_quant_inplace(kvn[t], d - rd, DSV4_ACT_GROUP, 0);
            model_rope_freqs_release(rope, &owned);
        }
        for (int t = 0; t < count; t++) {
            free(xq[t]);
            free(xsc[t]);
            const int token = base + t;
            const int position = start_position + token;
            attention_forward_impl(
                m, w, r, x + (size_t)token * hid, position,
                NULL, qr[t], qrn[t], q[t], kv[t], kvn[t], raw[t]);
            context_snapshot_capture_layer(m, m->cur_layer, position);
            raw_in[t] = raw[t];
            out_ptr[t] = out + (size_t)token * hid;
        }
        if (w->wo_a_codes)
            gemv_packed_wo_a_grouped_batch(
                oall, raw_in, count, w->wo_a_codes, w->wo_a_scale,
                groups, qwidth / groups, ol);
        else
            gemv_bf16_grouped_batch(oall, raw_in, count, w->wo_a, groups,
                                    qwidth / groups, ol);
        for (int t = 0; t < count; t++) {
            dsv4_quant_fp8_codes(oall[t], osc[t], oall[t], oall_n);
            oall_in[t] = oall[t];
            osc_in[t] = osc[t];
        }
        dsv4_gemv_fp8_batch_q(out_ptr, oall_in, osc_in, count, w->wo_b,
                              w->wo_b_scale, oall_n, hid, 1);
        for (int t = 0; t < count; t++) {
            free(raw[t]);
            free(oall[t]);
            free(osc[t]);
        }
    }
}

/* ================================================== quant codes ======== */

/* E4M3 activation quant: qx[i] = dsv4_round_e4m3(x[i]/scale) as a float, and
 * scale[g] = 2^ceil(log2(amax/448)) with the 1e-4 amax floor. The returned
 * qx is on the E4M3 grid (so a later multiply by scale reproduces the
 * quantised value). */
void dsv4_quant_fp8_codes(float *qx, float *qscale, const float *x, int n)
{
    const int group = DSV4_FP8_BLOCK;
    for (int g = 0; g * group < n; g++) {
        int base = g * group, len = n - base;
        if (len > group) len = group;
        float amax = 0.0f;
        for (int i = 0; i < len; i++) {
            float a = fabsf(x[base + i]);
            if (a > amax) amax = a;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        float r = amax * (1.0f / 448.0f);
        int e;
        float m = frexpf(r, &e);  /* r = m * 2^e, m in [0.5,1) */
        if (m == 0.5f) e--;        /* ceil(log2(r)) = e - (m==0.5) */
        float scale = ldexpf(1.0f, e);
        qscale[g] = scale;
        float inv = 1.0f / scale;
        for (int i = 0; i < len; i++) {
            float q = x[base + i] * inv;
            if (q > 448.0f) q = 448.0f;
            else if (q < -448.0f) q = -448.0f;
            qx[base + i] = dsv4_round_e4m3(q);
        }
    }
}

/* ======================================================== indexer ====== */

typedef struct {
    float score;
    int index;
} DSV4ScoreIndex;

static int score_index_better(DSV4ScoreIndex a, DSV4ScoreIndex b)
{
    int a_nan = isnan(a.score), b_nan = isnan(b.score);
    if (a_nan != b_nan) return !a_nan;
    if (a.score > b.score) return 1;
    if (a.score < b.score) return 0;
    return a.index < b.index;
}

static int score_index_desc(const void *pa, const void *pb)
{
    DSV4ScoreIndex a = *(const DSV4ScoreIndex *)pa;
    DSV4ScoreIndex b = *(const DSV4ScoreIndex *)pb;
    if (score_index_better(a, b)) return -1;
    if (score_index_better(b, a)) return 1;
    return 0;
}

/* Exact top-k set with the same ordering as the former repeated maximum
 * scan: score descending, then lower source index first. The heap root is the
 * worst selected item, so the scan is O(n log k) instead of O(n*k). */
void dsv4_indexer_select_topk(const float *score, int n, int k, int offset,
                              int *out)
{
    if (k <= 0) return;
    DSV4ScoreIndex *heap = (DSV4ScoreIndex *)malloc((size_t)k * sizeof(*heap));
    if (!heap) { fprintf(stderr, "dsv4: OOM index top-k heap\n"); exit(1); }

    int count = 0;
    for (int i = 0; i < n; i++) {
        DSV4ScoreIndex item = { score[i], i };
        if (count < k) {
            int pos = count++;
            heap[pos] = item;
            while (pos > 0) {
                int parent = (pos - 1) / 2;
                if (!score_index_better(heap[parent], heap[pos])) break;
                DSV4ScoreIndex tmp = heap[parent];
                heap[parent] = heap[pos];
                heap[pos] = tmp;
                pos = parent;
            }
        } else if (score_index_better(item, heap[0])) {
            heap[0] = item;
            int pos = 0;
            for (;;) {
                int left = pos * 2 + 1;
                if (left >= count) break;
                int right = left + 1;
                int worst = left;
                if (right < count && score_index_better(heap[left], heap[right]))
                    worst = right;
                if (!score_index_better(heap[pos], heap[worst])) break;
                DSV4ScoreIndex tmp = heap[pos];
                heap[pos] = heap[worst];
                heap[worst] = tmp;
                pos = worst;
            }
        }
    }

    qsort(heap, (size_t)count, sizeof(*heap), score_index_desc);
    for (int i = 0; i < count; i++) out[i] = offset + heap[i].index;
    free(heap);
}

/* Indexer forward for one token. qr_qx/qr_qscale are the q_lora projection
 * already quantised for the main attention projection.
 * Writes up to index_topk compressed-position indices (already offset by win)
 * into comp_topk in score-descending order; returns the count. */
static int indexer_forward(DSV4Model *m, const DSV4LayerW *w, DSV4LayerRun *r,
                           const float *x, const float *qr_qx,
                           const float *qr_qscale, int pos, int *comp_topk)
{
    const int ih = m->cfg.index_heads;
    const int id = m->cfg.index_dim;
    const int rd = m->cfg.rope_dim;
    (void)rd;
    const int ql = m->cfg.q_lora;
    const int ratio = 4;
    const int coff = 2;
    const int win = m->cfg.window;
    const int maxc = (pos + 1) / ratio;

    /* q = idx_wq_b(qr) [ih*id] bf16 -> [ih][id]; rope; hadamard; fp4 quant */
    float *q = (float *)attention_scratch_alloc(
        m, (size_t)ih * id * sizeof(float));
    if (!q) { fprintf(stderr, "dsv4: OOM indexer q\n"); exit(1); }
    dsv4_gemv_fp8_q(q, qr_qx, qr_qscale, w->idx_wq_b, w->idx_wq_b_scale,
                    ql, ih * id, 1);
    {
        DSV4RopeFreq owned;
        const DSV4RopeFreq *f = model_rope_freqs(m, 1, pos, &owned);
        dsv4_rope_apply_buf(q, ih, id, f);
        model_rope_freqs_release(f, &owned);
    }
    for (int h = 0; h < ih; h++) dsv4_hadamard_inplace(q + (size_t)h * id, id);
    for (int h = 0; h < ih; h++) dsv4_fp4_act_quant_inplace(q + (size_t)h * id, id, 0);   /* BF16 */

    /* indexer compressor (head_dim = id, rotate = True): produce the current
     * compressed position into idx_cache. The compressor consumes the FULL
     * hidden state x (not the q_lora projection qr). */
    {
        int did = compressor_update(m, w, r, x, pos, id, w->idx_comp_wkv,
                                    w->idx_comp_wgate, w->idx_ape, w->idx_comp_norm,
                                    ratio, coff, 1,
                                    r->idx_cache + (size_t)r->idx_count * id);
        if (did) r->idx_count++;
    }

    /* weights = idx_weights_proj(x) * (id^-0.5 * ih^-0.5) [ih] bf16 */
    float *wts = (float *)attention_scratch_alloc(
        m, (size_t)ih * sizeof(float));
    if (!wts) { fprintf(stderr, "dsv4: OOM indexer wts\n"); exit(1); }
    dsv4_gemv_bf16(wts, x, w->idx_weights_proj, m->cfg.hidden, ih, 1);
    {
        float s = (1.0f / sqrtf((float)id)) * (1.0f / sqrtf((float)ih));
        for (int h = 0; h < ih; h++) wts[h] = f32bf(wts[h] * s);
    }

    /* index_score[t] = sum_h relu(sum_d q[h,d]*idx_cache[t,h,d]) * wts[h] */
    int nt = r->idx_count;
    if (nt > maxc) nt = maxc;
    float *score = (float *)attention_scratch_alloc(
        m, (size_t)(nt > 0 ? nt : 1) * sizeof(float));
    if (!score) { fprintf(stderr, "dsv4: OOM indexer score\n"); exit(1); }
    for (int t = 0; t < nt; t++) {
        const float *kp = r->idx_cache + (size_t)t * id;
        float acc = 0.0f;
        for (int h = 0; h < ih; h++) {
            float dot = 0.0f;
            const float *qh = q + (size_t)h * id;
            for (int dd = 0; dd < id; dd++) dot = fmaf(qh[dd], kp[dd], dot);
            if (dot < 0.0f) dot = 0.0f;
            acc = fmaf(dot, wts[h], acc);
        }
        score[t] = acc;
    }
    int topk = m->cfg.index_topk;
    if (topk > nt) topk = nt;
    dsv4_indexer_select_topk(score, nt, topk, win, comp_topk);
    attention_scratch_free(m, q);
    attention_scratch_free(m, wts);
    attention_scratch_free(m, score);
    return topk;
}

/* ==================================================== sparse_attn ====== */

/* q [H][d]; kv [n][d]; topk_idx [topk_total] with -1 padding; attn_sink [H].
 * Two passes: first iterate every key for scores and the per-head max, then sum
 * the denominator (plus the sink), then a final pass accumulates value. */
static void sparse_attn(DSV4Model *m, float *o, const float *q, const float *kv,
                        const float *attn_sink, const int *topk_idx,
                        int topk_total, int H, int d, float scale)
{
    const size_t stride = (size_t)(topk_total > 0 ? topk_total : 1);
    float *scores = (float *)attention_scratch_alloc(
        m, (size_t)H * stride * sizeof(float));
    float *wbuf = (float *)attention_scratch_alloc(
        m, (size_t)H * stride * sizeof(float));
    if (!scores || !wbuf) { fprintf(stderr, "dsv4: OOM attn scores\n"); exit(1); }
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < H; h++) {
        float *head_scores = scores + (size_t)h * stride;
        float *head_wbuf = wbuf + (size_t)h * stride;
        const float *qh = q + (size_t)h * d;
        float mx = -INFINITY;
        for (int t = 0; t < topk_total; t++) {
            int idx = topk_idx[t];
            float s;
            if (idx < 0) {
                s = -INFINITY;
            } else {
                const float *kp = kv + (size_t)idx * d;
                float dot = 0.0f;
                for (int i = 0; i < d; i++) dot = fmaf(qh[i], kp[i], dot);
                s = dot * scale;
            }
            head_scores[t] = s;
            if (s > mx) mx = s;
        }
        /* Denominator uses raw expf; the released reference adds the sink
         * before the scores. The value weight is the BF16-rounded expf. */
        float sum = expf(attn_sink[h] - mx);
        for (int t = 0; t < topk_total; t++) {
            if (head_scores[t] != -INFINITY) {
                float e = expf(head_scores[t] - mx);
                head_wbuf[t] = f32bf(e);       /* BF16 weight */
                sum += e;                      /* raw expf denominator */
            } else {
                head_wbuf[t] = 0.0f;
            }
        }
        float *oh = o + (size_t)h * d;
        for (int i = 0; i < d; i++) oh[i] = 0.0f;
        for (int t = 0; t < topk_total; t++) {
            int idx = topk_idx[t];
            if (idx < 0) continue;
            float a = head_wbuf[t];
            const float *kp = kv + (size_t)idx * d;
            for (int i = 0; i < d; i++) oh[i] = fmaf(a, kp[i], oh[i]);
        }
        for (int i = 0; i < d; i++) oh[i] = f32bf(oh[i] / sum);
    }
    attention_scratch_free(m, scores);
    attention_scratch_free(m, wbuf);
}

/* ============================================================ MoE ====== */

/* sqrt(softplus(z)), stable for large |z|. */
static float sqrt_softplus(float z)
{
    if (z > 20.0f) return sqrtf(z);
    return sqrtf(log1pf(expf(z)));
}

static int debug_moe_at(const DSV4Model *m, int position)
{
    const char *enabled = getenv("DSV4_DEBUG_MOE");
    const char *wanted = getenv("DSV4_DEBUG_MOE_POS");
    if (!enabled || m->cur_layer != 0) return 0;
    return !wanted || position == (int)strtol(wanted, NULL, 10);
}

static int debug_moe_detail(void)
{
    const char *value = getenv("DSV4_DEBUG_MOE");
    return value && strcmp(value, "detail") == 0;
}

static void debug_vec_summary(const char *name, const float *x, int n)
{
    double ss = 0.0;
    float lo = n ? x[0] : 0.0f;
    float hi = lo;
    uint64_t hash = UINT64_C(1469598103934665603);
    for (int i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, x + i, sizeof bits);
        ss += (double)x[i] * (double)x[i];
        if (x[i] < lo) lo = x[i];
        if (x[i] > hi) hi = x[i];
        for (int b = 0; b < 4; b++) {
            hash ^= (bits >> (8 * b)) & 0xffu;
            hash *= UINT64_C(1099511628211);
        }
    }
    fprintf(stderr, "    %-10s n=%d l2=%.10f min=%a max=%a hash=%016llx\n",
            name, n, sqrt(ss), (double)lo, (double)hi,
            (unsigned long long)hash);
}

/* One expert's SwiGLU block: gate/up FP4 GEMV from the FP8-quantised x, clamp,
 * silu, then w2 from the bf16-rounded activation. Returns the FP4 down result
 * as float (bf16 values) in out[hid]. */
typedef struct {
    float *gate, *up, *y, *qy, *qys;
} DSV4ExpertWork;

typedef struct {
    float *qx, *qsc;
    float *score, *choice, *wt, *acc, *routed, *shared;
    int *order, *idx;
} DSV4MoEScratch;

static void bind_expert_work(DSV4Model *m, DSV4ExpertWork *b, int slot)
{
    const int mi = m->cfg.moe_inter;
    const size_t stride = (size_t)4 * mi + (mi + 127) / 128;
    float *p = m->expert_work + (size_t)slot * stride;
    b->gate = p; p += mi;
    b->up = p; p += mi;
    b->y = p; p += mi;
    b->qy = p; p += mi;
    b->qys = p;
}

static void bind_moe_scratch(DSV4Model *m, DSV4MoEScratch *b)
{
    const int hid = m->cfg.hidden;
    float *p = m->scratch;
    b->qx = p; p += hid;
    b->qsc = p; p += (hid + 127) / 128;
    b->score = p; p += m->cfg.n_experts;
    b->choice = p; p += m->cfg.n_experts;
    b->wt = p; p += m->cfg.topk;
    b->acc = p; p += hid;
    b->routed = p; p += (size_t)m->cfg.topk * hid;
    b->shared = p;
    b->order = m->route_scratch;
    b->idx = m->route_scratch + m->cfg.n_experts;
}

static void route_prediction_prepare(DSV4Model *m, const DSV4LayerW *w,
                                     const float *attention_input, int token_id,
                                     DSV4RoutePrefetch *prefetch)
{
    m->route_prediction_valid = 0;
    const char *prefetch_value = getenv("DSV4_ROUTE_PREFETCH");
    const int debug_predict = getenv("DSV4_DEBUG_ROUTE_PREDICT") != NULL;
    if (!debug_predict && !prefetch_value) return;

    const int topk = m->cfg.topk;
    long learned_ranks = 0;
    if (prefetch_value) {
        if (!strcmp(prefetch_value, "learned")) {
            if (w->is_hash) return;
            learned_ranks = 1;
        } else if (!strcmp(prefetch_value, "hash")) {
            if (!w->is_hash) return;
        } else {
            char *end = NULL;
            learned_ranks = strtol(prefetch_value, &end, 10);
            if (end == prefetch_value || *end != '\0' || learned_ranks < 0 ||
                learned_ranks > topk) learned_ranks = 1;
        }
    }
    /* Rank zero is a useful control: exact hash routes are prefetched while
     * learned layers avoid both prediction GEMVs and speculative reads. */
    if (!debug_predict && learned_ranks == 0 && !w->is_hash) return;

    m->route_prediction_count = topk;
    if (w->is_hash) {
        for (int k = 0; k < topk; k++)
            m->route_prediction[k] =
                (int)w->tid2eid[(size_t)token_id * topk + k];
    } else {
        DSV4MoEScratch scratch;
        bind_moe_scratch(m, &scratch);
        int candidates = 0;
        const char *candidate_env = getenv("DSV4_ROUTE_PREDICT_CANDIDATES");
        if (!debug_predict && learned_ranks == 1 && w->bias && candidate_env) {
            char *end = NULL;
            long requested = strtol(candidate_env, &end, 10);
            if (end != candidate_env && *end == '\0' && requested >= 1 &&
                requested < m->cfg.n_experts) candidates = (int)requested;
        }
        if (candidates > 0) {
            for (int k = 0; k < candidates; k++) {
                int best = -1;
                for (int e = 0; e < m->cfg.n_experts; e++) {
                    int used = 0;
                    for (int j = 0; j < k; j++)
                        if (scratch.order[j] == e) { used = 1; break; }
                    if (!used && (best < 0 || w->bias[e] > w->bias[best]))
                        best = e;
                }
                scratch.order[k] = best;
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
#endif
            for (int k = 0; k < candidates; k++) {
                int e = scratch.order[k];
                const uint16_t *row = w->gate_w + (size_t)e * m->cfg.hidden;
                float score = 0.0f;
                for (int i = 0; i < m->cfg.hidden; i++)
                    score = fmaf(bf16f(row[i]), attention_input[i], score);
                scratch.score[e] = score;
                scratch.choice[e] = sqrt_softplus(score) + w->bias[e];
            }
            int best = scratch.order[0];
            for (int k = 1; k < candidates; k++) {
                int e = scratch.order[k];
                if (scratch.choice[e] > scratch.choice[best]) best = e;
            }
            m->route_prediction[0] = best;
            m->route_prediction_count = 1;
        } else {
            dsv4_gemv_bf16(scratch.score, attention_input, w->gate_w,
                            m->cfg.hidden, m->cfg.n_experts, 0);
            for (int e = 0; e < m->cfg.n_experts; e++)
                scratch.choice[e] = sqrt_softplus(scratch.score[e]) +
                                    (w->bias ? w->bias[e] : 0.0f);
        }
        if (m->route_prediction_count != 1) {
            for (int k = 0; k < topk; k++) {
                int best = -1;
                float best_value = -INFINITY;
                for (int e = 0; e < m->cfg.n_experts; e++) {
                    int used = 0;
                    for (int j = 0; j < k; j++)
                        if (m->route_prediction[j] == e) { used = 1; break; }
                    if (!used && scratch.choice[e] > best_value) {
                        best_value = scratch.choice[e];
                        best = e;
                    }
                }
                m->route_prediction[k] = best;
            }
        }
    }
    const int kind = w->is_hash ? 1 : 0;
    for (int k = 0; k < m->route_prediction_count; k++) {
        m->route_prediction_was_miss[k] =
            cache_find(m, m->cur_layer, m->route_prediction[k]) < 0;
        m->route_prediction_predicted_misses[kind] +=
            m->route_prediction_was_miss[k];
        m->route_prediction_rank_misses[kind][k] +=
            m->route_prediction_was_miss[k];
    }
    m->route_prediction_valid = 1;

    if (!prefetch || !prefetch_value || expert_mmap_enabled(m)) return;
    int ranks = w->is_hash ? topk : (int)learned_ranks;
    if (ranks > m->route_prediction_count) ranks = m->route_prediction_count;
    prefetch->count = 0;
    prefetch->active = 0;
    for (int rank = 0; rank < ranks; rank++) {
        if (!m->route_prediction_was_miss[rank]) continue;
        prefetch->experts[prefetch->count++] = m->route_prediction[rank];
    }
    for (int i = 0; i < prefetch->count; i++) {
        for (int j = i + 1; j < prefetch->count; j++) {
            if (prefetch->experts[j] < prefetch->experts[i]) {
                int tmp = prefetch->experts[i];
                prefetch->experts[i] = prefetch->experts[j];
                prefetch->experts[j] = tmp;
            }
        }
    }
    if (prefetch->count > 0) {
        prefetch->temporary = 1;
        prefetch->active = expert_temp_batch_begin(
            m, m->cur_layer, prefetch->experts, prefetch->count,
            prefetch->slot_ids, prefetch->is_miss, &prefetch->batch);
    }
}

static void route_prefetch_finish_batch(DSV4RoutePrefetch *prefetch,
                                        const int *actual_experts,
                                        int n_actual)
{
    if (!prefetch || !prefetch->active) return;
    if (prefetch->temporary)
        expert_temp_batch_finish(&prefetch->batch, actual_experts, n_actual);
    else
        expert_batch_finish(&prefetch->batch);
    prefetch->active = 0;
}

static void route_prefetch_finish(DSV4RoutePrefetch *prefetch)
{
    if (prefetch->temporary) return;
    if (!prefetch->active) return;
    expert_batch_finish(&prefetch->batch);
    prefetch->active = 0;
}

static void route_prediction_record_values(
    DSV4Model *m, const DSV4LayerW *w, const int *predicted,
    const unsigned char *predicted_was_miss, int predicted_count,
    const int *actual)
{
    const int topk = m->cfg.topk;
    int matches = 0;
    for (int p = 0; p < predicted_count; p++) {
        for (int k = 0; k < topk; k++) {
            if (predicted[p] == actual[k]) {
                matches++;
                break;
            }
        }
    }
    const int kind = w->is_hash ? 1 : 0;
    for (int k = 0; k < topk; k++) {
        if (cache_find(m, m->cur_layer, actual[k]) < 0)
            m->route_prediction_actual_misses[kind]++;
    }
    for (int p = 0; p < predicted_count; p++) {
        if (!predicted_was_miss[p]) continue;
        int is_actual = 0;
        for (int k = 0; k < topk; k++) {
            if (predicted[p] == actual[k]) {
                is_actual = 1;
                break;
            }
        }
        if (is_actual) {
            m->route_prediction_covered_misses[kind]++;
            m->route_prediction_rank_covered[kind][p]++;
        } else {
            m->route_prediction_wasted_misses[kind]++;
            m->route_prediction_rank_wasted[kind][p]++;
        }
    }
    m->route_prediction_groups[kind]++;
    m->route_prediction_matches[kind] += matches;
    m->route_prediction_hist[kind][matches]++;
}

static void route_prediction_record(DSV4Model *m, const DSV4LayerW *w,
                                    const int *actual)
{
    if (!m->route_prediction_valid) return;
    route_prediction_record_values(
        m, w, m->route_prediction, m->route_prediction_was_miss,
        m->route_prediction_count, actual);
    m->route_prediction_valid = 0;
}

#define DSV4_MOE_BATCH_TOKENS 32

static int route_prediction_observe_batch(
    DSV4Model *m, const DSV4LayerW *w, const float *attention_input,
    const int *token_ids, int n_tokens, int *predicted,
    unsigned char *predicted_was_miss)
{
    const char *prefetch_value = getenv("DSV4_BATCH_ROUTE_PREFETCH");
    if ((!getenv("DSV4_DEBUG_ROUTE_PREDICT") && !prefetch_value) ||
        n_tokens < 1 || n_tokens > DSV4_MOE_BATCH_TOKENS)
        return 0;
    if (prefetch_value &&
        ((!strcmp(prefetch_value, "hash") && !w->is_hash) ||
         (!strcmp(prefetch_value, "learned") && w->is_hash)))
        return 0;

    const int topk = m->cfg.topk;
    const int ne = m->cfg.n_experts;
    const int hid = m->cfg.hidden;
    if (w->is_hash) {
        for (int t = 0; t < n_tokens; t++) {
            for (int k = 0; k < topk; k++)
                predicted[(size_t)t * topk + k] =
                    (int)w->tid2eid[(size_t)token_ids[t] * topk + k];
        }
    } else {
        float *scores = (float *)malloc(
            (size_t)n_tokens * ne * sizeof(float));
        if (!scores) {
            fprintf(stderr, "dsv4: OOM batch route prediction\n");
            exit(1);
        }
        const float *inputs[DSV4_MOE_BATCH_TOKENS];
        float *outputs[DSV4_MOE_BATCH_TOKENS];
        for (int t = 0; t < n_tokens; t++) {
            inputs[t] = attention_input + (size_t)t * hid;
            outputs[t] = scores + (size_t)t * ne;
        }
        dsv4_gemv_bf16_batch(outputs, inputs, n_tokens, w->gate_w,
                             hid, ne, 0);
        for (int t = 0; t < n_tokens; t++) {
            float *token_scores = scores + (size_t)t * ne;
            for (int e = 0; e < ne; e++)
                token_scores[e] = sqrt_softplus(token_scores[e]) +
                                  (w->bias ? w->bias[e] : 0.0f);
            for (int k = 0; k < topk; k++) {
                int best = -1;
                float best_value = -INFINITY;
                for (int e = 0; e < ne; e++) {
                    int used = 0;
                    for (int j = 0; j < k; j++)
                        if (predicted[(size_t)t * topk + j] == e) {
                            used = 1;
                            break;
                        }
                    if (!used && token_scores[e] > best_value) {
                        best_value = token_scores[e];
                        best = e;
                    }
                }
                predicted[(size_t)t * topk + k] = best;
            }
        }
        free(scores);
    }

    const int kind = w->is_hash ? 1 : 0;
    for (int t = 0; t < n_tokens; t++) {
        for (int k = 0; k < topk; k++) {
            const size_t i = (size_t)t * topk + k;
            predicted_was_miss[i] =
                cache_find(m, m->cur_layer, predicted[i]) < 0;
            m->route_prediction_predicted_misses[kind] +=
                predicted_was_miss[i];
            m->route_prediction_rank_misses[kind][k] +=
                predicted_was_miss[i];
        }
    }
    return topk;
}

static void route_prefetch_begin_batch(
    DSV4Model *m, const DSV4LayerW *w, const int *predicted,
    const unsigned char *predicted_was_miss, int predicted_count,
    int n_tokens, DSV4RoutePrefetch *prefetch)
{
    memset(prefetch, 0, sizeof(*prefetch));
    const char *value = getenv("DSV4_BATCH_ROUTE_PREFETCH");
    if (!value || predicted_count < 1 || expert_mmap_enabled(m)) return;
    if ((!strcmp(value, "hash") && !w->is_hash) ||
        (!strcmp(value, "learned") && w->is_hash))
        return;

    char *end = NULL;
    long ranks = strtol(value, &end, 10);
    if (end == value || *end != '\0' || ranks < 1 ||
        ranks > predicted_count)
        ranks = 1;
    for (int rank = 0; rank < ranks; rank++) {
        for (int t = 0; t < n_tokens; t++) {
            const size_t i = (size_t)t * predicted_count + rank;
            if (!predicted_was_miss[i]) continue;
            const int expert = predicted[i];
            int duplicate = 0;
            for (int j = 0; j < prefetch->count; j++)
                if (prefetch->experts[j] == expert) {
                    duplicate = 1;
                    break;
                }
            if (!duplicate && prefetch->count < DSV4_MAX_TOPK)
                prefetch->experts[prefetch->count++] = expert;
        }
    }
    if (prefetch->count > 0) {
        prefetch->temporary = 1;
        prefetch->active = expert_temp_batch_begin(
            m, m->cur_layer, prefetch->experts, prefetch->count,
            prefetch->slot_ids, prefetch->is_miss, &prefetch->batch);
    }
}

static void routed_swiglu_quant(float *y, float *qy, float *qscale,
                                const float *gate, const float *up, int n,
                                int limit, float route_weight)
{
    if (getenv("DSV4_NO_FUSED_SWIGLU_QUANT")) {
        for (int i = 0; i < n; i++) {
            float g = gate[i], u = up[i];
            if (u > limit) u = (float)limit;
            else if (u < -limit) u = (float)(-limit);
            if (g > limit) g = (float)limit;
            float silu = g / (1.0f + expf(-g));
            y[i] = silu * u * route_weight;
        }
        round_bf16_vec(y, n);
        dsv4_quant_fp8_codes(qy, qscale, y, n);
        return;
    }

    for (int base = 0; base < n; base += DSV4_FP8_BLOCK) {
        int len = n - base;
        if (len > DSV4_FP8_BLOCK) len = DSV4_FP8_BLOCK;
        float amax = 0.0f;
        for (int i = 0; i < len; i++) {
            float g = gate[base + i], u = up[base + i];
            if (u > limit) u = (float)limit;
            else if (u < -limit) u = (float)(-limit);
            if (g > limit) g = (float)limit;
            float silu = g / (1.0f + expf(-g));
            float value = f32bf(silu * u * route_weight);
            y[base + i] = value;
            float magnitude = fabsf(value);
            if (magnitude > amax) amax = magnitude;
        }
        if (amax < 1e-4f) amax = 1e-4f;
        float ratio = amax * (1.0f / 448.0f);
        int exponent;
        float mantissa = frexpf(ratio, &exponent);
        if (mantissa == 0.5f) exponent--;
        float scale = ldexpf(1.0f, exponent);
        qscale[base / DSV4_FP8_BLOCK] = scale;
        float inverse = 1.0f / scale;
        for (int i = 0; i < len; i++) {
            float q = y[base + i] * inverse;
            if (q > 448.0f) q = 448.0f;
            else if (q < -448.0f) q = -448.0f;
            qy[base + i] = dsv4_round_e4m3(q);
        }
    }
}

static void expert_forward_l1(DSV4Model *m, const DSV4ExpertSlot *s,
                              const float *qx, const float *qsc,
                              DSV4ExpertWork *b, float weight, int expert_id,
                              int debug)
{
    const int hid = m->cfg.hidden;
    const int mi = m->cfg.moe_inter;
    const int limit = (int)m->cfg.swiglu_limit;

    /* w1/w2/w3 packed FP4 in the weight run: [w1][w2][w3] (verified order) */
    const uint8_t *w1 = s->weights;
    const uint8_t *w3 = s->weights + (size_t)mi * ((hid + 1) / 2)
                                  + (size_t)hid * ((mi + 1) / 2);
    /* scales run: [w1][w2][w3] */
    const uint8_t *s1 = s->scales;
    const uint8_t *s3 = s->scales + (size_t)mi * ((hid + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP)
                                     + (size_t)hid * ((mi + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP);

    dsv4_gemv_fp4_pair_q(b->gate, b->up, qx, qsc,
                         w1, s1, w3, s3, hid, mi);
    if (debug) {
        fprintf(stderr, "  expert=%d weight=%a\n", expert_id, (double)weight);
        debug_vec_summary("x.fp8", qx, hid);
        debug_vec_summary("x.scale", qsc, (hid + 127) / 128);
        debug_vec_summary("gate", b->gate, mi);
        debug_vec_summary("up", b->up, mi);
    }
    routed_swiglu_quant(b->y, b->qy, b->qys, b->gate, b->up, mi,
                        limit, weight);
    if (debug) debug_vec_summary("silu", b->y, mi);
    if (debug) debug_vec_summary("silu.bf16", b->y, mi);
    if (debug) {
        debug_vec_summary("silu.fp8", b->qy, mi);
        debug_vec_summary("silu.scale", b->qys, (mi + 127) / 128);
    }
}

static void expert_forward_l2(DSV4Model *m, const DSV4ExpertSlot *s,
                              const DSV4ExpertWork *b, int debug, float *out)
{
    const int hid = m->cfg.hidden;
    const int mi = m->cfg.moe_inter;
    const uint8_t *w2 = s->weights + (size_t)mi * ((hid + 1) / 2);
    const uint8_t *s2 = s->scales +
        (size_t)mi * ((hid + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP);
    dsv4_gemv_fp4_q(out, b->qy, b->qys, w2, s2, mi, hid);
    if (debug) debug_vec_summary("down", out, hid);
}

static void expert_forward(DSV4Model *m, const DSV4ExpertSlot *s,
                           const float *qx, const float *qsc,
                           DSV4ExpertWork *b, float weight, int expert_id,
                           int debug, float *out)
{
    expert_forward_l1(m, s, qx, qsc, b, weight, expert_id, debug);
    expert_forward_l2(m, s, b, debug, out);
}

static void expert_forward_many(DSV4Model *m, const DSV4ExpertSlot *slot,
                                const float *const *qx,
                                const float *const *qsc,
                                const float *route_weight, float *const *out,
                                int count, int expert_id)
{
    if (count == 1 || getenv("DSV4_NO_BATCH_EXPERT_GEMV")) {
        for (int t = 0; t < count; t++) {
            DSV4ExpertWork work;
            bind_expert_work(m, &work, t);
            expert_forward(m, slot, qx[t], qsc[t], &work, route_weight[t],
                           expert_id, 0, out[t]);
        }
        return;
    }

    const int hid = m->cfg.hidden;
    const int mi = m->cfg.moe_inter;
    const int limit = (int)m->cfg.swiglu_limit;
    const uint8_t *w1 = slot->weights;
    const uint8_t *w2 = slot->weights + (size_t)mi * ((hid + 1) / 2);
    const uint8_t *w3 = slot->weights + (size_t)mi * ((hid + 1) / 2)
                                  + (size_t)hid * ((mi + 1) / 2);
    const uint8_t *s1 = slot->scales;
    const uint8_t *s2 = slot->scales +
        (size_t)mi * ((hid + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP);
    const uint8_t *s3 = slot->scales +
        (size_t)mi * ((hid + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP) +
        (size_t)hid * ((mi + DSV4_FP4_GROUP - 1) / DSV4_FP4_GROUP);

    DSV4ExpertWork work[DSV4_MAX_TOPK + 1];
    float *gate[DSV4_MAX_TOPK + 1], *up[DSV4_MAX_TOPK + 1];
    const float *down_input_const[DSV4_MAX_TOPK + 1];
    const float *down_scale[DSV4_MAX_TOPK + 1];
    for (int t = 0; t < count; t++) {
        bind_expert_work(m, &work[t], t);
        gate[t] = work[t].gate;
        up[t] = work[t].up;
        down_input_const[t] = work[t].qy;
        down_scale[t] = work[t].qys;
    }
    dsv4_gemv_fp4_batch_q(gate, qx, qsc, count, w1, s1, hid, mi);
    dsv4_gemv_fp4_batch_q(up, qx, qsc, count, w3, s3, hid, mi);
    for (int t = 0; t < count; t++)
        routed_swiglu_quant(work[t].y, work[t].qy, work[t].qys,
                            work[t].gate, work[t].up, mi, limit,
                            route_weight[t]);
    dsv4_gemv_fp4_batch_q(out, down_input_const, down_scale, count,
                          w2, s2, mi, hid);
}

/* Shared expert: FP8 GEMVs, same activation but weight 1. */
static void shared_expert_forward(DSV4Model *m, const DSV4LayerW *w,
                                  const float *qx, const float *qsc,
                                  DSV4ExpertWork *b, int debug, float *out)
{
    const int hid = m->cfg.hidden;
    const int mi = m->cfg.moe_inter;
    const int limit = (int)m->cfg.swiglu_limit;

    dsv4_gemv_fp8_pair_q(b->gate, b->up, qx, qsc, w->sh1, w->sh1_s,
                         w->sh3, w->sh3_s, hid, mi);
    if (debug) {
        fprintf(stderr, "  shared\n");
        debug_vec_summary("x.fp8", qx, hid);
        debug_vec_summary("x.scale", qsc, (hid + 127) / 128);
        debug_vec_summary("gate", b->gate, mi);
        debug_vec_summary("up", b->up, mi);
    }
    for (int i = 0; i < mi; i++) {
        float g = b->gate[i], u = b->up[i];
        if (u > limit) u = (float)limit; else if (u < -limit) u = (float)(-limit);
        if (g > limit) g = (float)limit;
        float silu = g / (1.0f + expf(-g));
        b->y[i] = silu * u;
    }
    if (debug) debug_vec_summary("silu", b->y, mi);
    round_bf16_vec(b->y, mi);
    if (debug) debug_vec_summary("silu.bf16", b->y, mi);
    dsv4_quant_fp8_codes(b->qy, b->qys, b->y, mi);
    if (debug) {
        debug_vec_summary("silu.fp8", b->qy, mi);
        debug_vec_summary("silu.scale", b->qys, (mi + 127) / 128);
    }
    dsv4_gemv_fp8_q(out, b->qy, b->qys, w->sh2, w->sh2_s, mi, hid, 1);
    if (debug) debug_vec_summary("down", out, hid);
}

static void shared_expert_forward_many(DSV4Model *m, const DSV4LayerW *w,
                                       const float *qx, const float *qsc,
                                       int qsc_n, int count, float *out)
{
    const int hid = m->cfg.hidden;
    const int mi = m->cfg.moe_inter;
    const int limit = (int)m->cfg.swiglu_limit;
    const int chunk_cap = m->cfg.topk + 1;

    for (int first = 0; first < count; first += chunk_cap) {
        int chunk = count - first;
        if (chunk > chunk_cap) chunk = chunk_cap;
        DSV4ExpertWork work[DSV4_MAX_TOPK + 1];
        const float *inputs[DSV4_MAX_TOPK + 1];
        const float *input_scales[DSV4_MAX_TOPK + 1];
        const float *down_inputs[DSV4_MAX_TOPK + 1];
        const float *down_scales[DSV4_MAX_TOPK + 1];
        float *gate[DSV4_MAX_TOPK + 1];
        float *up[DSV4_MAX_TOPK + 1];
        float *outputs[DSV4_MAX_TOPK + 1];

        for (int t = 0; t < chunk; t++) {
            bind_expert_work(m, &work[t], t);
            inputs[t] = qx + (size_t)(first + t) * hid;
            input_scales[t] = qsc + (size_t)(first + t) * qsc_n;
            gate[t] = work[t].gate;
            up[t] = work[t].up;
            outputs[t] = out + (size_t)(first + t) * hid;
        }
        dsv4_gemv_fp8_batch_q(gate, inputs, input_scales, chunk, w->sh1,
                              w->sh1_s, hid, mi, 1);
        dsv4_gemv_fp8_batch_q(up, inputs, input_scales, chunk, w->sh3,
                              w->sh3_s, hid, mi, 1);
        for (int t = 0; t < chunk; t++) {
            for (int i = 0; i < mi; i++) {
                float g = work[t].gate[i], u = work[t].up[i];
                if (u > limit) u = (float)limit;
                else if (u < -limit) u = (float)(-limit);
                if (g > limit) g = (float)limit;
                work[t].y[i] = (g / (1.0f + expf(-g))) * u;
            }
            round_bf16_vec(work[t].y, mi);
            dsv4_quant_fp8_codes(work[t].qy, work[t].qys, work[t].y, mi);
            down_inputs[t] = work[t].qy;
            down_scales[t] = work[t].qys;
        }
        dsv4_gemv_fp8_batch_q(outputs, down_inputs, down_scales, chunk, w->sh2,
                              w->sh2_s, mi, hid, 1);
    }
}

static int expert_forward_batch_routes(DSV4Model *m,
                                       const DSV4ExpertSlot *slot,
                                       int expert, const float *qx,
                                       const float *qsc, int qsc_n,
                                       const int *idx, const float *wt,
                                       int n_tokens, float *acc,
                                       float *routed,
                                       float *route_outputs,
                                       int record_route_touches)
{
    const int hid = m->cfg.hidden;
    const int topk = m->cfg.topk;
    const int chunk_cap = topk + 1;
    int next_token = 0;
    int total_routes = 0;
    while (next_token < n_tokens) {
        const float *inputs[DSV4_MAX_TOPK + 1];
        const float *scales[DSV4_MAX_TOPK + 1];
        float weights[DSV4_MAX_TOPK + 1];
        float *outputs[DSV4_MAX_TOPK + 1];
        int tokens[DSV4_MAX_TOPK + 1];
        int count = 0;
        while (next_token < n_tokens && count < chunk_cap) {
            const int t = next_token++;
            const int *token_idx = idx + (size_t)t * topk;
            const float *token_wt = wt + (size_t)t * topk;
            int route = -1;
            for (int k = 0; k < topk; k++)
                if (token_idx[k] == expert) { route = k; break; }
            if (route < 0) continue;
            tokens[count] = t;
            inputs[count] = qx + (size_t)t * hid;
            scales[count] = qsc + (size_t)t * qsc_n;
            weights[count] = token_wt[route];
            outputs[count] = route_outputs
                ? route_outputs + ((size_t)t * topk + route) * hid
                : routed + (size_t)count * hid;
            count++;
        }
        if (count == 0) continue;
        total_routes += count;
        expert_forward_many(m, slot, inputs, scales, weights, outputs, count,
                            expert);
        if (!route_outputs) {
            for (int route = 0; route < count; route++) {
                float *token_acc = acc + (size_t)tokens[route] * hid;
                const float *expert_out = outputs[route];
                for (int i = 0; i < hid; i++) token_acc[i] += expert_out[i];
            }
        }
    }
    /* A batched dispatch fetches each unique expert once, but its weights may
     * serve several token routes. Let the experimental cache policy observe
     * those semantic reuses so a one-off expert does not look equally hot. */
    if (record_route_touches && total_routes > 1 &&
        slot->layer == m->cur_layer && slot->expert == expert) {
        const int slot_id = (int)(slot - m->cache);
        for (int use = 1; use < total_routes; use++)
            cache_touch(m, m->cur_layer, slot_id);
    }
    return total_routes;
}

static void expert_forward_batch_group(
    DSV4Model *m, const int *experts, int count, int *slot_ids,
    const int *is_miss, DSV4ExpertBatch *batch, int async_reads,
    const float *qx, const float *qsc, int qsc_n, const int *idx,
    const float *wt, int n_tokens, float *acc, float *routed,
    float *route_outputs, int record_route_touches)
{
    if (!async_reads) {
        for (int u = 0; u < count; u++)
            (void)expert_forward_batch_routes(
                m, &m->cache[slot_ids[u]], experts[u], qx, qsc, qsc_n,
                idx, wt, n_tokens, acc, routed, route_outputs,
                record_route_touches);
        return;
    }

    unsigned char computed[DSV4_MAX_TOPK] = {0};
    int route_count[DSV4_MAX_TOPK] = {0};
    for (int u = 0; u < count; u++) {
        if (is_miss[u]) continue;
        route_count[u] = expert_forward_batch_routes(
            m, &m->cache[slot_ids[u]], experts[u], qx, qsc, qsc_n,
            idx, wt, n_tokens, acc, routed, route_outputs,
            0);
        computed[u] = 1;
    }
    for (int done = 0; done < batch->miss_count; done++) {
        int u = expert_batch_next_ready(batch, computed);
        if (u < 0) break;
        DSV4ExpertSlot *slot = expert_batch_wait_one(batch, u);
        route_count[u] = expert_forward_batch_routes(
            m, slot, experts[u], qx, qsc, qsc_n, idx, wt, n_tokens,
            acc, routed, route_outputs, 0);
        computed[u] = 1;
    }
    expert_batch_finish(batch);
    /* Publish every miss and touch every hit before recording semantic reuse.
     * This gives hits and misses the same batch boundary and keeps replacement
     * metadata independent of which O_DIRECT read happened to complete first. */
    if (record_route_touches) {
        for (int u = 0; u < count; u++) {
            if (route_count[u] <= 1) continue;
            for (int use = 1; use < route_count[u]; use++)
                cache_touch(m, m->cur_layer, slot_ids[u]);
        }
    }
}

/* MoE: gate, top-k, then routed experts in ascending expert-id order (the
 * released dispatch order), plus the shared expert. Output bf16-rounded. */
static void moe_forward(DSV4Model *m, const DSV4LayerW *w, const float *x,
                        int token_id, int position,
                        DSV4RoutePrefetch *early_prefetch, float *out)
{
    const int hid = m->cfg.hidden;
    const int ne = m->cfg.n_experts;
    const int topk = m->cfg.topk;

    DSV4MoEScratch scratch;
    bind_moe_scratch(m, &scratch);
    float *score = scratch.score;
    float *choice = scratch.choice;
    int *order = scratch.order;
    float *wt = scratch.wt;
    int *idx = scratch.idx;

    dsv4_gemv_bf16(score, x, w->gate_w, hid, ne, 0);   /* fp32                 */
    const int debug = debug_moe_at(m, position);
    const int detail = debug && debug_moe_detail();
    if (debug) {
        fprintf(stderr, "MOE_DETAIL pos=%d L=%d token=%d\n", position, m->cur_layer,
                token_id);
        debug_vec_summary("input", x, hid);
        debug_vec_summary("gate.raw", score, ne);
        fprintf(stderr, "  gate.raw[0:8]");
        for (int e = 0; e < 8 && e < ne; e++) fprintf(stderr, " %d=%a", e, (double)score[e]);
        fprintf(stderr, "\n");
    }
    for (int e = 0; e < ne; e++) score[e] = sqrt_softplus(score[e]);
    for (int e = 0; e < ne; e++) choice[e] = score[e] + (w->bias ? w->bias[e] : 0.0f);

    if (w->is_hash) {
        for (int k = 0; k < topk; k++) {
            idx[k] = (int)w->tid2eid[(size_t)token_id * topk + k];
            order[k] = idx[k];
        }
    } else {
        for (int k = 0; k < topk; k++) {
            int best = -1;
            float bv = -INFINITY;
            for (int e = 0; e < ne; e++) {
                int used = 0;
                for (int t = 0; t < k; t++) if (order[t] == e) { used = 1; break; }
                if (!used && choice[e] > bv) { bv = choice[e]; best = e; }
            }
            order[k] = best;
            idx[k] = best;
        }
    }
    route_prediction_record(m, w, idx);

    /* combining weights from the UNBIASED scores, normalised, * route_scale */
    float sum = 0.0f;
    for (int k = 0; k < topk; k++) sum += score[idx[k]];
    if (sum > 0.0f) {
        float rs = m->cfg.route_scale / sum;
        for (int k = 0; k < topk; k++) wt[k] = score[idx[k]] * rs;
    } else {
        for (int k = 0; k < topk; k++) wt[k] = 0.0f;
    }
    if (debug) {
        fprintf(stderr, "  route sum=%a scale=%a", (double)sum,
                (double)m->cfg.route_scale);
        for (int k = 0; k < topk; k++)
            fprintf(stderr, " [%d score=%a weight=%a]", idx[k],
                    (double)score[idx[k]], (double)wt[k]);
        fprintf(stderr, "\n");
    }

    /* ascending expert-id order, carrying weights (released dispatch order) */
    for (int i = 0; i < topk; i++) {
        for (int j = i + 1; j < topk; j++) {
            if (idx[j] < idx[i]) {
                int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
                float tw = wt[i]; wt[i] = wt[j]; wt[j] = tw;
            }
        }
    }

    if (m->expert_trace) {
        for (int k = 0; k < topk; k++)
            fprintf(m->expert_trace, "%d\t%d\t%d\n", m->cur_layer,
                    idx[k], position);
        m->expert_trace_records += topk;
    }

    route_prefetch_finish_batch(early_prefetch, idx, topk);

    /* Start miss reads, then use their I/O window for already-resident routed
     * experts and the independent shared expert. Each result has its own buffer;
     * accumulation still happens below in ascending expert-id order. */
    int slot_ids[DSV4_MAX_TOPK];
    int is_miss[DSV4_MAX_TOPK];
    DSV4ExpertBatch batch;
    int async_reads = expert_batch_begin(m, m->cur_layer, idx, topk,
                                         slot_ids, is_miss, &batch, !detail);
    /* NOTE: layer is per-layer; the caller passes w, so experts live in the
     * layer that loaded them. The batch keys the cache by (layer, expert)
     * using m->cur_layer (set by forward_token). */
    float *acc = scratch.acc;
    dsv4_quant_fp8_codes(scratch.qx, scratch.qsc, x, hid);
    DSV4ExpertWork work;
    bind_expert_work(m, &work, 0);

    int parallel_experts = getenv("DSV4_EXPERT_PARALLEL") != NULL && !detail;
    if (parallel_experts) {
        /* Each routed expert owns its result and workspace. Nested OpenMP is
         * disabled at model open, so inner GEMVs execute on the task's thread;
         * the outer team provides expert-level parallelism. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(topk + 1)
#endif
        for (int task = 0; task <= topk; task++) {
            DSV4ExpertWork task_work;
            bind_expert_work(m, &task_work, task);
            if (task == topk) {
                shared_expert_forward(m, w, scratch.qx, scratch.qsc, &task_work,
                                      0, scratch.shared);
            } else {
                DSV4ExpertSlot *slot = async_reads
                    ? expert_batch_wait_one(&batch, task)
                    : &m->cache[slot_ids[task]];
                expert_forward(m, slot, scratch.qx, scratch.qsc, &task_work,
                               wt[task], idx[task], 0,
                               scratch.routed + (size_t)task * hid);
            }
        }
        if (async_reads) expert_batch_finish(&batch);
    } else if (async_reads) {
        for (int k = 0; k < topk; k++) {
            if (is_miss[k]) continue;
            expert_forward(m, &m->cache[slot_ids[k]], scratch.qx, scratch.qsc,
                           &work, wt[k], idx[k], detail,
                           scratch.routed + (size_t)k * hid);
        }
        shared_expert_forward(m, w, scratch.qx, scratch.qsc, &work, detail,
                              scratch.shared);
        const int l1_pipeline = getenv("DSV4_EXPERT_L1_PIPELINE") != NULL &&
                                m->expert_pool != NULL;
        if (l1_pipeline) {
            unsigned char l1_computed[DSV4_MAX_TOPK] = {0};
            unsigned char l2_computed[DSV4_MAX_TOPK] = {0};
            DSV4ExpertWork miss_work[DSV4_MAX_TOPK];
            for (int k = 0; k < topk; k++)
                bind_expert_work(m, &miss_work[k], k + 1);
            for (int done = 0; done < batch.miss_count; done++) {
                int k = expert_batch_next_l1_ready(&batch, l1_computed);
                DSV4ExpertSlot *slot = expert_batch_wait_l1(&batch, k);
                expert_forward_l1(m, slot, scratch.qx, scratch.qsc,
                                  &miss_work[k], wt[k], idx[k], detail);
                l1_computed[k] = 1;
            }
            for (int done = 0; done < batch.miss_count; done++) {
                int k = expert_batch_next_ready(&batch, l2_computed);
                DSV4ExpertSlot *slot = expert_batch_wait_one(&batch, k);
                expert_forward_l2(m, slot, &miss_work[k], detail,
                                  scratch.routed + (size_t)k * hid);
                l2_computed[k] = 1;
            }
        /* Routed outputs have independent buffers, so consuming whichever read
         * completes first cannot alter the final expert-id accumulation order. */
        } else if (!getenv("DSV4_EXPERT_ID_ORDER")) {
            unsigned char computed[DSV4_MAX_TOPK] = {0};
            for (int done = 0; done < batch.miss_count; done++) {
                int k = expert_batch_next_ready(&batch, computed);
                if (k < 0) break;
                DSV4ExpertSlot *slot = expert_batch_wait_one(&batch, k);
                expert_forward(m, slot, scratch.qx, scratch.qsc, &work, wt[k],
                               idx[k], detail,
                               scratch.routed + (size_t)k * hid);
                computed[k] = 1;
            }
        } else {
            for (int k = 0; k < topk; k++) {
                if (!is_miss[k]) continue;
                DSV4ExpertSlot *slot = expert_batch_wait_one(&batch, k);
                expert_forward(m, slot, scratch.qx, scratch.qsc, &work, wt[k],
                               idx[k], detail,
                               scratch.routed + (size_t)k * hid);
            }
        }
        expert_batch_finish(&batch);
    } else {
        if (expert_mmap_enabled(m))
            shared_expert_forward(m, w, scratch.qx, scratch.qsc, &work, detail,
                                  scratch.shared);
        for (int k = 0; k < topk; k++) {
            expert_forward(m, &m->cache[slot_ids[k]], scratch.qx, scratch.qsc,
                           &work, wt[k], idx[k], detail,
                           scratch.routed + (size_t)k * hid);
        }
        if (!expert_mmap_enabled(m))
            shared_expert_forward(m, w, scratch.qx, scratch.qsc, &work, detail,
                                  scratch.shared);
    }

    memset(acc, 0, (size_t)hid * sizeof(float));
    for (int k = 0; k < topk; k++) {
        const float *expert_out = scratch.routed + (size_t)k * hid;
        for (int i = 0; i < hid; i++) acc[i] += expert_out[i];
        if (detail) debug_vec_summary("routed.acc", acc, hid);
    }
    for (int i = 0; i < hid; i++) acc[i] += scratch.shared[i];
    if (debug) debug_vec_summary("moe.fp32", acc, hid);
    round_bf16_vec(acc, hid);
    if (debug) debug_vec_summary("moe.bf16", acc, hid);
    memcpy(out, acc, (size_t)hid * sizeof(float));

}

/* Route a token batch first, then evaluate each distinct routed expert once
 * while its weights are resident. Per-token contributions are still added in
 * ascending expert-id order, exactly matching moe_forward's accumulation. */
static void moe_forward_batch(DSV4Model *m, const DSV4LayerW *w,
                              const float *x, const int *token_ids,
                              int start_position, int n_tokens,
                              const int *predicted_routes,
                              const unsigned char *predicted_misses,
                              int predicted_count,
                              DSV4RoutePrefetch *early_prefetch, float *out)
{
    if (n_tokens <= 1 || getenv("DSV4_DEBUG_MOE") ||
        getenv("DSV4_NO_BATCH_MOE")) {
        for (int t = 0; t < n_tokens; t++)
            moe_forward(m, w, x + (size_t)t * m->cfg.hidden, token_ids[t],
                        start_position + t, NULL,
                        out + (size_t)t * m->cfg.hidden);
        return;
    }

    const int hid = m->cfg.hidden;
    const int ne = m->cfg.n_experts;
    const int topk = m->cfg.topk;
    const int qsc_n = (hid + 127) / 128;
    float *qx = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *qsc = (float *)malloc((size_t)n_tokens * qsc_n * sizeof(float));
    float *wt = (float *)malloc((size_t)n_tokens * topk * sizeof(float));
    float *acc = (float *)calloc((size_t)n_tokens * hid, sizeof(float));
    float *shared = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *routed = (float *)malloc(
        (size_t)(DSV4_MAX_TOPK + 1) * hid * sizeof(float));
    const int physical_order = getenv("DSV4_BATCH_EXPERT_PHYSICAL") != NULL;
    const int record_route_touches =
        getenv("DSV4_NO_BATCH_CACHE_ROUTE_TOUCHES") == NULL;
    float *route_outputs = NULL;
    float *score = (float *)malloc((size_t)ne * sizeof(float));
    float *choice = (float *)malloc((size_t)ne * sizeof(float));
    int *idx = (int *)malloc((size_t)n_tokens * topk * sizeof(int));
    int *order = (int *)malloc((size_t)topk * sizeof(int));
    int *unique = (int *)malloc((size_t)ne * sizeof(int));
    unsigned char *used = (unsigned char *)calloc((size_t)ne, 1);
    if (!qx || !qsc || !wt || !acc || !shared || !routed ||
        !score || !choice || !idx || !order || !unique || !used) {
        fprintf(stderr, "dsv4: OOM batched MoE\n");
        exit(1);
    }

    DSV4ExpertWork work;
    bind_expert_work(m, &work, 0);
    const int mapped_experts = expert_mmap_enabled(m);
    const int pipeline_requested =
        getenv("DSV4_NO_BATCH_MOE_PIPELINE") == NULL;
    for (int t = 0; t < n_tokens; t++) {
        const float *xt = x + (size_t)t * hid;
        int *ti = idx + (size_t)t * topk;
        float *tw = wt + (size_t)t * topk;
        dsv4_gemv_bf16(score, xt, w->gate_w, hid, ne, 0);
        for (int e = 0; e < ne; e++) score[e] = sqrt_softplus(score[e]);
        for (int e = 0; e < ne; e++)
            choice[e] = score[e] + (w->bias ? w->bias[e] : 0.0f);

        if (w->is_hash) {
            for (int k = 0; k < topk; k++) {
                ti[k] = (int)w->tid2eid[(size_t)token_ids[t] * topk + k];
                order[k] = ti[k];
            }
        } else {
            for (int k = 0; k < topk; k++) {
                int best = -1;
                float bv = -INFINITY;
                for (int e = 0; e < ne; e++) {
                    int already = 0;
                    for (int j = 0; j < k; j++)
                        if (order[j] == e) { already = 1; break; }
                    if (!already && choice[e] > bv) { bv = choice[e]; best = e; }
                }
                order[k] = best;
                ti[k] = best;
            }
        }

        float sum = 0.0f;
        for (int k = 0; k < topk; k++) sum += score[ti[k]];
        if (sum > 0.0f) {
            float rs = m->cfg.route_scale / sum;
            for (int k = 0; k < topk; k++) tw[k] = score[ti[k]] * rs;
        } else {
            for (int k = 0; k < topk; k++) tw[k] = 0.0f;
        }
        for (int i = 0; i < topk; i++) {
            for (int j = i + 1; j < topk; j++) {
                if (ti[j] < ti[i]) {
                    int ei = ti[i]; ti[i] = ti[j]; ti[j] = ei;
                    float ew = tw[i]; tw[i] = tw[j]; tw[j] = ew;
                }
            }
            used[ti[i]] = 1;
        }
        if (predicted_count > 0) {
            route_prediction_record_values(
                m, w, predicted_routes + (size_t)t * predicted_count,
                predicted_misses + (size_t)t * predicted_count,
                predicted_count, ti);
        }
        if (m->expert_trace) {
            for (int k = 0; k < topk; k++)
                fprintf(m->expert_trace, "%d\t%d\t%d\n", m->cur_layer,
                        ti[k], start_position + t);
            m->expert_trace_records += topk;
        }

        float *tqx = qx + (size_t)t * hid;
        float *tqsc = qsc + (size_t)t * qsc_n;
        dsv4_quant_fp8_codes(tqx, tqsc, xt, hid);
    }

    int nunique = 0;
    for (int e = 0; e < ne; e++) if (used[e]) unique[nunique++] = e;
    route_prefetch_finish_batch(early_prefetch, unique, nunique);
    if (physical_order) {
        for (int u = 1; u < nunique; u++) {
            int item = unique[u];
            DSV4ExpertMeta *item_meta = expert_meta(m, m->cur_layer, item);
            int j = u;
            while (j > 0) {
                DSV4ExpertMeta *previous =
                    expert_meta(m, m->cur_layer, unique[j - 1]);
                if (previous->shard < item_meta->shard ||
                    (previous->shard == item_meta->shard &&
                     previous->weight_off <= item_meta->weight_off))
                    break;
                unique[j] = unique[j - 1];
                j--;
            }
            unique[j] = item;
        }
    }
    if (mapped_experts) {
        for (int u = 0; u < nunique; u++)
            expert_mmap_advise(m, expert_meta(m, m->cur_layer, unique[u]));
    }
    int cache_begin, cache_end;
    cache_bounds(m, m->cur_layer, &cache_begin, &cache_end);
    int group_cap = cache_end - cache_begin;
    if (group_cap > DSV4_MAX_TOPK) group_cap = DSV4_MAX_TOPK;
    if (group_cap < 1) group_cap = 1;
    const int cache_capacity = cache_end - cache_begin;
    /* A group that occupies the whole per-layer partition leaves no victims
     * for the next read batch. Halving it provides two disjoint slot sets so
     * long prefill batches can actually use the existing double buffer. */
    int half_groups = getenv("DSV4_NO_BATCH_MOE_HALF_GROUPS") == NULL;
    if (getenv("DSV4_BATCH_MOE_HALF_GROUPS")) half_groups = 1;
    if (half_groups && nunique > group_cap && cache_capacity >= 2) {
        group_cap = cache_capacity / 2;
        if (group_cap > DSV4_MAX_TOPK) group_cap = DSV4_MAX_TOPK;
        if (group_cap < 1) group_cap = 1;
    }
    const int pipeline = pipeline_requested && !mapped_experts &&
        getenv("DSV4_NO_EXPERT_OVERLAP") == NULL && nunique > 0 &&
        (nunique <= group_cap || cache_capacity >= 2 * group_cap);
    const int ready_order = pipeline &&
        getenv("DSV4_NO_BATCH_MOE_READY_ORDER") == NULL;
    if (ready_order && getenv("DSV4_BATCH_HITS_FIRST")) {
        int hit_count = 0;
        for (int u = 0; u < nunique; u++) {
            if (cache_find(m, m->cur_layer, unique[u]) < 0) continue;
            const int expert = unique[u];
            for (int j = u; j > hit_count; j--)
                unique[j] = unique[j - 1];
            unique[hit_count++] = expert;
        }
    }
    if (physical_order || ready_order) {
        route_outputs = (float *)malloc(
            (size_t)n_tokens * topk * hid * sizeof(float));
        if (!route_outputs) {
            fprintf(stderr, "dsv4: OOM batched MoE route outputs\n");
            exit(1);
        }
    }
    if (!pipeline) {
        shared_expert_forward_many(m, w, qx, qsc, qsc_n, n_tokens, shared);
    }

    if (ready_order) {
        int slot_ids[2][DSV4_MAX_TOPK];
        int is_miss[2][DSV4_MAX_TOPK];
        DSV4ExpertBatch batches[2];
        int group_start[2] = {0, 0};
        int group_count[2] = {0, 0};
        int async_reads[2] = {0, 0};
        int current = 0;

        group_count[current] = nunique < group_cap ? nunique : group_cap;
        async_reads[current] = expert_batch_begin_pinned(
            m, m->cur_layer, unique, group_count[current],
            slot_ids[current], is_miss[current], &batches[current], 1,
            NULL, 0);

        shared_expert_forward_many(m, w, qx, qsc, qsc_n, n_tokens, shared);
        for (;;) {
            const int next = 1 - current;
            group_start[next] = group_start[current] + group_count[current];
            group_count[next] = nunique - group_start[next];
            if (group_count[next] > group_cap) group_count[next] = group_cap;
            if (group_count[next] > 0) {
                int pinned[DSV4_MAX_TOPK];
                for (int u = 0; u < group_count[current]; u++) {
                    pinned[u] = async_reads[current] && is_miss[current][u]
                        ? batches[current].victim[u]
                        : slot_ids[current][u];
                }
                async_reads[next] = expert_batch_begin_pinned(
                    m, m->cur_layer, unique + group_start[next],
                    group_count[next], slot_ids[next], is_miss[next],
                    &batches[next], 1, pinned, group_count[current]);
            }

            expert_forward_batch_group(
                m, unique + group_start[current], group_count[current],
                slot_ids[current], is_miss[current], &batches[current],
                async_reads[current], qx, qsc, qsc_n, idx, wt, n_tokens,
                acc, routed, route_outputs, record_route_touches);
            if (group_count[next] == 0) break;
            current = next;
        }
    } else if (pipeline) {
        int u0 = 0;
        int count = nunique < group_cap ? nunique : group_cap;
        int slot_ids[DSV4_MAX_TOPK], is_miss[DSV4_MAX_TOPK];
        DSV4ExpertBatch first_batch;
        int async_reads = expert_batch_begin_pinned(
            m, m->cur_layer, unique, count, slot_ids, is_miss,
            &first_batch, 1, NULL, 0);

        /* The shared expert is independent of routed expert weights, so it
         * fills the first read window without changing accumulation order. */
        shared_expert_forward_many(m, w, qx, qsc, qsc_n, n_tokens, shared);
        if (async_reads) expert_batch_finish(&first_batch);

        while (u0 < nunique) {
            int next_u0 = u0 + count;
            int next_count = nunique - next_u0;
            if (next_count > group_cap) next_count = group_cap;
            int next_slot_ids[DSV4_MAX_TOPK], next_is_miss[DSV4_MAX_TOPK];
            DSV4ExpertBatch next_batch;
            int next_async = 0;
            if (next_count > 0) {
                next_async = expert_batch_begin_pinned(
                    m, m->cur_layer, unique + next_u0, next_count,
                    next_slot_ids, next_is_miss, &next_batch, 1,
                    slot_ids, count);
            }

            for (int u = 0; u < count; u++) {
                const int expert = unique[u0 + u];
                const DSV4ExpertSlot *slot = &m->cache[slot_ids[u]];
                expert_forward_batch_routes(m, slot, expert, qx, qsc, qsc_n,
                                            idx, wt, n_tokens, acc, routed,
                                            route_outputs,
                                            record_route_touches);
            }

            if (next_count > 0) {
                if (next_async) expert_batch_finish(&next_batch);
                for (int u = 0; u < next_count; u++)
                    slot_ids[u] = next_slot_ids[u];
            }
            u0 = next_u0;
            count = next_count;
        }
    } else for (int u0 = 0; u0 < nunique; u0 += group_cap) {
        int count = nunique - u0;
        if (count > group_cap) count = group_cap;
        int slot_ids[DSV4_MAX_TOPK], is_miss[DSV4_MAX_TOPK];
        expert_get_many(m, m->cur_layer, unique + u0, count, slot_ids,
                        is_miss);
        for (int u = 0; u < count; u++) {
            const int expert = unique[u0 + u];
            const DSV4ExpertSlot *slot = &m->cache[slot_ids[u]];
            expert_forward_batch_routes(m, slot, expert, qx, qsc, qsc_n,
                                        idx, wt, n_tokens, acc, routed,
                                        route_outputs,
                                        record_route_touches);
        }
    }

    if (route_outputs) {
        for (int t = 0; t < n_tokens; t++) {
            float *token_acc = acc + (size_t)t * hid;
            for (int k = 0; k < topk; k++) {
                const float *expert_out = route_outputs +
                    ((size_t)t * topk + k) * hid;
                for (int i = 0; i < hid; i++) token_acc[i] += expert_out[i];
            }
        }
    }
    for (int t = 0; t < n_tokens; t++) {
        float *ta = acc + (size_t)t * hid;
        const float *ts = shared + (size_t)t * hid;
        for (int i = 0; i < hid; i++) ta[i] += ts[i];
        round_bf16_vec(ta, hid);
        memcpy(out + (size_t)t * hid, ta, (size_t)hid * sizeof(float));
    }

    free(qx); free(qsc); free(wt); free(acc); free(shared); free(routed);
    free(route_outputs);
    free(score); free(choice); free(idx); free(order); free(unique); free(used);
}

/* ====================================================== hyper conn ==== */

/* hc_pre: mixes = hc_fn @ state_flat * rsqrt; sinkhorn; bin = pre . state.
 * Writes bin (bf16), post[mult], comb[mult*mult]. */
static void hc_pre_scratch(DSV4Model *m, const DSV4LayerW *w,
                           const float *state, const float *fn,
                           const float *scale3, const float *base,
                           float *bin, float *post, float *comb,
                           float *mixes)
{
    const int mult = m->cfg.hc_mult;
    const int hid = m->cfg.hidden;
    const int mix_n = (2 + mult) * mult;

    double ss = 0.0;
    for (int i = 0; i < mult * hid; i++) ss += (double)state[i] * (double)state[i];
    float rsqrt = 1.0f / sqrtf((float)(ss / (mult * hid)) + m->cfg.rms_eps);

    /* mixes[j] = (fn[j] . state) * rsqrt */
    for (int j = 0; j < mix_n; j++) {
        const float *row = fn + (size_t)j * (mult * hid);
        float acc = 0.0f;
        for (int i = 0; i < mult * hid; i++) acc = fmaf(row[i], state[i], acc);
        mixes[j] = acc * rsqrt;
    }
    float pre[8];
    dsv4_hc_split(pre, post, comb, mixes, scale3, base, mult, m->cfg.hc_iters,
                  m->cfg.hc_eps);

    /* bin = sum_i pre[i] * state[i] (bf16) */
    for (int j = 0; j < hid; j++) {
        float acc = 0.0f;
        for (int i = 0; i < mult; i++) acc = fmaf(pre[i], state[(size_t)i * hid + j], acc);
        bin[j] = f32bf(acc);
    }
}

static void hc_pre(DSV4Model *m, const DSV4LayerW *w, const float *state,
                   const float *fn, const float *scale3, const float *base,
                   float *bin, float *post, float *comb)
{
    hc_pre_scratch(m, w, state, fn, scale3, base, bin, post, comb,
                   m->mixes_buf);
}

/* hc_post: state = post . branch_out + comb @ state, bf16-rounded. */
static void hc_post_scratch(float *state, const float *branch_out,
                            const float *post, const float *comb, int mult,
                            int hid, float *newst)
{
    for (int i = 0; i < mult; i++) {
        for (int j = 0; j < hid; j++) {
            float y = post[i] * branch_out[j];
            for (int k = 0; k < mult; k++) y = fmaf(comb[(size_t)i * mult + k], state[(size_t)k * hid + j], y);
            newst[(size_t)i * hid + j] = y;
        }
    }
    for (int i = 0; i < mult * hid; i++) state[i] = f32bf(newst[i]);
}

static void hc_post(DSV4Model *m, float *state, const float *branch_out,
                    const float *post, const float *comb, int mult, int hid)
{
    hc_post_scratch(state, branch_out, post, comb, mult, hid,
                    m->hc_state_buf);
}

/* ============================================================ head ====== */

/* hc_head: mixes = hc_head_fn @ state_flat * rsqrt; pre = sigmoid(m*scale+base)
 * + eps; h = sum pre[i] * state[i] (bf16). Then final RMSNorm. */
static void model_head(DSV4Model *m, const float *state, float *h)
{
    const int mult = m->cfg.hc_mult;
    const int hid = m->cfg.hidden;

    double ss = 0.0;
    for (int i = 0; i < mult * hid; i++) ss += (double)state[i] * (double)state[i];
    float rsqrt = 1.0f / sqrtf((float)(ss / (mult * hid)) + m->cfg.rms_eps);

    float *mixes = m->mixes_buf;
    float pre[8];
    for (int j = 0; j < mult; j++) {
        const float *row = m->hc_head_fn + (size_t)j * (mult * hid);
        float acc = 0.0f;
        for (int i = 0; i < mult * hid; i++) acc = fmaf(row[i], state[i], acc);
        mixes[j] = acc * rsqrt;
        pre[j] = 1.0f / (1.0f + expf(-(mixes[j] * m->hc_head_scale[0] + m->hc_head_base[j])))
                 + m->cfg.hc_eps;
    }
    for (int j = 0; j < hid; j++) {
        float acc = 0.0f;
        for (int i = 0; i < mult; i++) acc = fmaf(pre[i], state[(size_t)i * hid + j], acc);
        h[j] = f32bf(acc);
    }
    /* final RMSNorm with gain */
    rmsnorm_bf16(h, h, m->norm, hid, m->cfg.rms_eps, 1);
}

/* ====================================================== forward ======== */

static void head_logits_block(const float *h, const uint16_t *rows, float *logits,
                              int hid, int output_base, int row_count, int block,
                              int use_simd)
{
    const int local_r = block * 8;
    const int lanes = row_count - local_r < 8 ? row_count - local_r : 8;
#if defined(__AVX2__) && defined(__FMA__)
    if (use_simd && lanes == 8) {
        __m256 acc = _mm256_setzero_ps();
        for (int i = 0; i < hid; i++) {
            __m256 wv = _mm256_set_ps(
                bf16f(rows[(size_t)(local_r + 7) * hid + i]),
                bf16f(rows[(size_t)(local_r + 6) * hid + i]),
                bf16f(rows[(size_t)(local_r + 5) * hid + i]),
                bf16f(rows[(size_t)(local_r + 4) * hid + i]),
                bf16f(rows[(size_t)(local_r + 3) * hid + i]),
                bf16f(rows[(size_t)(local_r + 2) * hid + i]),
                bf16f(rows[(size_t)(local_r + 1) * hid + i]),
                bf16f(rows[(size_t)local_r * hid + i]));
            acc = _mm256_fmadd_ps(wv, _mm256_set1_ps(h[i]), acc);
        }
        _mm256_storeu_ps(logits + output_base + local_r, acc);
        return;
    }
#else
    (void)use_simd;
#endif
    for (int lane = 0; lane < lanes; lane++) {
        float acc = 0.0f;
        const uint16_t *row = rows + (size_t)(local_r + lane) * hid;
        for (int i = 0; i < hid; i++) acc = fmaf(bf16f(row[i]), h[i], acc);
        logits[output_base + local_r + lane] = acc;
    }
}

static void head_logits_block_batch(const float *h, int hidden_stride,
                                    const uint16_t *rows, float *logits,
                                    int logits_stride, int n_tokens, int hid,
                                    int output_base, int row_count, int block,
                                    int use_simd)
{
    if (n_tokens == 1) {
        head_logits_block(h, rows, logits, hid, output_base, row_count,
                          block, use_simd);
        return;
    }
    const int local_r = block * 8;
    const int lanes = row_count - local_r < 8 ? row_count - local_r : 8;
#if defined(__AVX2__) && defined(__FMA__)
    if (use_simd && lanes == 8) {
        __m256 acc[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < n_tokens; t++) acc[t] = _mm256_setzero_ps();
        for (int i = 0; i < hid; i++) {
            __m256 wv = _mm256_set_ps(
                bf16f(rows[(size_t)(local_r + 7) * hid + i]),
                bf16f(rows[(size_t)(local_r + 6) * hid + i]),
                bf16f(rows[(size_t)(local_r + 5) * hid + i]),
                bf16f(rows[(size_t)(local_r + 4) * hid + i]),
                bf16f(rows[(size_t)(local_r + 3) * hid + i]),
                bf16f(rows[(size_t)(local_r + 2) * hid + i]),
                bf16f(rows[(size_t)(local_r + 1) * hid + i]),
                bf16f(rows[(size_t)local_r * hid + i]));
            for (int t = 0; t < n_tokens; t++)
                acc[t] = _mm256_fmadd_ps(
                    wv, _mm256_set1_ps(h[(size_t)t * hidden_stride + i]),
                    acc[t]);
        }
        for (int t = 0; t < n_tokens; t++)
            _mm256_storeu_ps(logits + (size_t)t * logits_stride +
                             output_base + local_r, acc[t]);
        return;
    }
#else
    (void)use_simd;
#endif
    for (int lane = 0; lane < lanes; lane++) {
        float acc[DSV4_MAX_TOPK + 1] = {0};
        const uint16_t *row = rows + (size_t)(local_r + lane) * hid;
        for (int i = 0; i < hid; i++) {
            float weight = bf16f(row[i]);
            for (int t = 0; t < n_tokens; t++)
                acc[t] = fmaf(weight, h[(size_t)t * hidden_stride + i],
                              acc[t]);
        }
        for (int t = 0; t < n_tokens; t++)
            logits[(size_t)t * logits_stride + output_base + local_r + lane] =
                acc[t];
    }
}

/* A read-only mapping avoids copying the full head from page cache on every
 * token. If mapping is unavailable, retain the 8 MiB streamed path. Both use
 * the same per-row FMA order. */
static void head_logits(DSV4Model *m, const float *h, float *logits)
{
    const int hid = m->cfg.hidden;
    const int vocab = m->cfg.vocab;
    const int chunk = 1024;
    const int esz = 2;
    int use_simd = 0;
#if defined(__AVX2__) && defined(__FMA__)
    use_simd = getenv("DSV4_NO_SIMD") == NULL &&
               getenv("DSV4_NO_HEAD_SIMD") == NULL;
#endif
    if (m->head_map_rows) {
        const int row_blocks = (vocab + 7) / 8;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(m->threads)
#endif
        for (int block = 0; block < row_blocks; block++)
            head_logits_block(h, m->head_map_rows, logits, hid, 0, vocab, block,
                              use_simd);
        return;
    }

    uint8_t *buf = m->head_buf;
#ifdef _OPENMP
    #pragma omp parallel num_threads(m->threads) shared(buf, logits)
#endif
    {
        for (int r0 = 0; r0 < vocab; r0 += chunk) {
            int r1 = r0 + chunk;
            if (r1 > vocab) r1 = vocab;
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                int64_t off = m->head_t->off + (int64_t)r0 * hid * esz;
                int64_t want = (int64_t)(r1 - r0) * hid * esz;
                int64_t got = 0;
                while (got < want) {
                    ssize_t nr = pread(m->st.fd[m->head_t->shard], buf + got,
                                       (size_t)(want - got), (off_t)(off + got));
                    if (nr <= 0) { fprintf(stderr, "dsv4: short read head\n"); exit(1); }
                    got += nr;
                }
            }
            const uint16_t *rows = (const uint16_t *)buf;
            const int row_count = r1 - r0;
            const int row_blocks = (row_count + 7) / 8;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int block = 0; block < row_blocks; block++)
                head_logits_block(h, rows, logits, hid, r0, row_count, block,
                                  use_simd);
        }
    }
}

/* Batched vocabulary projection for speculative verification. The arithmetic
 * for each (token, row) is exactly head_logits_block; only the loop order is
 * changed so a mapped or streamed weight block serves every token while hot. */
static void head_logits_batch(DSV4Model *m, const float *h, int n_tokens,
                              float *logits)
{
    const int hid = m->cfg.hidden;
    const int vocab = m->cfg.vocab;
    const int chunk = 1024;
    const int esz = 2;
    int use_simd = 0;
#if defined(__AVX2__) && defined(__FMA__)
    use_simd = getenv("DSV4_NO_SIMD") == NULL &&
               getenv("DSV4_NO_HEAD_SIMD") == NULL;
#endif
    if (m->head_map_rows) {
        const int row_blocks = (vocab + 7) / 8;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(m->threads)
#endif
        for (int block = 0; block < row_blocks; block++) {
            for (int base = 0; base < n_tokens; base += DSV4_MAX_TOPK + 1) {
                int count = n_tokens - base;
                if (count > DSV4_MAX_TOPK + 1) count = DSV4_MAX_TOPK + 1;
                head_logits_block_batch(
                    h + (size_t)base * hid, hid, m->head_map_rows,
                    logits + (size_t)base * vocab, vocab, count, hid, 0,
                    vocab, block, use_simd);
            }
        }
        return;
    }

    uint8_t *buf = m->head_buf;
#ifdef _OPENMP
    #pragma omp parallel num_threads(m->threads) shared(buf, logits)
#endif
    {
        for (int r0 = 0; r0 < vocab; r0 += chunk) {
            int r1 = r0 + chunk;
            if (r1 > vocab) r1 = vocab;
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                int64_t off = m->head_t->off + (int64_t)r0 * hid * esz;
                int64_t want = (int64_t)(r1 - r0) * hid * esz;
                int64_t got = 0;
                while (got < want) {
                    ssize_t nr = pread(m->st.fd[m->head_t->shard], buf + got,
                                       (size_t)(want - got), (off_t)(off + got));
                    if (nr <= 0) { fprintf(stderr, "dsv4: short read head\n"); exit(1); }
                    got += nr;
                }
            }
            const uint16_t *rows = (const uint16_t *)buf;
            const int row_count = r1 - r0;
            const int row_blocks = (row_count + 7) / 8;
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int block = 0; block < row_blocks; block++) {
                for (int base = 0; base < n_tokens;
                     base += DSV4_MAX_TOPK + 1) {
                    int count = n_tokens - base;
                    if (count > DSV4_MAX_TOPK + 1)
                        count = DSV4_MAX_TOPK + 1;
                    head_logits_block_batch(
                        h + (size_t)base * hid, hid, rows,
                        logits + (size_t)base * vocab, vocab, count, hid, r0,
                        row_count, block, use_simd);
                }
            }
        }
    }
}

typedef struct {
    float *bin, *attn_out, *ffn_out, *post, *comb;
    DSV4RoutePrefetch route_prefetch;
} DSV4ForwardBuffers;

static void free_forward_buffers(DSV4ForwardBuffers *b)
{
    free(b->bin);
    free(b->attn_out);
    free(b->ffn_out);
    free(b->post);
    free(b->comb);
    memset(b, 0, sizeof(*b));
}

static int init_forward_buffers(DSV4ForwardBuffers *b, int hid, int mult)
{
    memset(b, 0, sizeof(*b));
    b->bin = (float *)malloc((size_t)hid * sizeof(float));
    b->attn_out = (float *)malloc((size_t)hid * sizeof(float));
    b->ffn_out = (float *)malloc((size_t)hid * sizeof(float));
    b->post = (float *)malloc((size_t)mult * sizeof(float));
    b->comb = (float *)malloc((size_t)mult * mult * sizeof(float));
    if (!b->bin || !b->attn_out || !b->ffn_out || !b->post || !b->comb) {
        fprintf(stderr, "dsv4: OOM forward buffers\n");
        free_forward_buffers(b);
        return 0;
    }
    return 1;
}

static int embed_token(DSV4Model *m, int token_id, int position, float *state)
{
    const int hid = m->cfg.hidden;
    const int mult = m->cfg.hc_mult;
    uint16_t *row = (uint16_t *)malloc((size_t)hid * sizeof(*row));
    if (!row) { fprintf(stderr, "dsv4: OOM embedding row\n"); return -1; }

    int64_t off = m->embed_t->off + (int64_t)token_id * hid * 2;
    int64_t got = 0;
    while (got < hid * 2) {
        ssize_t r = pread(m->st.fd[m->embed_t->shard], (char *)row + got,
                          (size_t)(hid * 2 - got), (off_t)(off + got));
        if (r <= 0) {
            fprintf(stderr, "dsv4: short read embed\n");
            free(row);
            return -1;
        }
        got += r;
    }
    for (int i = 0; i < hid; i++) {
        float v = bf16f(row[i]);
        for (int k = 0; k < mult; k++) state[(size_t)k * hid + i] = v;
    }
    free(row);

    if (getenv("DSV4_DEBUG_L0")) {
        double es = 0.0;
        for (int i = 0; i < mult * hid; i++)
            es += (double)state[i] * (double)state[i];
        fprintf(stderr, "L0 pos=%d stateL2(pre)=%.7f\n", position, sqrt(es));
    }
    return 0;
}

static void forward_loaded_layer(DSV4Model *m, const DSV4LayerW *w, int layer,
                                  int token_id, int position, float *state,
                                  DSV4ForwardBuffers *b)
{
    const int hid = m->cfg.hidden;
    const int mult = m->cfg.hc_mult;
    DSV4LayerRun *r = &m->run[layer];
    double t0 = now_s(), t1, t2, t3;

    m->cur_layer = layer;
    hc_pre(m, w, state, w->hc_attn_fn, w->hc_attn_scale, w->hc_attn_base,
           b->bin, b->post, b->comb);
    t1 = now_s();
    rmsnorm_bf16(b->bin, b->bin, w->attn_norm, hid, m->cfg.rms_eps, 1);
    route_prediction_prepare(m, w, b->bin, token_id, &b->route_prefetch);
    attention_forward(m, w, r, b->bin, position, b->attn_out);
    context_snapshot_capture_layer(m, layer, position);
    t2 = now_s();
    if (getenv("DSV4_DEBUG_L0") && layer == 0) {
        double as = 0.0;
        for (int i = 0; i < hid; i++) as += (double)b->attn_out[i] * (double)b->attn_out[i];
        fprintf(stderr, "L0 pos=%d attn_outL2=%.7f post=", position, sqrt(as));
        for (int i = 0; i < mult; i++) fprintf(stderr, "%.7f ", b->post[i]);
        fprintf(stderr, "comb=");
        for (int i = 0; i < mult * mult; i++) fprintf(stderr, "%.7f ", b->comb[i]);
        fprintf(stderr, "\n");
    }
    hc_post(m, state, b->attn_out, b->post, b->comb, mult, hid);
    t3 = now_s();
    if (getenv("DSV4_DEBUG_TIME")) {
        fprintf(stderr, "T pos=%d L=%d hcpre=%.3f attn=%.3f hcpost=%.3f",
                position, layer, t1 - t0, t2 - t1, t3 - t2);
    }
    if (getenv("DSV4_DEBUG_L2")) {
        double ss = 0.0;
        for (int i = 0; i < mult * hid; i++) ss += (double)state[i] * (double)state[i];
        fprintf(stderr, "L2 pos=%d L=%d attn %.7f\n", position, layer, sqrt(ss));
    }

    hc_pre(m, w, state, w->hc_ffn_fn, w->hc_ffn_scale, w->hc_ffn_base,
           b->bin, b->post, b->comb);
    double t4 = now_s();
    rmsnorm_bf16(b->bin, b->bin, w->ffn_norm, hid, m->cfg.rms_eps, 1);
    route_prefetch_finish(&b->route_prefetch);
    moe_forward(m, w, b->bin, token_id, position, &b->route_prefetch,
                b->ffn_out);
    double t5 = now_s();
    hc_post(m, state, b->ffn_out, b->post, b->comb, mult, hid);
    double t6 = now_s();
    if (m->profiling) {
        m->time_attn += t2 - t1;
        m->time_moe += t5 - t4;
        m->time_hc += (t1 - t0) + (t3 - t2) + (t4 - t3) + (t6 - t5);
    }
    if (getenv("DSV4_DEBUG_TIME")) {
        fprintf(stderr, " hcpre2=%.3f moe=%.3f hcpost2=%.3f\n",
                t4 - t3, t5 - t4, t6 - t5);
    }
    if (getenv("DSV4_DEBUG_L2")) {
        double ss = 0.0;
        for (int i = 0; i < mult * hid; i++) ss += (double)state[i] * (double)state[i];
        fprintf(stderr, "L2 pos=%d L=%d ffn  %.7f\n", position, layer, sqrt(ss));
    }
}

static void forward_loaded_layer_batch(DSV4Model *m, const DSV4LayerW *w,
                                       int layer, const int *token_ids,
                                       int start_position, float *states,
                                       int n_tokens, DSV4ForwardBuffers *b)
{
    if (n_tokens <= 1 || getenv("DSV4_NO_BATCH_MOE")) {
        const size_t stride = (size_t)m->cfg.hc_mult * m->cfg.hidden;
        for (int t = 0; t < n_tokens; t++)
            forward_loaded_layer(m, w, layer, token_ids[t], start_position + t,
                                 states + (size_t)t * stride, b);
        return;
    }
    const int hid = m->cfg.hidden;
    const int mult = m->cfg.hc_mult;
    const size_t state_stride = (size_t)mult * hid;
    float *bins = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *ffn = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *post = (float *)malloc((size_t)n_tokens * mult * sizeof(float));
    float *comb = (float *)malloc((size_t)n_tokens * mult * mult * sizeof(float));
    const int batch_attention = getenv("DSV4_NO_BATCH_ATTN_PROJ") == NULL;
    const int parallel_hc = batch_attention && n_tokens >= 4 &&
                            getenv("DSV4_NO_BATCH_HC_PARALLEL") == NULL;
    const int mix_n = (2 + mult) * mult;
    float *hc_mixes = parallel_hc
        ? (float *)malloc((size_t)n_tokens * mix_n * sizeof(float)) : NULL;
    float *hc_states = parallel_hc
        ? (float *)malloc((size_t)n_tokens * state_stride * sizeof(float)) : NULL;
    float *attn_bins = batch_attention
        ? (float *)malloc((size_t)n_tokens * hid * sizeof(float)) : NULL;
    float *attn = batch_attention
        ? (float *)malloc((size_t)n_tokens * hid * sizeof(float)) : NULL;
    float *attn_post = batch_attention
        ? (float *)malloc((size_t)n_tokens * mult * sizeof(float)) : NULL;
    float *attn_comb = batch_attention
        ? (float *)malloc((size_t)n_tokens * mult * mult * sizeof(float)) : NULL;
    if (!bins || !ffn || !post || !comb ||
        (parallel_hc && (!hc_mixes || !hc_states)) ||
        (batch_attention && (!attn_bins || !attn || !attn_post ||
                             !attn_comb))) {
        fprintf(stderr, "dsv4: OOM layer batch\n");
        exit(1);
    }

    m->cur_layer = layer;
    int predicted_routes[DSV4_MOE_BATCH_TOKENS * DSV4_MAX_TOPK];
    unsigned char predicted_misses[DSV4_MOE_BATCH_TOKENS * DSV4_MAX_TOPK];
    int predicted_count = 0;
    DSV4RoutePrefetch batch_route_prefetch;
    memset(&batch_route_prefetch, 0, sizeof(batch_route_prefetch));
    if (batch_attention) {
        double hc_start = m->profiling ? now_s() : 0.0;
        if (parallel_hc) {
            #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
            for (int t = 0; t < n_tokens; t++) {
                float *state = states + (size_t)t * state_stride;
                hc_pre_scratch(
                    m, w, state, w->hc_attn_fn, w->hc_attn_scale,
                    w->hc_attn_base, attn_bins + (size_t)t * hid,
                    attn_post + (size_t)t * mult,
                    attn_comb + (size_t)t * mult * mult,
                    hc_mixes + (size_t)t * mix_n);
                rmsnorm_bf16(attn_bins + (size_t)t * hid,
                             attn_bins + (size_t)t * hid, w->attn_norm, hid,
                             m->cfg.rms_eps, 1);
            }
        } else {
            for (int t = 0; t < n_tokens; t++) {
                float *state = states + (size_t)t * state_stride;
                hc_pre(m, w, state, w->hc_attn_fn, w->hc_attn_scale,
                       w->hc_attn_base, attn_bins + (size_t)t * hid,
                       attn_post + (size_t)t * mult,
                       attn_comb + (size_t)t * mult * mult);
                rmsnorm_bf16(attn_bins + (size_t)t * hid,
                             attn_bins + (size_t)t * hid, w->attn_norm, hid,
                             m->cfg.rms_eps, 1);
            }
        }
        if (m->profiling) m->time_hc += now_s() - hc_start;
        predicted_count = route_prediction_observe_batch(
            m, w, attn_bins, token_ids, n_tokens, predicted_routes,
            predicted_misses);
        route_prefetch_begin_batch(
            m, w, predicted_routes, predicted_misses, predicted_count,
            n_tokens, &batch_route_prefetch);
        double attn_start = m->profiling ? now_s() : 0.0;
        attention_forward_batch_projected(m, w, &m->run[layer], attn_bins,
                                          start_position, n_tokens, attn);
        if (m->profiling) m->time_attn += now_s() - attn_start;
        hc_start = m->profiling ? now_s() : 0.0;
        if (parallel_hc) {
            #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
            for (int t = 0; t < n_tokens; t++) {
                float *state = states + (size_t)t * state_stride;
                hc_post_scratch(
                    state, attn + (size_t)t * hid,
                    attn_post + (size_t)t * mult,
                    attn_comb + (size_t)t * mult * mult, mult, hid,
                    hc_states + (size_t)t * state_stride);
                hc_pre_scratch(
                    m, w, state, w->hc_ffn_fn, w->hc_ffn_scale,
                    w->hc_ffn_base, bins + (size_t)t * hid,
                    post + (size_t)t * mult,
                    comb + (size_t)t * mult * mult,
                    hc_mixes + (size_t)t * mix_n);
                rmsnorm_bf16(bins + (size_t)t * hid,
                             bins + (size_t)t * hid, w->ffn_norm, hid,
                             m->cfg.rms_eps, 1);
            }
        } else {
            for (int t = 0; t < n_tokens; t++) {
                float *state = states + (size_t)t * state_stride;
                hc_post(m, state, attn + (size_t)t * hid,
                        attn_post + (size_t)t * mult,
                        attn_comb + (size_t)t * mult * mult, mult, hid);
                hc_pre(m, w, state, w->hc_ffn_fn, w->hc_ffn_scale,
                       w->hc_ffn_base, bins + (size_t)t * hid,
                       post + (size_t)t * mult,
                       comb + (size_t)t * mult * mult);
                rmsnorm_bf16(bins + (size_t)t * hid,
                             bins + (size_t)t * hid, w->ffn_norm, hid,
                             m->cfg.rms_eps, 1);
            }
        }
        if (m->profiling) m->time_hc += now_s() - hc_start;
    } else {
        for (int t = 0; t < n_tokens; t++) {
            double t0 = m->profiling ? now_s() : 0.0;
            float *state = states + (size_t)t * state_stride;
            hc_pre(m, w, state, w->hc_attn_fn, w->hc_attn_scale,
                   w->hc_attn_base, b->bin, b->post, b->comb);
            double t1 = m->profiling ? now_s() : 0.0;
            rmsnorm_bf16(b->bin, b->bin, w->attn_norm, hid,
                         m->cfg.rms_eps, 1);
            attention_forward(m, w, &m->run[layer], b->bin,
                              start_position + t, b->attn_out);
            context_snapshot_capture_layer(m, layer, start_position + t);
            double t2 = m->profiling ? now_s() : 0.0;
            hc_post(m, state, b->attn_out, b->post, b->comb, mult, hid);
            hc_pre(m, w, state, w->hc_ffn_fn, w->hc_ffn_scale,
                   w->hc_ffn_base, bins + (size_t)t * hid,
                   post + (size_t)t * mult,
                   comb + (size_t)t * mult * mult);
            rmsnorm_bf16(bins + (size_t)t * hid,
                         bins + (size_t)t * hid, w->ffn_norm, hid,
                         m->cfg.rms_eps, 1);
            if (m->profiling) {
                double t3 = now_s();
                m->time_attn += t2 - t1;
                m->time_hc += (t1 - t0) + (t3 - t2);
            }
        }
    }
    double moe_start = m->profiling ? now_s() : 0.0;
    moe_forward_batch(m, w, bins, token_ids, start_position, n_tokens,
                      predicted_routes, predicted_misses, predicted_count,
                      &batch_route_prefetch, ffn);
    if (m->profiling) m->time_moe += now_s() - moe_start;
    double post_start = m->profiling ? now_s() : 0.0;
    if (parallel_hc) {
        #pragma omp parallel for schedule(static) num_threads(dsv4_gemv_threads())
        for (int t = 0; t < n_tokens; t++)
            hc_post_scratch(
                states + (size_t)t * state_stride,
                ffn + (size_t)t * hid, post + (size_t)t * mult,
                comb + (size_t)t * mult * mult, mult, hid,
                hc_states + (size_t)t * state_stride);
    } else {
        for (int t = 0; t < n_tokens; t++)
            hc_post(m, states + (size_t)t * state_stride,
                    ffn + (size_t)t * hid, post + (size_t)t * mult,
                    comb + (size_t)t * mult * mult, mult, hid);
    }
    if (m->profiling) m->time_hc += now_s() - post_start;
    free(bins); free(ffn); free(post); free(comb);
    free(attn_bins); free(attn); free(attn_post); free(attn_comb);
    free(hc_mixes); free(hc_states);
}

static int compute_logits(DSV4Model *m, const float *state, int position,
                          float *logits)
{
    double head_start = m->profiling ? now_s() : 0.0;
    float *h = m->head_hidden;
    model_head(m, state, h);
    head_logits(m, h, logits);

    if (getenv("DSV4_DEBUG_LOGITS")) {
        int top[10];
        for (int k = 0; k < 10; k++) top[k] = -1;
        for (int v = 0; v < m->cfg.vocab; v++) {
            for (int k = 0; k < 10; k++) {
                if (top[k] < 0 || logits[v] > logits[top[k]]) {
                    for (int j = 9; j > k; j--) top[j] = top[j - 1];
                    top[k] = v;
                    break;
                }
            }
        }
        fprintf(stderr, "top10 pos=%d:", position);
        for (int k = 0; k < 10; k++)
            fprintf(stderr, " %d:%.7f", top[k], logits[top[k]]);
        fprintf(stderr, "\n");
    }
    if (m->profiling) m->time_head += now_s() - head_start;
    return 0;
}

static int compute_logits_batch(DSV4Model *m, const float *states, int n_tokens,
                                float *logits)
{
    const int hid = m->cfg.hidden;
    const int mult = m->cfg.hc_mult;
    const size_t state_elems = (size_t)mult * hid;
    if ((size_t)n_tokens > SIZE_MAX / (size_t)hid / sizeof(float)) return -1;
    double head_start = m->profiling ? now_s() : 0.0;
    float *hidden = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    if (!hidden) { fprintf(stderr, "dsv4: OOM batched head states\n"); return -1; }
    for (int i = 0; i < n_tokens; i++)
        model_head(m, states + (size_t)i * state_elems,
                   hidden + (size_t)i * hid);
    head_logits_batch(m, hidden, n_tokens, logits);
    free(hidden);
    if (m->profiling) m->time_head += now_s() - head_start;
    return 0;
}

/* DSpark uses one non-causal draft block: all block positions see the same
 * target-hidden prefix (strictly before anchor_position) and every position in
 * the current block. The released implementation exposes up to window target
 * rows in addition to every row in the current draft block. */
static int dspark_attention_block(DSV4Model *m, const DSV4LayerW *w, int stage,
                                  int anchor_position, const float *x,
                                  int n_tokens, float *out)
{
    const DSV4Config *c = &m->cfg;
    const int hid = c->hidden, H = c->n_heads, d = c->head_dim;
    const int rd = c->rope_dim, ql = c->q_lora;
    const int xsc = (hid + 127) / 128;
    const int qrsc = (ql + 127) / 128;
    const int qwidth = H * d;
    const int G = c->o_groups, ol = c->o_lora;
    const int oall_width = G * ol;
    const int osc_width = (oall_width + 127) / 128;
    const float scale = 1.0f / sqrtf((float)d);
    const int batch_projections =
        getenv("DSV4_NO_BATCH_DSPARK_ATTN") == NULL;

    float *q = (float *)malloc((size_t)n_tokens * qwidth * sizeof(float));
    float *block_kv = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    float *xq = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *xqs = (float *)malloc((size_t)n_tokens * xsc * sizeof(float));
    float *qr = (float *)malloc((size_t)n_tokens * ql * sizeof(float));
    float *qrn = (float *)malloc((size_t)n_tokens * ql * sizeof(float));
    float *qrs = (float *)malloc((size_t)n_tokens * qrsc * sizeof(float));
    float *kv = (float *)malloc((size_t)n_tokens * d * sizeof(float));
    float *raw = (float *)malloc((size_t)n_tokens * qwidth * sizeof(float));
    float *oall = (float *)malloc((size_t)n_tokens * oall_width * sizeof(float));
    float *osc = (float *)malloc((size_t)n_tokens * osc_width * sizeof(float));
    if (!q || !block_kv || !xq || !xqs || !qr || !qrn || !qrs || !kv ||
        !raw || !oall || !osc) {
        free(q); free(block_kv); free(xq); free(xqs); free(qr); free(qrn);
        free(qrs); free(kv); free(raw); free(oall); free(osc);
        return -1;
    }

    for (int t = 0; t < n_tokens; t++) {
        const float *xt = x + (size_t)t * hid;
        dsv4_quant_fp8_codes(xq + (size_t)t * hid,
                             xqs + (size_t)t * xsc, xt, hid);
    }
    for (int base = 0; base < n_tokens; base += DSV4_MAX_TOPK + 1) {
        int count = n_tokens - base;
        if (count > DSV4_MAX_TOPK + 1) count = DSV4_MAX_TOPK + 1;
        float *outputs[DSV4_MAX_TOPK + 1];
        const float *inputs[DSV4_MAX_TOPK + 1];
        const float *scales[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < count; t++) {
            outputs[t] = qr + (size_t)(base + t) * ql;
            inputs[t] = xq + (size_t)(base + t) * hid;
            scales[t] = xqs + (size_t)(base + t) * xsc;
        }
        if (batch_projections)
            dsv4_gemv_fp8_batch_q(outputs, inputs, scales, count, w->wq_a,
                                  w->wq_a_scale, hid, ql, 1);
        else
            for (int t = 0; t < count; t++)
                dsv4_gemv_fp8_q(outputs[t], inputs[t], scales[t], w->wq_a,
                                w->wq_a_scale, hid, ql, 1);
    }
    for (int t = 0; t < n_tokens; t++) {
        float *qrnt = qrn + (size_t)t * ql;
        rmsnorm_bf16(qrnt, qr + (size_t)t * ql, w->q_norm, ql,
                     c->rms_eps, 1);
        dsv4_quant_fp8_codes(qrnt, qrs + (size_t)t * qrsc, qrnt, ql);
    }
    for (int base = 0; base < n_tokens; base += DSV4_MAX_TOPK + 1) {
        int count = n_tokens - base;
        if (count > DSV4_MAX_TOPK + 1) count = DSV4_MAX_TOPK + 1;
        float *q_out[DSV4_MAX_TOPK + 1], *kv_out[DSV4_MAX_TOPK + 1];
        const float *q_in[DSV4_MAX_TOPK + 1];
        const float *q_scale[DSV4_MAX_TOPK + 1];
        const float *x_in[DSV4_MAX_TOPK + 1];
        const float *x_scale[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < count; t++) {
            int token = base + t;
            q_out[t] = q + (size_t)token * qwidth;
            kv_out[t] = kv + (size_t)token * d;
            q_in[t] = qrn + (size_t)token * ql;
            q_scale[t] = qrs + (size_t)token * qrsc;
            x_in[t] = xq + (size_t)token * hid;
            x_scale[t] = xqs + (size_t)token * xsc;
        }
        if (batch_projections) {
            dsv4_gemv_fp8_batch_q(q_out, q_in, q_scale, count, w->wq_b,
                                  w->wq_b_scale, ql, qwidth, 1);
            dsv4_gemv_fp8_batch_q(kv_out, x_in, x_scale, count, w->wkv,
                                  w->wkv_scale, hid, d, 1);
        } else {
            for (int t = 0; t < count; t++) {
                dsv4_gemv_fp8_q(q_out[t], q_in[t], q_scale[t], w->wq_b,
                                w->wq_b_scale, ql, qwidth, 1);
                dsv4_gemv_fp8_q(kv_out[t], x_in[t], x_scale[t], w->wkv,
                                w->wkv_scale, hid, d, 1);
            }
        }
    }

    for (int t = 0; t < n_tokens; t++) {
        const int pos = anchor_position + t;
        float *qt = q + (size_t)t * qwidth;
        float *kvt = block_kv + (size_t)t * d;
        for (int h = 0; h < H; h++) {
            float *qh = qt + (size_t)h * d;
            double ss = 0.0;
            for (int i = 0; i < d; i++) ss += (double)qh[i] * qh[i];
            float r = 1.0f / sqrtf((float)(ss / d) + c->rms_eps);
            for (int i = 0; i < d; i++) qh[i] = f32bf(qh[i] * r);
        }
        DSV4RopeFreq rope;
        dsv4_rope_freqs(&rope, rd, c->rope_theta, 0, c->rope_factor,
                        c->beta_fast, c->beta_slow, pos, c->original_position);
        dsv4_rope_apply_buf(qt, H, d, &rope);

        rmsnorm_bf16(kvt, kv + (size_t)t * d, w->kv_norm, d,
                     c->rms_eps, 1);
        dsv4_rope_apply_buf(kvt, 1, d, &rope);
        dsv4_act_quant_inplace(kvt, d - rd, DSV4_ACT_GROUP, 0);
        free(rope.cosv);
        free(rope.sinv);
    }

    int context_cap = c->window;
    int context_start = anchor_position - context_cap;
    if (context_start < 0) context_start = 0;
    int context_count = 0;
    for (int pos = context_start; pos < anchor_position; pos++)
        if (m->dspark->target_position[pos % c->window] == pos) context_count++;
    const int total = context_count + n_tokens;
    float *all_kv = (float *)malloc((size_t)(total ? total : 1) * d * sizeof(float));
    int *indices = (int *)malloc((size_t)(total ? total : 1) * sizeof(int));
    if (!all_kv || !indices || !raw || !oall || !osc) {
        free(q); free(block_kv); free(xq); free(xqs); free(qr); free(qrn);
        free(qrs); free(kv); free(all_kv); free(indices); free(raw); free(oall);
        free(osc);
        return -1;
    }
    const float *target = m->dspark->target_kv +
        (size_t)stage * c->window * d;
    int at = 0;
    for (int pos = context_start; pos < anchor_position; pos++) {
        int slot = pos % c->window;
        if (m->dspark->target_position[slot] != pos) continue;
        memcpy(all_kv + (size_t)at * d, target + (size_t)slot * d,
               (size_t)d * sizeof(float));
        indices[at] = at;
        at++;
    }
    memcpy(all_kv + (size_t)at * d, block_kv,
           (size_t)n_tokens * d * sizeof(float));
    for (int i = 0; i < n_tokens; i++) indices[at + i] = at + i;

    const int gdim = H * d / G;
    for (int t = 0; t < n_tokens; t++) {
        const int pos = anchor_position + t;
        float *qt = q + (size_t)t * qwidth;
        float *ot = raw + (size_t)t * qwidth;
        attention_scratch_reset(m);
        sparse_attn(m, ot, qt, all_kv, w->attn_sink, indices, total,
                    H, d, scale);
        DSV4RopeFreq rope;
        dsv4_rope_freqs(&rope, rd, c->rope_theta, 0, c->rope_factor,
                        c->beta_fast, c->beta_slow, pos, c->original_position);
        dsv4_rope_apply_buf_inv(ot, H, d, &rope);
        free(rope.cosv);
        free(rope.sinv);
    }
    for (int base = 0; base < n_tokens; base += DSV4_MAX_TOPK + 1) {
        int count = n_tokens - base;
        if (count > DSV4_MAX_TOPK + 1) count = DSV4_MAX_TOPK + 1;
        float *oall_out[DSV4_MAX_TOPK + 1];
        const float *raw_in[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < count; t++) {
            int token = base + t;
            oall_out[t] = oall + (size_t)token * oall_width;
            raw_in[t] = raw + (size_t)token * qwidth;
        }
        if (batch_projections) {
            if (w->wo_a_codes)
                gemv_packed_wo_a_grouped_batch(
                    oall_out, raw_in, count, w->wo_a_codes, w->wo_a_scale,
                    G, gdim, ol);
            else
                gemv_bf16_grouped_batch(oall_out, raw_in, count, w->wo_a,
                                        G, gdim, ol);
        } else {
            for (int t = 0; t < count; t++) {
                if (w->wo_a_codes) {
                    float *one_out[1] = { oall_out[t] };
                    const float *one_in[1] = { raw_in[t] };
                    gemv_packed_wo_a_grouped_batch(
                        one_out, one_in, 1, w->wo_a_codes, w->wo_a_scale,
                        G, gdim, ol);
                } else {
                    gemv_bf16_grouped(oall_out[t], raw_in[t], w->wo_a,
                                      G, gdim, ol);
                }
            }
        }
    }
    for (int t = 0; t < n_tokens; t++) {
        float *oallt = oall + (size_t)t * oall_width;
        dsv4_quant_fp8_codes(oallt, osc + (size_t)t * osc_width,
                             oallt, oall_width);
    }
    for (int base = 0; base < n_tokens; base += DSV4_MAX_TOPK + 1) {
        int count = n_tokens - base;
        if (count > DSV4_MAX_TOPK + 1) count = DSV4_MAX_TOPK + 1;
        float *outputs[DSV4_MAX_TOPK + 1];
        const float *inputs[DSV4_MAX_TOPK + 1];
        const float *scales[DSV4_MAX_TOPK + 1];
        for (int t = 0; t < count; t++) {
            int token = base + t;
            outputs[t] = out + (size_t)token * hid;
            inputs[t] = oall + (size_t)token * oall_width;
            scales[t] = osc + (size_t)token * osc_width;
        }
        if (batch_projections)
            dsv4_gemv_fp8_batch_q(outputs, inputs, scales, count, w->wo_b,
                                  w->wo_b_scale, oall_width, hid, 1);
        else
            for (int t = 0; t < count; t++)
                dsv4_gemv_fp8_q(outputs[t], inputs[t], scales[t], w->wo_b,
                                w->wo_b_scale, oall_width, hid, 1);
    }

    free(q); free(block_kv); free(xq); free(xqs); free(qr); free(qrn);
    free(qrs); free(kv); free(all_kv); free(indices); free(raw); free(oall);
    free(osc);
    return 0;
}

static int dspark_stage_forward(DSV4Model *m, const DSV4LayerW *w, int stage,
                                const int *token_ids, int anchor_position,
                                float *states, int n_tokens)
{
    const int hid = m->cfg.hidden, mult = m->cfg.hc_mult;
    const size_t state_stride = (size_t)mult * hid;
    float *bins = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *attn = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *ffn = (float *)malloc((size_t)n_tokens * hid * sizeof(float));
    float *post = (float *)malloc((size_t)n_tokens * mult * sizeof(float));
    float *comb = (float *)malloc((size_t)n_tokens * mult * mult * sizeof(float));
    if (!bins || !attn || !ffn || !post || !comb) {
        free(bins); free(attn); free(ffn); free(post); free(comb);
        return -1;
    }
    m->cur_layer = m->cfg.n_layers + stage;
    for (int t = 0; t < n_tokens; t++) {
        float *bin = bins + (size_t)t * hid;
        hc_pre(m, w, states + (size_t)t * state_stride,
               w->hc_attn_fn, w->hc_attn_scale, w->hc_attn_base, bin,
               post + (size_t)t * mult, comb + (size_t)t * mult * mult);
        rmsnorm_bf16(bin, bin, w->attn_norm, hid, m->cfg.rms_eps, 1);
    }
    if (dspark_attention_block(m, w, stage, anchor_position, bins,
                               n_tokens, attn) != 0) {
        free(bins); free(attn); free(ffn); free(post); free(comb);
        return -1;
    }
    for (int t = 0; t < n_tokens; t++) {
        float *state = states + (size_t)t * state_stride;
        hc_post(m, state, attn + (size_t)t * hid,
                post + (size_t)t * mult, comb + (size_t)t * mult * mult,
                mult, hid);
        hc_pre(m, w, state, w->hc_ffn_fn, w->hc_ffn_scale, w->hc_ffn_base,
               bins + (size_t)t * hid, post + (size_t)t * mult,
               comb + (size_t)t * mult * mult);
        rmsnorm_bf16(bins + (size_t)t * hid, bins + (size_t)t * hid,
                     w->ffn_norm, hid, m->cfg.rms_eps, 1);
    }
    moe_forward_batch(m, w, bins, token_ids, anchor_position, n_tokens,
                      NULL, NULL, 0, NULL, ffn);
    for (int t = 0; t < n_tokens; t++)
        hc_post(m, states + (size_t)t * state_stride,
                ffn + (size_t)t * hid, post + (size_t)t * mult,
                comb + (size_t)t * mult * mult, mult, hid);
    free(bins); free(attn); free(ffn); free(post); free(comb);
    return 0;
}

static void dspark_head_hidden(DSV4Model *m, const float *state, float *h,
                              float *pre_norm)
{
    const int mult = m->cfg.hc_mult, hid = m->cfg.hidden;
    const DSV4DSpark *d = m->dspark;
    double ss = 0.0;
    for (int i = 0; i < mult * hid; i++) ss += (double)state[i] * state[i];
    float rsqrt = 1.0f / sqrtf((float)(ss / (mult * hid)) + m->cfg.rms_eps);
    float pre[8];
    for (int lane = 0; lane < mult; lane++) {
        const float *row = d->hc_head_fn + (size_t)lane * mult * hid;
        float acc = 0.0f;
        for (int i = 0; i < mult * hid; i++) acc = fmaf(row[i], state[i], acc);
        float mix = acc * rsqrt;
        pre[lane] = 1.0f /
            (1.0f + expf(-(mix * d->hc_head_scale[0] + d->hc_head_base[lane]))) +
            m->cfg.hc_eps;
    }
    for (int i = 0; i < hid; i++) {
        float acc = 0.0f;
        for (int lane = 0; lane < mult; lane++)
            acc = fmaf(pre[lane], state[(size_t)lane * hid + i], acc);
        h[i] = f32bf(acc);
    }
    if (pre_norm) memcpy(pre_norm, h, (size_t)hid * sizeof(float));
    rmsnorm_bf16(h, h, d->final_norm, hid, m->cfg.rms_eps, 1);
}

static const uint16_t *mapped_bf16_tensor(const DSV4Model *m,
                                           const K3Tensor *t)
{
    if (!m->layer_shard_map || t->shard < 0 || t->shard >= m->st.nshard ||
        t->off < 0 || (uint64_t)t->off + (uint64_t)t->nbytes >
                      m->layer_shard_map_len[t->shard])
        return NULL;
    return (const uint16_t *)((const uint8_t *)m->layer_shard_map[t->shard] +
                              t->off);
}

static int pread_tensor_bytes(const DSV4Model *m, const K3Tensor *t,
                              int64_t byte_offset, void *buf, int64_t nbytes)
{
    int64_t got = 0;
    while (got < nbytes) {
        ssize_t nr = pread(m->st.fd[t->shard], (uint8_t *)buf + got,
                           (size_t)(nbytes - got),
                           (off_t)(t->off + byte_offset + got));
        if (nr <= 0) return -1;
        got += nr;
    }
    return 0;
}

static int dspark_confidence_score(DSV4Model *m, const float *head_hidden,
                                   int previous_token, float *raw_out,
                                   float *prob_out)
{
    const int hid = m->cfg.hidden;
    const int rank = m->cfg.dspark_markov_rank;
    const K3Tensor *proj_t = m->dspark->confidence_proj;
    const K3Tensor *w1_t = m->dspark->markov_w1;
    const uint16_t *proj = mapped_bf16_tensor(m, proj_t);
    const uint16_t *w1_all = mapped_bf16_tensor(m, w1_t);
    uint16_t *owned_proj = NULL;
    uint16_t *owned_w1 = NULL;
    const uint16_t *w1;

    if (!proj) {
        owned_proj = (uint16_t *)malloc((size_t)(hid + rank) * sizeof(uint16_t));
        if (!owned_proj || pread_tensor_bytes(m, proj_t, 0, owned_proj,
                (int64_t)(hid + rank) * 2) != 0) {
            free(owned_proj);
            return -1;
        }
        proj = owned_proj;
    }
    if (w1_all) {
        w1 = w1_all + (size_t)previous_token * rank;
    } else {
        owned_w1 = (uint16_t *)malloc((size_t)rank * sizeof(uint16_t));
        if (!owned_w1 || pread_tensor_bytes(m, w1_t,
                (int64_t)previous_token * rank * 2, owned_w1,
                (int64_t)rank * 2) != 0) {
            free(owned_proj);
            free(owned_w1);
            return -1;
        }
        w1 = owned_w1;
    }

    float raw = 0.0f;
    for (int i = 0; i < hid; i++)
        raw = fmaf(bf16f(proj[i]), head_hidden[i], raw);
    for (int i = 0; i < rank; i++)
        raw = fmaf(bf16f(proj[hid + i]), bf16f(w1[i]), raw);
    free(owned_proj);
    free(owned_w1);
    *raw_out = raw;
    *prob_out = 1.0f / (1.0f + expf(-raw));
    return 0;
}

static int dspark_markov_argmax(DSV4Model *m, float *base_logits,
                                int previous_token)
{
    const int vocab = m->cfg.vocab;
    const int rank = m->cfg.dspark_markov_rank;
    const K3Tensor *w1t = m->dspark->markov_w1;
    const K3Tensor *w2t = m->dspark->markov_w2;
    const uint16_t *w1_all = mapped_bf16_tensor(m, w1t);
    const uint16_t *w2_all = mapped_bf16_tensor(m, w2t);
    uint16_t *owned_w1 = NULL;
    const uint16_t *w1;
    if (w1_all) {
        w1 = w1_all + (size_t)previous_token * rank;
    } else {
        owned_w1 = (uint16_t *)malloc((size_t)rank * sizeof(uint16_t));
        if (!owned_w1 || pread_tensor_bytes(m, w1t,
                (int64_t)previous_token * rank * 2, owned_w1,
                (int64_t)rank * 2) != 0) {
            free(owned_w1);
            return -1;
        }
        w1 = owned_w1;
    }

    if (w2_all) {
        if (getenv("DSV4_DSPARK_MARKOV_SIMD")) {
            float *bias = (float *)malloc((size_t)vocab * sizeof(float));
            if (!bias) {
                free(owned_w1);
                return -1;
            }
            float input[256];
            for (int i = 0; i < rank; i++) input[i] = bf16f(w1[i]);
            dsv4_gemv_bf16(bias, input, w2_all, rank, vocab, 0);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(m->threads)
#endif
            for (int v = 0; v < vocab; v++) base_logits[v] += bias[v];
            free(bias);
        } else {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static) num_threads(m->threads)
#endif
        for (int v = 0; v < vocab; v++) {
            const uint16_t *row = w2_all + (size_t)v * rank;
            float bias = 0.0f;
            for (int i = 0; i < rank; i++)
                bias = fmaf(bf16f(row[i]), bf16f(w1[i]), bias);
            base_logits[v] += bias;
        }
        }
    } else {
        const int chunk = 1024;
        uint16_t *rows = (uint16_t *)malloc((size_t)chunk * rank *
                                             sizeof(uint16_t));
        if (!rows) {
            free(owned_w1);
            return -1;
        }
        for (int r0 = 0; r0 < vocab; r0 += chunk) {
            int nr = vocab - r0;
            if (nr > chunk) nr = chunk;
            if (pread_tensor_bytes(m, w2t, (int64_t)r0 * rank * 2, rows,
                                   (int64_t)nr * rank * 2) != 0) {
                free(rows);
                free(owned_w1);
                return -1;
            }
#ifdef _OPENMP
            #pragma omp parallel for schedule(static) num_threads(m->threads)
#endif
            for (int r = 0; r < nr; r++) {
                const uint16_t *row = rows + (size_t)r * rank;
                float bias = 0.0f;
                for (int i = 0; i < rank; i++)
                    bias = fmaf(bf16f(row[i]), bf16f(w1[i]), bias);
                base_logits[r0 + r] += bias;
            }
        }
        free(rows);
    }
    free(owned_w1);
    int best = 0;
    for (int v = 1; v < vocab; v++)
        if (base_logits[v] > base_logits[best]) best = v;
    return best;
}

int dsv4_dspark_propose(DSV4Model *m, int anchor_token, int anchor_position,
                        int n_drafts, int *draft_tokens)
{
    if (!m || !m->dspark || !draft_tokens || anchor_token < 0 ||
        anchor_token >= m->cfg.vocab || anchor_position < 0 ||
        n_drafts < 1 || n_drafts > m->cfg.dspark_block_size ||
        anchor_position > m->context - n_drafts)
        return -1;
    const int n = n_drafts;
    const int hid = m->cfg.hidden, mult = m->cfg.hc_mult;
    const size_t state_stride = (size_t)mult * hid;
    int token_ids[32];
    token_ids[0] = anchor_token;
    for (int t = 1; t < n; t++) token_ids[t] = m->cfg.dspark_noise_token;

    float *states = (float *)malloc((size_t)n * state_stride * sizeof(float));
    float *hidden = (float *)malloc((size_t)n * hid * sizeof(float));
    float *base_logits = (float *)malloc((size_t)n * m->cfg.vocab *
                                          sizeof(float));
    const int debug_confidence =
        getenv("DSV4_DEBUG_DSPARK_CONFIDENCE") != NULL;
    float *confidence_hidden = debug_confidence
        ? (float *)malloc((size_t)n * hid * sizeof(float)) : NULL;
    if (!states || !hidden || !base_logits ||
        (debug_confidence && !confidence_hidden)) {
        free(states); free(hidden); free(base_logits); free(confidence_hidden);
        return -1;
    }
    for (int t = 0; t < n; t++) {
        if (embed_token(m, token_ids[t], anchor_position + t,
                        states + (size_t)t * state_stride) != 0) {
            free(states); free(hidden); free(base_logits); free(confidence_hidden);
            return -1;
        }
    }

    for (int stage = 0; stage < m->cfg.dspark_stages; stage++) {
        const int virtual_layer = m->cfg.n_layers + stage;
        prefetch_layer_weights(m, virtual_layer);
        DSV4LayerW *w = load_layer_profiled(m, virtual_layer);
        int rc = dspark_stage_forward(m, w, stage, token_ids,
                                      anchor_position, states, n);
        release_layer_weights(m, w);
        if (rc != 0) {
            free(states); free(hidden); free(base_logits); free(confidence_hidden);
            return -1;
        }
    }
    for (int t = 0; t < n; t++)
        dspark_head_hidden(m, states + (size_t)t * state_stride,
                           hidden + (size_t)t * hid,
                           confidence_hidden
                               ? confidence_hidden + (size_t)t * hid : NULL);
    head_logits_batch(m, hidden, n, base_logits);

    int previous = anchor_token;
    for (int t = 0; t < n; t++) {
        float *step_logits = base_logits + (size_t)t * m->cfg.vocab;
        int base_best = 0;
        for (int v = 1; v < m->cfg.vocab; v++)
            if (step_logits[v] > step_logits[base_best]) base_best = v;
        int next = getenv("DSV4_DSPARK_NO_MARKOV")
            ? base_best
            : dspark_markov_argmax(m, step_logits, previous);
        if (next < 0) {
            free(states); free(hidden); free(base_logits); free(confidence_hidden);
            return -1;
        }
        if (confidence_hidden) {
            float raw, probability;
            if (dspark_confidence_score(
                    m, confidence_hidden + (size_t)t * hid, previous,
                    &raw, &probability) != 0) {
                free(states); free(hidden); free(base_logits);
                free(confidence_hidden);
                return -1;
            }
            fprintf(stderr,
                    "dspark confidence: step=%d previous=%d raw=%.7g "
                    "probability=%.7g\n",
                    t, previous, (double)raw, (double)probability);
        }
        if (getenv("DSV4_DEBUG_DSPARK"))
            fprintf(stderr,
                    "dspark debug: step=%d previous=%d base=%d corrected=%d\n",
                    t, previous, base_best, next);
        draft_tokens[t] = next;
        previous = next;
    }
    free(states); free(hidden); free(base_logits); free(confidence_hidden);
    return n;
}

static void capture_mean_hidden(const float *states, int n_tokens, int mult,
                                int hid, int capture_index, int n_captures,
                                float *capture_hidden)
{
    const size_t state_elems = (size_t)mult * hid;
    for (int t = 0; t < n_tokens; t++) {
        const float *state = states + (size_t)t * state_elems;
        float *dst = capture_hidden +
                     ((size_t)t * n_captures + capture_index) * hid;
        for (int j = 0; j < hid; j++) {
            float sum = 0.0f;
            for (int lane = 0; lane < mult; lane++)
                sum += state[(size_t)lane * hid + j];
            dst[j] = f32bf(sum / (float)mult);
        }
    }
}

int dsv4_forward_token(DSV4Model *m, int token_id, int position, float *logits)
{
    const int hid = m->cfg.hidden;
    const int mult = m->cfg.hc_mult;
    const char *max_layer_env = getenv("DSV4_DEBUG_MAX_LAYER");
    int max_layer = max_layer_env ? (int)strtol(max_layer_env, NULL, 10) : -1;
    int completed_layers = 0;
    double forward_start = m->profiling ? now_s() : 0.0;

    if (token_id < 0 || token_id >= m->cfg.vocab) return -1;
    double embed_start = m->profiling ? now_s() : 0.0;
    if (embed_token(m, token_id, position, m->state) != 0) return -1;
    if (m->profiling) m->time_embed += now_s() - embed_start;

    DSV4ForwardBuffers b;
    if (!init_forward_buffers(&b, hid, mult)) return -1;

    if (m->layer_shard_map) prefetch_layer_weights(m, 0);
    DSV4LayerW *w = load_layer_profiled(m, 0);
    for (int layer = 0; layer < m->cfg.n_layers; layer++) {
        const int stop_after_layer = max_layer >= 0 && layer >= max_layer;
        const int has_next = layer + 1 < m->cfg.n_layers && !stop_after_layer;
        DSV4LayerLoadJob next_job;
        int next_started = has_next && layer_load_begin(m, layer + 1, &next_job);
        double prefetch_start = m->profiling ? now_s() : 0.0;
        if (has_next && !next_started) prefetch_layer_weights(m, layer + 1);
        if (m->profiling) m->time_prefetch += now_s() - prefetch_start;
        forward_loaded_layer(m, w, layer, token_id, position, m->state, &b);
        double release_start = m->profiling ? now_s() : 0.0;
        release_layer_weights(m, w);
        if (m->profiling) m->time_release += now_s() - release_start;
        completed_layers = layer + 1;
        if (!has_next) break;
        w = next_started ? layer_load_finish(&next_job)
                         : load_layer_profiled(m, layer + 1);
    }

    int rc = 0;
    if (logits && completed_layers == m->cfg.n_layers)
        rc = compute_logits(m, m->state, position, logits);
    free_forward_buffers(&b);
    if (m->profiling) m->time_total += now_s() - forward_start;
    m->forward_steps++;
    return rc;
}

int dsv4_prefill_capture(DSV4Model *m, const int *token_ids, int n_tokens,
                         int start_position, float *last_logits,
                         float *all_logits, const int *capture_layers,
                         int n_capture_layers, float *capture_hidden)
{
    if (!m || !token_ids || n_tokens <= 0 || start_position < 0 ||
        start_position > m->context - n_tokens) return -1;
    if (n_capture_layers < 0 ||
        ((n_capture_layers > 0) != (capture_layers != NULL)) ||
        ((n_capture_layers > 0) != (capture_hidden != NULL))) return -1;
    for (int i = 0; i < n_capture_layers; i++) {
        if (capture_layers[i] < 0 || capture_layers[i] >= m->cfg.n_layers ||
            (i > 0 && capture_layers[i] <= capture_layers[i - 1])) return -1;
    }

    for (int i = 0; i < n_tokens; i++)
        if (token_ids[i] < 0 || token_ids[i] >= m->cfg.vocab) return -1;

    const int hid = m->cfg.hidden;
    const int mult = m->cfg.hc_mult;
    const size_t state_elems = (size_t)mult * hid;
    if ((size_t)n_tokens > SIZE_MAX / state_elems / sizeof(float)) return -1;
    double forward_start = m->profiling ? now_s() : 0.0;
    float *states = (float *)malloc((size_t)n_tokens * state_elems * sizeof(float));
    if (!states) { fprintf(stderr, "dsv4: OOM prefill states\n"); return -1; }

    for (int i = 0; i < n_tokens; i++) {
        double embed_start = m->profiling ? now_s() : 0.0;
        if (embed_token(m, token_ids[i], start_position + i,
                        states + (size_t)i * state_elems) != 0) {
            free(states);
            return -1;
        }
        if (m->profiling) m->time_embed += now_s() - embed_start;
    }

    DSV4ForwardBuffers b;
    if (!init_forward_buffers(&b, hid, mult)) { free(states); return -1; }
    const char *max_layer_env = getenv("DSV4_DEBUG_MAX_LAYER");
    int max_layer = max_layer_env ? (int)strtol(max_layer_env, NULL, 10) : -1;
    int completed_layers = 0;
    int capture_index = 0;

    if (m->layer_shard_map) prefetch_layer_weights(m, 0);
    for (int layer = 0; layer < m->cfg.n_layers; layer++) {
        DSV4LayerW *w = load_layer_profiled(m, layer);
        double load_end = m->profiling ? now_s() : 0.0;
        prefetch_layer_weights(m, layer + 1);
        double prefetch_end = m->profiling ? now_s() : 0.0;
        for (int i = 0; i < n_tokens; i += DSV4_MOE_BATCH_TOKENS) {
            int count = n_tokens - i;
            if (count > DSV4_MOE_BATCH_TOKENS) count = DSV4_MOE_BATCH_TOKENS;
            forward_loaded_layer_batch(m, w, layer, token_ids + i,
                                       start_position + i,
                                       states + (size_t)i * state_elems,
                                       count, &b);
        }
        if (capture_index < n_capture_layers &&
            layer == capture_layers[capture_index]) {
            capture_mean_hidden(states, n_tokens, mult, hid, capture_index,
                                n_capture_layers, capture_hidden);
            capture_index++;
        }
        if (m->profiling) {
            m->time_prefetch += prefetch_end - load_end;
        }
        double release_start = m->profiling ? now_s() : 0.0;
        release_layer_weights(m, w);
        if (m->profiling) m->time_release += now_s() - release_start;
        completed_layers = layer + 1;
        if (max_layer >= 0 && layer >= max_layer) break;
    }

    float *last_state = states + (size_t)(n_tokens - 1) * state_elems;
    context_snapshot_capture_model_state(m, states, n_tokens, start_position);
    memcpy(m->state, last_state, state_elems * sizeof(float));
    int rc = 0;
    if (completed_layers == m->cfg.n_layers) {
        if (all_logits) {
            rc = compute_logits_batch(m, states, n_tokens, all_logits);
            if (rc == 0 && last_logits) {
                const float *last = all_logits + (size_t)(n_tokens - 1) * m->cfg.vocab;
                if (last_logits != last)
                    memcpy(last_logits, last, (size_t)m->cfg.vocab * sizeof(float));
            }
        } else if (last_logits) {
            rc = compute_logits(m, last_state, start_position + n_tokens - 1,
                                last_logits);
        }
    }
    free_forward_buffers(&b);
    free(states);
    if (m->profiling) m->time_total += now_s() - forward_start;
    m->forward_steps += n_tokens;
    return rc;
}

int dsv4_prefill(DSV4Model *m, const int *token_ids, int n_tokens,
                 int start_position, float *logits)
{
    if (n_tokens == 1)
        return dsv4_forward_token(m, token_ids[0], start_position, logits);
    return dsv4_prefill_capture(m, token_ids, n_tokens, start_position, logits,
                                NULL, NULL, 0, NULL);
}

/* =========================================================== report ==== */
#include <sys/resource.h>
void dsv4_model_report(const DSV4Model *m)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    long peak = (long)ru.ru_maxrss;    /* KiB on Linux/macOS */
    fprintf(stderr, "report: forward_steps=%ld threads=%d expert_io_workers=%d "
            "expert_access=%s head_access=%s layer_access=%s context=%d dspark=%s\n",
            m->forward_steps, m->threads,
            m->expert_pool ? m->expert_pool->workers : 0,
            expert_mmap_enabled(m) ? "mmap" :
                (m->expert_ring ? "io_uring" : "direct"),
            m->head_map_rows ? "mmap" : "stream",
            m->layer_shard_map ? "mmap" : "pread", m->context,
            m->dspark ? "ready" : "unavailable");
    fprintf(stderr,
            "report: expert cache policy=%s slots=%d (main=%d draft=%d) "
            "hits=%ld misses=%ld "
            "coalesced=%ld batch_prefetched=%ld\n",
            m->cache_per_layer
                ? (m->cache_lrfu
                    ? (m->cache_hash_min ? "hash-min-lrfu" : "per-layer-lrfu")
                    : m->cache_arc
                    ? (m->cache_hash_min ? "hash-min-arc" : "per-layer-arc")
                    : m->cache_slru
                    ? (m->cache_hash_min ? "hash-min-slru" : "per-layer-slru")
                    : (m->cache_hash_min ? "hash-min-lru" : "per-layer-lru"))
                : "global-lru",
            m->cache_slots,
            m->main_cache_slots, m->draft_cache_slots,
            m->cache_hits, m->cache_misses, m->coalesced_loads, m->batch_prefetched);
    fprintf(stderr, "report: expert bytes %s=%.2f GiB\n",
            expert_mmap_enabled(m) ? "advised" : "read",
            (double)m->expert_bytes_read / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "report: expert %s=%ld\n",
            expert_mmap_enabled(m) ? "advice ranges" : "read operations",
            m->expert_read_ops);
    if (m->expert_trace) {
        fflush(m->expert_trace);
        fprintf(stderr, "report: expert trace records=%ld\n",
                m->expert_trace_records);
    }
    for (int kind = 0; kind < 2; kind++) {
        if (m->route_prediction_groups[kind] == 0) continue;
        fprintf(stderr, "report: route predictor %s groups=%ld mean_overlap=%.3f "
                        "histogram",
                kind ? "hash" : "learned", m->route_prediction_groups[kind],
                (double)m->route_prediction_matches[kind] /
                    m->route_prediction_groups[kind]);
        for (int overlap = 0; overlap <= m->cfg.topk; overlap++)
            fprintf(stderr, " %d:%ld", overlap,
                    m->route_prediction_hist[kind][overlap]);
        fputc('\n', stderr);
        fprintf(stderr, "report: route predictor %s misses predicted=%ld "
                        "actual=%ld covered=%ld wasted=%ld\n",
                kind ? "hash" : "learned",
                m->route_prediction_predicted_misses[kind],
                m->route_prediction_actual_misses[kind],
                m->route_prediction_covered_misses[kind],
                m->route_prediction_wasted_misses[kind]);
        fprintf(stderr, "report: route predictor %s miss ranks",
                kind ? "hash" : "learned");
        for (int rank = 0; rank < m->cfg.topk; rank++)
            fprintf(stderr, " %d:%ld/%ld/%ld", rank,
                    m->route_prediction_rank_misses[kind][rank],
                    m->route_prediction_rank_covered[kind][rank],
                    m->route_prediction_rank_wasted[kind][rank]);
        fputs(" (predicted/covered/wasted)\n", stderr);
    }
    if (m->route_prefetch_reads > 0)
        fprintf(stderr,
                "report: batch route prefetch reads=%ld adopted=%ld "
                "wasted=%ld unplaced=%ld\n",
                m->route_prefetch_reads, m->route_prefetch_adopted,
                m->route_prefetch_wasted, m->route_prefetch_unplaced);
    fprintf(stderr, "report: wo_a cache layers=%d hits=%ld misses=%ld\n",
            m->wo_a_cache_layers, m->wo_a_cache_hits, m->wo_a_cache_misses);
    fprintf(stderr, "report: layer prefetch calls=%ld advised=%.2f GiB\n",
            m->prefetch_calls,
            (double)m->prefetch_bytes / (1024.0 * 1024.0 * 1024.0));
    if (m->profiling) {
        double known = m->time_embed + m->time_load + m->time_prefetch +
                       m->time_attn + m->time_moe + m->time_hc +
                       m->time_release + m->time_head;
        fprintf(stderr,
                "profile: total=%.3f embed=%.3f load=%.3f (layer_io=%.3f) "
                "prefetch=%.3f attn=%.3f moe=%.3f (expert_wait=%.3f "
                "expert_io=%.3f) hc=%.3f release=%.3f head=%.3f other=%.3f\n",
                m->time_total, m->time_embed, m->time_load, m->time_layer_io,
                m->time_prefetch, m->time_attn, m->time_moe,
                m->time_expert_read, m->time_expert_io, m->time_hc,
                m->time_release, m->time_head, m->time_total - known);
    }
    fprintf(stderr, "report: peak RSS=%.2f GiB (%ld bytes from rusage)\n",
            (double)peak / (1024.0 * 1024.0), peak * 1024L);
    fprintf(stderr, "report: page faults major=%ld minor=%ld context switches "
            "voluntary=%ld involuntary=%ld\n", (long)ru.ru_majflt,
            (long)ru.ru_minflt, (long)ru.ru_nvcsw, (long)ru.ru_nivcsw);
}
