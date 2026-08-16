/* SPDX-License-Identifier: Apache-2.0 */
/*
 * bench_dsv4_kernels.c - microbenchmarks for the FP8/FP4 GEMV kernels.
 *
 * Measures the exact shapes the expert layers run:
 *   FP4 2048x4096   (gate/up: 2048 rows x 4096 cols)
 *   FP4 4096x2048   (down:   4096 rows x 2048 cols)
 *   FP8 4096x4096   (dense projection)
 *   BF16 1024x4096  (wo_a group)
 *
 * Inputs are deterministic (a fixed xorshift sequence) and each run prints a
 * checksum so a before/after optimisation comparison is reproducible. Absolute
 * milliseconds vary by machine; the checksums must not.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_internal.h"

static uint64_t xs_state = 0x9E3779B97F4A7C15ull;
static uint64_t xs_next(void)
{
    uint64_t x = xs_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    xs_state = x;
    return x;
}

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static float checksum(const float *v, int n)
{
    /* sum of squares scaled; stable across optimisation levels */
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    return (float)sqrt(s);
}

int main(void)
{
    enum { FP4_BATCH = 6, DENSE_BATCH = 5 };
    const int R1 = 2048, C1 = 4096;   /* FP4 gate/up */
    const int R2 = 4096, C2 = 2048;   /* FP4 down */
    const int R3 = 4096, C3 = 4096;   /* FP8 dense */
    const int R4 = 1024, C4 = 4096;   /* BF16 wo_a group */

    float *x2048 = (float *)malloc((size_t)C1 * sizeof(float));
    float *x4096 = (float *)malloc((size_t)C3 * sizeof(float));
    float *y = (float *)malloc((size_t)R2 * sizeof(float));
    float *y_pair = (float *)malloc((size_t)R2 * sizeof(float));
    float *y_batch[FP4_BATCH];
    float *y_dense_batch[DENSE_BATCH];
    const float *qx_batch[FP4_BATCH];
    const float *qsc_batch[FP4_BATCH];
    for (int t = 0; t < FP4_BATCH; t++)
        y_batch[t] = (float *)malloc((size_t)R1 * sizeof(float));
    for (int t = 0; t < DENSE_BATCH; t++)
        y_dense_batch[t] = (float *)malloc((size_t)R3 * sizeof(float));
    if (!x2048 || !x4096 || !y || !y_pair) {
        fprintf(stderr, "OOM\n"); return 1;
    }
    for (int t = 0; t < FP4_BATCH; t++) {
        if (!y_batch[t]) { fprintf(stderr, "OOM\n"); return 1; }
    }
    for (int t = 0; t < DENSE_BATCH; t++) {
        if (!y_dense_batch[t]) { fprintf(stderr, "OOM\n"); return 1; }
    }
    for (int i = 0; i < C1; i++) x2048[i] = (float)((int)(xs_next() % 2001) - 1000) / 1000.0f;
    for (int i = 0; i < C3; i++) x4096[i] = (float)((int)(xs_next() % 2001) - 1000) / 1000.0f;

    /* deterministic FP4 weight bytes and E8M0 scale codes (all scale 1.0) */
    uint8_t *w2048 = (uint8_t *)malloc((size_t)R1 * (C1 / 2));
    uint8_t *w2048_pair = (uint8_t *)malloc((size_t)R1 * (C1 / 2));
    uint8_t *w4096d = (uint8_t *)malloc((size_t)R2 * (C2 / 2));
    uint8_t *s2048 = (uint8_t *)malloc((size_t)R1 * ((C1 + 31) / 32));
    uint8_t *s2048_pair = (uint8_t *)malloc((size_t)R1 * ((C1 + 31) / 32));
    uint8_t *s4096d = (uint8_t *)malloc((size_t)R2 * ((C2 + 31) / 32));
    uint8_t *w4096f = (uint8_t *)malloc((size_t)R3 * C3);
    uint8_t *s4096f = (uint8_t *)malloc((size_t)((R3 + 127) / 128) * ((C3 + 127) / 128));
    uint16_t *w1024b = (uint16_t *)malloc((size_t)R4 * C4 * sizeof(uint16_t));
    float *y1024_ref = (float *)malloc((size_t)R4 * sizeof(float));
    if (!w2048 || !w2048_pair || !w4096d || !s2048 || !s2048_pair ||
        !s4096d || !w4096f || !s4096f || !w1024b || !y1024_ref) {
        fprintf(stderr, "OOM\n"); return 1;
    }
    for (size_t i = 0; i < (size_t)R1 * (C1 / 2); i++) w2048[i] = (uint8_t)xs_next();
    for (size_t i = 0; i < (size_t)R1 * (C1 / 2); i++) w2048_pair[i] = (uint8_t)xs_next();
    for (size_t i = 0; i < (size_t)R2 * (C2 / 2); i++) w4096d[i] = (uint8_t)xs_next();
    /* FP8 E4M3: 0..126 are finite (127 and 255 are NaN); stay in the finite range */
    for (size_t i = 0; i < (size_t)R3 * C3; i++) w4096f[i] = (uint8_t)(xs_next() % 127);
    memset(s2048, 127, (size_t)R1 * ((C1 + 31) / 32));
    memset(s2048_pair, 127, (size_t)R1 * ((C1 + 31) / 32));
    memset(s4096d, 127, (size_t)R2 * ((C2 + 31) / 32));
    memset(s4096f, 127, (size_t)((R3 + 127) / 128) * ((C3 + 127) / 128));
    for (int r = 0; r < R4; r++) {
        for (int c = 0; c < C4; c++) {
            uint16_t v = dsv4_f32_to_bf16((float)((int)(xs_next() % 2001) - 1000) /
                                          1000.0f);
            w1024b[(size_t)r * C4 + c] = v;
        }
    }

    /* FP4 activation quantisation (per 128) for the two input widths */
    float *qx2048 = (float *)malloc((size_t)C1 * sizeof(float));
    float *qsc2048 = (float *)malloc((size_t)(C1 / 128) * sizeof(float));
    float *qx4096 = (float *)malloc((size_t)C3 * sizeof(float));
    float *qsc4096 = (float *)malloc((size_t)(C3 / 128) * sizeof(float));
    if (!qx2048 || !qsc2048 || !qx4096 || !qsc4096) { fprintf(stderr, "OOM\n"); return 1; }
    dsv4_quant_fp8_codes(qx2048, qsc2048, x2048, C1);
    dsv4_quant_fp8_codes(qx4096, qsc4096, x4096, C3);
    for (int t = 0; t < FP4_BATCH; t++) {
        qx_batch[t] = qx2048;
        qsc_batch[t] = qsc2048;
    }

    double t0, t1, best;
    float ck;

    printf("bench_dsv4: FP4 2048x4096\n");
    dsv4_gemv_fp4_q(y, qx2048, qsc2048, w2048, s2048, C1, R1);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_fp4_q(y, qx2048, qsc2048, w2048, s2048, C1, R1);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    ck = checksum(y, R1);
    printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3, ck);

    printf("bench_dsv4: paired FP4 2x(2048x4096)\n");
    dsv4_gemv_fp4_pair_q(y, y_pair, qx2048, qsc2048,
                         w2048, s2048, w2048_pair, s2048_pair, C1, R1);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_fp4_pair_q(y, y_pair, qx2048, qsc2048,
                             w2048, s2048, w2048_pair, s2048_pair, C1, R1);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("  %.3f ms best-of-3  checksums %.6f %.6f\n", best * 1e3,
           checksum(y, R1), checksum(y_pair, R1));

    printf("bench_dsv4: batched FP4 6x(2048x4096)\n");
    dsv4_gemv_fp4_batch_q(y_batch, qx_batch, qsc_batch, FP4_BATCH,
                          w2048, s2048, C1, R1);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_fp4_batch_q(y_batch, qx_batch, qsc_batch, FP4_BATCH,
                              w2048, s2048, C1, R1);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3,
           checksum(y_batch[0], R1));

    printf("bench_dsv4: FP4 4096x2048\n");
    dsv4_gemv_fp4_q(y, qx4096, qsc4096, w4096d, s4096d, C2, R2);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_fp4_q(y, qx4096, qsc4096, w4096d, s4096d, C2, R2);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    ck = checksum(y, R2);
    printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3, ck);

    printf("bench_dsv4: FP8 4096x4096\n");
    dsv4_gemv_fp8_q(y, qx4096, qsc4096, w4096f, s4096f, C3, R3, 1);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_fp8_q(y, qx4096, qsc4096, w4096f, s4096f, C3, R3, 1);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    ck = checksum(y, R3);
    printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3, ck);

    printf("bench_dsv4: batched FP8 5x(4096x4096)\n");
    {
        float *outputs[DENSE_BATCH];
        const float *inputs[DENSE_BATCH];
        const float *scales[DENSE_BATCH];
        for (int t = 0; t < DENSE_BATCH; t++) {
            outputs[t] = y_dense_batch[t];
            inputs[t] = qx4096;
            scales[t] = qsc4096;
        }
        dsv4_gemv_fp8_batch_q(outputs, inputs, scales, DENSE_BATCH,
                              w4096f, s4096f, C3, R3, 1);
        best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            t0 = now_sec();
            dsv4_gemv_fp8_batch_q(outputs, inputs, scales, DENSE_BATCH,
                                  w4096f, s4096f, C3, R3, 1);
            t1 = now_sec();
            if (t1 - t0 < best) best = t1 - t0;
        }
        printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3,
               checksum(y_dense_batch[0], R3));
    }

    printf("bench_dsv4: paired FP8 2x(2048x4096)\n");
    {
        enum { INNER = 10 };
        const size_t matrix_bytes = (size_t)R1 * C1;
        const size_t scale_bytes = (size_t)(R1 / 128) * (C1 / 128);
        dsv4_gemv_fp8_pair_q(y, y_pair, qx2048, qsc2048,
                             w4096f, s4096f, w4096f + matrix_bytes,
                             s4096f + scale_bytes, C1, R1);
        best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            t0 = now_sec();
            for (int inner = 0; inner < INNER; inner++)
                dsv4_gemv_fp8_pair_q(y, y_pair, qx2048, qsc2048,
                                     w4096f, s4096f,
                                     w4096f + matrix_bytes,
                                     s4096f + scale_bytes, C1, R1);
            t1 = now_sec();
            if ((t1 - t0) / INNER < best) best = (t1 - t0) / INNER;
        }
        printf("  %.3f ms best-of-3x10  checksums %.6f %.6f\n", best * 1e3,
               checksum(y, R1), checksum(y_pair, R1));

        best = 1e30;
        for (int rep = 0; rep < 3; rep++) {
            t0 = now_sec();
            for (int inner = 0; inner < INNER; inner++) {
                dsv4_gemv_fp8_q(y, qx2048, qsc2048,
                                w4096f, s4096f, C1, R1, 1);
                dsv4_gemv_fp8_q(y_pair, qx2048, qsc2048,
                                w4096f + matrix_bytes,
                                s4096f + scale_bytes, C1, R1, 1);
            }
            t1 = now_sec();
            if ((t1 - t0) / INNER < best) best = (t1 - t0) / INNER;
        }
        printf("  %.3f ms separate 2x GEMV best-of-3x10\n", best * 1e3);
    }

    printf("bench_dsv4: FP8 2048x4096\n");
    dsv4_gemv_fp8_q(y, qx2048, qsc2048, w4096f, s4096f,
                    C1, R1, 1);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_fp8_q(y, qx2048, qsc2048, w4096f, s4096f,
                        C1, R1, 1);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3,
           checksum(y, R1));

    printf("bench_dsv4: BF16 1024x4096 row-major\n");
    dsv4_gemv_bf16(y1024_ref, x4096, w1024b, C4, R4, 1);
    best = 1e30;
    for (int rep = 0; rep < 3; rep++) {
        t0 = now_sec();
        dsv4_gemv_bf16(y1024_ref, x4096, w1024b, C4, R4, 1);
        t1 = now_sec();
        if (t1 - t0 < best) best = t1 - t0;
    }
    ck = checksum(y1024_ref, R4);
    printf("  %.3f ms best-of-3  checksum %.6f\n", best * 1e3, ck);

    free(x2048); free(x4096); free(y); free(y_pair);
    free(w2048); free(w2048_pair); free(w4096d);
    free(s2048); free(s2048_pair); free(s4096d);
    free(w4096f); free(s4096f); free(w1024b); free(y1024_ref);
    free(qx2048); free(qsc2048); free(qx4096); free(qsc4096);
    return 0;
}
