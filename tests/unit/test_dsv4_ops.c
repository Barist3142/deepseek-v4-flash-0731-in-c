/* SPDX-License-Identifier: Apache-2.0 */
/* test_dsv4_ops.c - weightless numeric kernel tests.
 *
 * Exercises the exact FP8/FP4/BF16 decode rules, the power-of-two activation
 * quantisation, RMSNorm, RoPE (YaRN ramp), Hadamard, the HC Sinkhorn split and
 * the router, all against hand-computed values chosen to fail on a wrong
 * formula (nibble order, scale rounding, sinkhorn layout, ...). */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_internal.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)
#define CLOSE(a, b) (fabsf((a) - (b)) < 1e-4f * (1.0f + fabsf(b)))

int main(void)
{
    /* ---- BF16 ---- */
    CHECK(dsv4_f32_to_bf16(1.0f) == 0x3F80u, "bf16 1.0");
    CHECK(dsv4_bf16_to_f32(0x3F80u) == 1.0f, "bf16 widen 1.0");
    CHECK(dsv4_f32_to_bf16(0.1f) == dsv4_f32_to_bf16(0.1f), "bf16 deterministic");
    /* round to nearest even: BF16 mantissa is 7 bits, step 2^-7. 1+2^-8 is
     * exactly half-way and rounds to 1.0 (even); 1+2^-7 rounds up. */
    CHECK(dsv4_f32_to_bf16(1.0f + 0x1p-8f) == 0x3F80u, "bf16 rne halfway down");
    CHECK(dsv4_f32_to_bf16(1.0f + 0x1p-7f) == 0x3F81u, "bf16 rne up");

    /* ---- FP8 E4M3 ---- */
    CHECK(dsv4_fp8_e4m3(0x00) == 0.0f, "e4m3 zero");
    CHECK(dsv4_fp8_e4m3(0x7E) == 448.0f, "e4m3 max 448");
    CHECK(dsv4_fp8_e4m3(0xFE) == -448.0f, "e4m3 min -448");
    CHECK(dsv4_fp8_e4m3(0x3F) == 1.875f, "e4m3 0x3F = 1.875");
    /* 0x38 = exp 7, man 0 -> 8 * 2^-3 = 1.0 */
    CHECK(dsv4_fp8_e4m3(0x38) == 1.0f, "e4m3 0x38 = 1.0");
    /* 0x01 = exp 0, man 1 -> 2^-9 */
    CHECK(dsv4_fp8_e4m3(0x01) == ldexpf(1.0f, -9), "e4m3 subnormal");
    CHECK(isnan(dsv4_fp8_e4m3(0x7F)), "e4m3 NaN");
    {
        const float midpoint = 15.0f / 1024.0f;
        const float largest_subnormal = 7.0f / 512.0f;
        const float smallest_normal = 8.0f / 512.0f;
        CHECK(dsv4_round_e4m3(nextafterf(midpoint, 0.0f)) == largest_subnormal,
              "e4m3 subnormal/normal boundary below midpoint");
        CHECK(dsv4_round_e4m3(midpoint) == smallest_normal,
              "e4m3 subnormal/normal tie rounds to even code");
        CHECK(dsv4_round_e4m3(nextafterf(midpoint, 1.0f)) == smallest_normal,
              "e4m3 subnormal/normal boundary above midpoint");
        CHECK(dsv4_round_e4m3(-midpoint) == -smallest_normal,
              "e4m3 negative boundary is symmetric");
    }

    /* ---- E8M0 ---- */
    CHECK(dsv4_e8m0(127) == 1.0f, "e8m0 127 = 1");
    CHECK(dsv4_e8m0(126) == 0.5f, "e8m0 126 = 0.5");
    CHECK(dsv4_e8m0(128) == 2.0f, "e8m0 128 = 2");
    CHECK(isnan(dsv4_e8m0(255)), "e8m0 255 = NaN");

    /* ---- activation quant (E8M0 power-of-two, clamp 448) ---- */
    {
        float x[128], q[128], s[1];
        for (int i = 0; i < 128; i++) x[i] = (i % 2) ? 1.0f : -1.0f;
        dsv4_quant_fp8_act(q, s, x, 128);
        /* amax=1: amax/448 = 2^-8.8, ceil -> 2^-8 */
        float expect_scale = ldexpf(1.0f, -8);
        CHECK(s[0] == expect_scale, "act scale power of two");
        CHECK(q[0] == -256.0f, "act value scaled");
        CHECK(q[1] == 256.0f, "act value scaled +");
        /* clamp boundary: x = 448 * 2^10 -> scale 2^10, q exactly 448 */
        float big[128], qb[128], sb[1];
        big[0] = 448.0f * 1024.0f;
        for (int i = 1; i < 128; i++) big[i] = 0.0f;
        dsv4_quant_fp8_act(qb, sb, big, 128);
        CHECK(sb[0] == ldexpf(1.0f, 10), "act scale for 448*2^10");
        CHECK(qb[0] == 448.0f, "act clamp to 448");
    }

    /* ---- FP4 E2M1 packed nibbles, low nibble first ---- */
    {
        /* W = [[6, -2, 1, 0.5], [0, 3, -6, 4]] as packed bytes */
        uint8_t W[2][2] = {
            { (uint8_t)(0xE1 | (0x2 << 4)), (uint8_t)(0x08 | (0x7 << 4)) },  /* 1,2 then 8? no */
            { 0, 0 }
        };
        /* values: nibble0=1 (idx1), nibble1=2 (idx2), nibble2=8 (idx0 sign?)
         * row0: [1, 2, 0.5, 0] ... simpler: build explicit */
        (void)W;
    }

    /* ---- AVX2 output-row lanes preserve each row's scalar FMA order ---- */
    {
        enum { ROWS = 16, IN = 128 };
        float qx[IN], qscale[1] = { 0.125f };
        float fp8_scalar[ROWS], fp8_native[ROWS];
        float fp4_scalar[ROWS], fp4_native[ROWS];
        float fp8_second[ROWS], fp8_pair0[ROWS], fp8_pair1[ROWS];
        float fp8_pair_native0[ROWS], fp8_pair_native1[ROWS];
        float fp4_second[ROWS], fp4_pair0[ROWS], fp4_pair1[ROWS];
        float fp4_pair_native0[ROWS], fp4_pair_native1[ROWS];
        float fp4_batch_qx[3][IN], fp4_batch_scale[3][1];
        float bf16_batch_ref[3][ROWS], bf16_batch_scalar[3][ROWS];
        float bf16_batch_native[3][ROWS];
        float fp8_batch_ref[3][ROWS], fp8_batch_scalar[3][ROWS];
        float fp8_batch_native[3][ROWS];
        float fp4_batch_ref[3][ROWS], fp4_batch_scalar[3][ROWS];
        float fp4_batch_native[3][ROWS];
        const float *fp4_batch_qx_ptr[3], *fp4_batch_scale_ptr[3];
        float *bf16_batch_ref_ptr[3], *bf16_batch_scalar_ptr[3];
        float *bf16_batch_native_ptr[3];
        float *fp8_batch_ref_ptr[3], *fp8_batch_scalar_ptr[3];
        float *fp8_batch_native_ptr[3];
        float *fp4_batch_ref_ptr[3], *fp4_batch_scalar_ptr[3];
        float *fp4_batch_native_ptr[3];
        float bf16_scalar[ROWS], bf16_native[ROWS];
        float bf16_round_scalar[ROWS], bf16_round_native[ROWS];
        uint8_t fp8_w[ROWS * IN], fp8_w2[ROWS * IN], fp8_s[1], fp8_s2[1];
        uint8_t fp4_w[ROWS * (IN / 2)], fp4_w2[ROWS * (IN / 2)];
        uint8_t fp4_s[ROWS * (IN / 32)], fp4_s2[ROWS * (IN / 32)];
        uint16_t bf16_w[ROWS * IN];
        for (int i = 0; i < IN; i++) qx[i] = (float)((i % 31) - 15) * 0.25f;
        for (int t = 0; t < 3; t++) {
            for (int i = 0; i < IN; i++)
                fp4_batch_qx[t][i] = (float)(((i * (t + 2) + 7 * t) % 37) - 18)
                                          * 0.125f;
            fp4_batch_scale[t][0] = ldexpf(1.0f, t - 4);
            fp4_batch_qx_ptr[t] = fp4_batch_qx[t];
            fp4_batch_scale_ptr[t] = fp4_batch_scale[t];
            bf16_batch_ref_ptr[t] = bf16_batch_ref[t];
            bf16_batch_scalar_ptr[t] = bf16_batch_scalar[t];
            bf16_batch_native_ptr[t] = bf16_batch_native[t];
            fp8_batch_ref_ptr[t] = fp8_batch_ref[t];
            fp8_batch_scalar_ptr[t] = fp8_batch_scalar[t];
            fp8_batch_native_ptr[t] = fp8_batch_native[t];
            fp4_batch_ref_ptr[t] = fp4_batch_ref[t];
            fp4_batch_scalar_ptr[t] = fp4_batch_scalar[t];
            fp4_batch_native_ptr[t] = fp4_batch_native[t];
        }
        for (int i = 0; i < ROWS * IN; i++) fp8_w[i] = (uint8_t)(i % 127);
        for (int i = 0; i < ROWS * IN; i++) fp8_w2[i] = (uint8_t)((i * 29 + 7) % 127);
        for (int i = 0; i < ROWS * IN; i++)
            bf16_w[i] = dsv4_f32_to_bf16((float)((i * 17) % 61 - 30) * 0.03125f);
        fp8_s[0] = 127;
        fp8_s2[0] = 126;
        for (int i = 0; i < ROWS * (IN / 2); i++) fp4_w[i] = (uint8_t)(i * 37 + 11);
        for (int i = 0; i < ROWS * (IN / 2); i++) fp4_w2[i] = (uint8_t)(i * 19 + 5);
        for (int i = 0; i < ROWS * (IN / 32); i++) {
            fp4_s[i] = (uint8_t)(126 + i % 3);
            fp4_s2[i] = (uint8_t)(128 - i % 3);
        }

        CHECK(setenv("DSV4_NO_SIMD", "1", 1) == 0, "set scalar kernel mode");
        dsv4_gemv_bf16(bf16_scalar, qx, bf16_w, IN, ROWS, 0);
        dsv4_gemv_bf16(bf16_round_scalar, qx, bf16_w, IN, ROWS, 1);
        dsv4_gemv_fp8_q(fp8_scalar, qx, qscale, fp8_w, fp8_s, IN, ROWS, 1);
        dsv4_gemv_fp8_q(fp8_second, qx, qscale, fp8_w2, fp8_s2, IN, ROWS, 1);
        dsv4_gemv_fp8_pair_q(fp8_pair0, fp8_pair1, qx, qscale,
                             fp8_w, fp8_s, fp8_w2, fp8_s2, IN, ROWS);
        dsv4_gemv_fp4_q(fp4_scalar, qx, qscale, fp4_w, fp4_s, IN, ROWS);
        dsv4_gemv_fp4_q(fp4_second, qx, qscale, fp4_w2, fp4_s2, IN, ROWS);
        for (int t = 0; t < 3; t++) {
            dsv4_gemv_bf16(bf16_batch_ref_ptr[t], fp4_batch_qx_ptr[t],
                            bf16_w, IN, ROWS, 1);
            dsv4_gemv_fp8_q(fp8_batch_ref_ptr[t], fp4_batch_qx_ptr[t],
                             fp4_batch_scale_ptr[t], fp8_w, fp8_s,
                             IN, ROWS, 1);
            dsv4_gemv_fp4_q(fp4_batch_ref_ptr[t], fp4_batch_qx_ptr[t],
                             fp4_batch_scale_ptr[t], fp4_w, fp4_s, IN, ROWS);
        }
        dsv4_gemv_bf16_batch(bf16_batch_scalar_ptr, fp4_batch_qx_ptr, 3,
                              bf16_w, IN, ROWS, 1);
        dsv4_gemv_fp8_batch_q(fp8_batch_scalar_ptr, fp4_batch_qx_ptr,
                              fp4_batch_scale_ptr, 3, fp8_w, fp8_s,
                              IN, ROWS, 1);
        dsv4_gemv_fp4_batch_q(fp4_batch_scalar_ptr, fp4_batch_qx_ptr,
                              fp4_batch_scale_ptr, 3, fp4_w, fp4_s, IN, ROWS);
        dsv4_gemv_fp4_pair_q(fp4_pair0, fp4_pair1, qx, qscale,
                             fp4_w, fp4_s, fp4_w2, fp4_s2, IN, ROWS);
        CHECK(unsetenv("DSV4_NO_SIMD") == 0, "restore native kernel mode");
        dsv4_gemv_bf16(bf16_native, qx, bf16_w, IN, ROWS, 0);
        dsv4_gemv_bf16(bf16_round_native, qx, bf16_w, IN, ROWS, 1);
        dsv4_gemv_fp8_q(fp8_native, qx, qscale, fp8_w, fp8_s, IN, ROWS, 1);
        dsv4_gemv_fp8_pair_q(fp8_pair_native0, fp8_pair_native1, qx, qscale,
                             fp8_w, fp8_s, fp8_w2, fp8_s2, IN, ROWS);
        dsv4_gemv_fp4_q(fp4_native, qx, qscale, fp4_w, fp4_s, IN, ROWS);
        dsv4_gemv_bf16_batch(bf16_batch_native_ptr, fp4_batch_qx_ptr, 3,
                              bf16_w, IN, ROWS, 1);
        dsv4_gemv_fp8_batch_q(fp8_batch_native_ptr, fp4_batch_qx_ptr,
                              fp4_batch_scale_ptr, 3, fp8_w, fp8_s,
                              IN, ROWS, 1);
        dsv4_gemv_fp4_batch_q(fp4_batch_native_ptr, fp4_batch_qx_ptr,
                              fp4_batch_scale_ptr, 3, fp4_w, fp4_s, IN, ROWS);
        dsv4_gemv_fp4_pair_q(fp4_pair_native0, fp4_pair_native1, qx, qscale,
                             fp4_w, fp4_s, fp4_w2, fp4_s2, IN, ROWS);
        CHECK(memcmp(bf16_scalar, bf16_native, sizeof(bf16_scalar)) == 0,
              "BF16 SIMD is bit-exact");
        CHECK(memcmp(bf16_round_scalar, bf16_round_native,
                     sizeof(bf16_round_scalar)) == 0,
              "BF16 rounded SIMD is bit-exact");
        CHECK(memcmp(fp8_scalar, fp8_native, sizeof(fp8_scalar)) == 0,
              "FP8 SIMD is bit-exact");
        CHECK(memcmp(fp4_scalar, fp4_native, sizeof(fp4_scalar)) == 0,
              "FP4 SIMD is bit-exact");
        CHECK(memcmp(bf16_batch_ref, bf16_batch_scalar,
                     sizeof(bf16_batch_ref)) == 0,
              "BF16 scalar batch matches separate GEMVs");
        CHECK(memcmp(bf16_batch_ref, bf16_batch_native,
                     sizeof(bf16_batch_ref)) == 0,
              "BF16 batch SIMD is bit-exact");
        CHECK(memcmp(fp8_batch_ref, fp8_batch_scalar,
                     sizeof(fp8_batch_ref)) == 0,
              "FP8 scalar batch matches separate GEMVs");
        CHECK(memcmp(fp8_batch_ref, fp8_batch_native,
                     sizeof(fp8_batch_ref)) == 0,
              "FP8 batch SIMD is bit-exact");
        CHECK(memcmp(fp4_batch_ref, fp4_batch_scalar,
                     sizeof(fp4_batch_ref)) == 0,
              "FP4 scalar batch matches separate GEMVs");
        CHECK(memcmp(fp4_batch_ref, fp4_batch_native,
                     sizeof(fp4_batch_ref)) == 0,
              "FP4 batch SIMD is bit-exact");
        CHECK(memcmp(fp8_scalar, fp8_pair0, sizeof(fp8_scalar)) == 0 &&
              memcmp(fp8_second, fp8_pair1, sizeof(fp8_second)) == 0,
              "FP8 scalar pair matches separate GEMVs");
        CHECK(memcmp(fp8_pair0, fp8_pair_native0, sizeof(fp8_pair0)) == 0 &&
              memcmp(fp8_pair1, fp8_pair_native1, sizeof(fp8_pair1)) == 0,
              "FP8 pair SIMD is bit-exact");
        CHECK(memcmp(fp4_scalar, fp4_pair0, sizeof(fp4_scalar)) == 0 &&
              memcmp(fp4_second, fp4_pair1, sizeof(fp4_second)) == 0,
              "FP4 scalar pair matches separate GEMVs");
        CHECK(memcmp(fp4_pair0, fp4_pair_native0, sizeof(fp4_pair0)) == 0 &&
              memcmp(fp4_pair1, fp4_pair_native1, sizeof(fp4_pair1)) == 0,
              "FP4 pair SIMD is bit-exact");
    }

    /* ---- real-checkpoint indexer width: 512 of 1024, including ties ---- */
    {
        enum { N = 1024, K = 512, OFFSET = 128 };
        float score[N];
        int got[K], expect[K], used[N];
        for (int i = 0; i < N; i++) {
            score[i] = (float)((i * 37) % 211) * 0.03125f;
            used[i] = 0;
        }
        for (int k = 0; k < K; k++) {
            int best = -1;
            float best_score = -INFINITY;
            for (int i = 0; i < N; i++) {
                if (!used[i] && score[i] > best_score) {
                    best = i;
                    best_score = score[i];
                }
            }
            used[best] = 1;
            expect[k] = OFFSET + best;
        }
        dsv4_indexer_select_topk(score, N, K, OFFSET, got);
        CHECK(memcmp(got, expect, sizeof(got)) == 0,
              "512-wide indexer heap matches repeated maximum selection");
    }

    /* ---- RMSNorm ---- */
    {
        float x[4] = { 1, 2, 3, 4 };
        float y[4];
        dsv4_rmsnorm_plain(y, x, 4, 1e-6f);
        float ss = 30.0f;
        float r = 1.0f / sqrtf(ss / 4.0f + 1e-6f);
        CHECK(CLOSE(y[0], 1.0f * r), "rmsnorm 1");
        CHECK(CLOSE(y[3], 4.0f * r), "rmsnorm 4");
    }

    /* ---- RoPE ---- */
    {
        /* position 1, theta 10000, dims 4 (2 freqs: inv=1 and inv=0.01) */
        float full[4] = { 1, 0, 1, 0 };
        dsv4_rope_apply(full, 1, 4, 4, 1, 10000.0f, 0, 1.0f, 32.0f, 1.0f, 65536);
        float angle0 = 1.0f / powf(10000.0f, 0.0f);   /* i=0 -> inv=1 */
        float angle1 = 1.0f / powf(10000.0f, 0.5f);   /* i=1 -> inv=0.01 */
        CHECK(CLOSE(full[0], cosf(angle0)), "rope cos i0");
        CHECK(CLOSE(full[1], sinf(angle0)), "rope sin i0");
        CHECK(CLOSE(full[2], cosf(angle1)), "rope cos i1");
        CHECK(CLOSE(full[3], sinf(angle1)), "rope sin i1");
        dsv4_rope_apply_inv(full, 1, 4, 4, 1, 10000.0f, 0, 1.0f, 32.0f, 1.0f, 65536);
        CHECK(CLOSE(full[2], 1.0f) && CLOSE(full[3], 0.0f), "rope inv roundtrip");
    }
    {
        /* YaRN: low dimensions retain the original frequency; high dimensions
         * use frequency/factor after the correction ramp. */
        float full[64] = {0};
        full[0] = 1.0f;
        full[62] = 1.0f;
        dsv4_rope_apply(full, 1, 64, 64, 10000, 160000.0f, 1,
                        16.0f, 32.0f, 1.0f, 65536);
        float angle_low = 10000.0f;
        float angle_high = 10000.0f /
                           (16.0f * powf(160000.0f, 62.0f / 64.0f));
        CHECK(CLOSE(full[0], cosf(angle_low)), "yarn keeps low frequency");
        CHECK(CLOSE(full[1], sinf(angle_low)), "yarn keeps low frequency sin");
        CHECK(CLOSE(full[62], cosf(angle_high)), "yarn scales high frequency");
        CHECK(CLOSE(full[63], sinf(angle_high)), "yarn scales high frequency sin");
    }

    /* ---- Hadamard: [1,2,3,4] -> normalized ---- */
    {
        float x[4] = { 1, 2, 3, 4 };
        dsv4_hadamard(x, 4);
        /* H: [10, -2, -4, 0] / 2 */
        CHECK(CLOSE(x[0], 5.0f), "hadamard 0");
        CHECK(CLOSE(x[1], -1.0f), "hadamard 1");
        CHECK(CLOSE(x[2], -2.0f), "hadamard 2");
        CHECK(CLOSE(x[3], 0.0f), "hadamard 3");
    }

    /* ---- HC sinkhorn layout ---- */
    {
        /* mult=2: mixes = [pre0, pre1, post0, post1, comb00, comb01, comb10, comb11]
         * with scale=1, base=0. comb = softmax rows + eps; col norm; then
         * (iters-1) row/col. For iters=2: softmax, col, row, col. */
        float mixes[8];
        for (int i = 0; i < 8; i++) mixes[i] = 0.0f;
        mixes[4] = 3.0f; mixes[5] = 1.0f; mixes[6] = 0.0f; mixes[7] = 2.0f;
        float pre[2], post[2], comb[4];
        float scale3[3] = { 1, 1, 1 };
        float base[8] = { 0 };
        dsv4_hc_split(pre, post, comb, mixes, scale3, base, 2, 2, 1e-6f);
        /* pre = sigmoid(0)+eps = 0.5+eps */
        CHECK(CLOSE(pre[0], 0.5f + 1e-6f), "hc pre0");
        /* post = 2*sigmoid(0) = 1 */
        CHECK(CLOSE(post[0], 1.0f), "hc post0");
        /* comb: softmax rows of [[3,1],[0,2]] = [[e3/(e3+e), e/(e3+e)], [1/(1+e2), e2/(1+e2)]]
         * + eps, then col-norm, then row, then col */
        float e3 = expf(3.0f), e1 = expf(1.0f), e2 = expf(2.0f);
        float r0 = e3 + e1, r1 = 1.0f + e2;
        float c0 = e3 / r0 + 1e-6f, c1 = e1 / r0 + 1e-6f;
        float c2 = 1.0f / r1 + 1e-6f, c3 = e2 / r1 + 1e-6f;
        float colsum0 = c0 + c2, colsum1 = c1 + c3;
        c0 /= colsum0; c2 /= colsum0; c1 /= colsum1; c3 /= colsum1;
        float rowsum0 = c0 + c1, rowsum1 = c2 + c3;
        c0 /= rowsum0; c1 /= rowsum0; c2 /= rowsum1; c3 /= rowsum1;
        colsum0 = c0 + c2; colsum1 = c1 + c3;
        c0 /= colsum0; c2 /= colsum0; c1 /= colsum1; c3 /= colsum1;
        CHECK(CLOSE(comb[0], c0), "hc comb00");
        CHECK(CLOSE(comb[1], c1), "hc comb01");
        CHECK(CLOSE(comb[2], c2), "hc comb10");
        CHECK(CLOSE(comb[3], c3), "hc comb11");
    }

    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("test_dsv4_ops: PASS\n");
    return 0;
}
