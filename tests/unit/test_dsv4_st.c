/* SPDX-License-Identifier: Apache-2.0 */
/* test_dsv4_st.c - safetensors reader coverage for awkward shapes and shard
 * boundaries. DeepSeek dtypes (I8/I64/F8_E4M3/F8_E8M0) are also exercised by
 * the tiny graph and the full-checkpoint verifier. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "k3_st.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "tests/fixtures/st";
    K3St st;
    if (k3_st_open(&st, dir) != 0) {
        printf("FAIL: open %s\n", dir);
        return 1;
    }
    /* Reader fixture spans two shards; DeepSeek dtype coverage lives in the
     * tiny model test and tools/verify_dsv4_checkpoint.py. */
    CHECK(st.nshard >= 2, "two fixture shards");
    const K3Tensor *t = k3_st_find(&st, "plain.f32.2d");
    CHECK(t != NULL, "find plain.f32.2d");
    if (t) {
        CHECK(t->dtype == K3_DT_F32, "f32 dtype");
        CHECK(t->ndim == 2 && t->shape[0] == 16 && t->shape[1] == 16, "f32 shape");
        float buf[256];
        int64_t got = k3_st_read_f32(&st, t, buf);
        CHECK(got == 256, "f32 read count");
        CHECK(buf[0] != 0.0f && buf[255] != 0.0f, "f32 values nonzero");
    }
    const K3Tensor *b = k3_st_find(&st, "plain.bf16.1d");
    CHECK(b != NULL, "find plain.bf16.1d");
    if (b) {
        CHECK(b->dtype == K3_DT_BF16, "bf16 dtype");
        float buf[128];
        CHECK(k3_st_read_f32(&st, b, buf) == 128, "bf16 read count");
        CHECK(buf[0] != 0.0f && buf[127] != 0.0f, "bf16 values nonzero");
    }
    const K3Tensor *f = k3_st_find(&st, "tricky.f16.1d");
    CHECK(f != NULL && f->dtype == K3_DT_F16, "f16 dtype");
    k3_st_close(&st);
    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("test_dsv4_st: PASS\n");
    return 0;
}
