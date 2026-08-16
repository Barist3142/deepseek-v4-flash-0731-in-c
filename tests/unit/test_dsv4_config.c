/* SPDX-License-Identifier: Apache-2.0 */
/* test_dsv4_config.c - config.json reader tests against the official fixture. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "tests/fixtures/dsv4_config.json";
    DSV4Config c;
    if (!dsv4_config_load_file(&c, path)) {
        printf("FAIL: config load\n");
        return 1;
    }
    CHECK(c.hidden == 4096, "hidden");
    CHECK(c.n_layers == 43, "n_layers");
    CHECK(c.vocab == 129280, "vocab");
    CHECK(c.moe_inter == 2048, "moe_inter");
    CHECK(c.n_experts == 256, "n_experts");
    CHECK(c.n_shared == 1, "n_shared");
    CHECK(c.topk == 6, "topk");
    CHECK(c.n_hash_layers == 3, "n_hash_layers");
    CHECK(c.n_heads == 64, "n_heads");
    CHECK(c.head_dim == 512, "head_dim");
    CHECK(c.rope_dim == 64, "rope_dim");
    CHECK(c.q_lora == 1024, "q_lora");
    CHECK(c.o_groups == 8, "o_groups");
    CHECK(c.o_lora == 1024, "o_lora");
    CHECK(c.window == 128, "window");
    CHECK(c.index_heads == 64, "index_heads");
    CHECK(c.index_dim == 128, "index_dim");
    CHECK(c.index_topk == 512, "index_topk");
    CHECK(c.hc_mult == 4, "hc_mult");
    CHECK(c.hc_iters == 20, "hc_iters");
    CHECK(c.max_position == 1048576, "max_position");
    CHECK(c.original_position == 65536, "original_position");
    CHECK(c.rope_theta == 10000.0f, "rope_theta");
    CHECK(c.compress_rope_theta == 160000.0f, "compress_rope_theta");
    CHECK(c.rope_factor == 16.0f, "rope_factor");
    CHECK(c.beta_fast == 32.0f, "beta_fast");
    CHECK(c.beta_slow == 1.0f, "beta_slow");
    CHECK(c.rms_eps == 1e-6f, "rms_eps");
    CHECK(c.hc_eps == 1e-6f, "hc_eps");
    CHECK(c.route_scale == 1.5f, "route_scale");
    CHECK(c.swiglu_limit == 10.0f, "swiglu_limit");
    CHECK(c.dspark_stages == 3, "three DSpark stages");
    CHECK(c.dspark_block_size == 5, "DSpark block size");
    CHECK(c.dspark_noise_token == 128799, "DSpark noise token");
    CHECK(c.dspark_markov_rank == 256, "DSpark Markov rank");
    CHECK(c.dspark_target_layer[0] == 40 && c.dspark_target_layer[1] == 41 &&
          c.dspark_target_layer[2] == 42, "DSpark target layers");

    /* compress_ratios: 0,0 then 4/128 alternating, 21 fours, 20 128s */
    CHECK(c.compress_ratio[0] == 0 && c.compress_ratio[1] == 0, "first two zeros");
    int n4 = 0, n128 = 0;
    for (int i = 2; i < c.n_layers; i++) {
        if (i % 2 == 0) { CHECK(c.compress_ratio[i] == 4, "ratio 4 at even layer"); n4++; }
        else { CHECK(c.compress_ratio[i] == 128, "ratio 128 at odd layer"); n128++; }
    }
    CHECK(n4 == 21 && n128 == 20, "ratio counts 21/20");

    /* memory plan sanity */
    DSV4MemoryPlan p;
    CHECK(dsv4_memory_plan(&c, 4096, 2ull * 1024 * 1024 * 1024, &p) == 1, "2 GiB plan ok");
    CHECK(p.expert_cache_slots >= 1, "at least one slot");
    uint64_t ctx = dsv4_context_bytes(&c, 4096);
    CHECK(ctx > 0, "context bytes positive");
    CHECK(dsv4_auto_context(&c, 15ull * 1024 * 1024 * 1024) == 65536,
          "15 GiB auto context reaches native 64K");
    CHECK(dsv4_auto_context(&c, 8ull * 1024 * 1024 * 1024) == 32768,
          "8 GiB auto context preserves cache capacity");
    CHECK(dsv4_auto_context(&c, 4ull * 1024 * 1024 * 1024) == 4096,
          "4 GiB auto context preserves wo_a prefix");
    DSV4MemoryPlan dense, packed;
    CHECK(unsetenv("DSV4_PACKED_WO_A") == 0, "select decoded wo_a plan");
    CHECK(dsv4_memory_plan(&c, 65536, 15ull * 1024 * 1024 * 1024,
                           &dense) == 1,
          "15 GiB decoded wo_a plan");
    CHECK(setenv("DSV4_PACKED_WO_A", "1", 1) == 0,
          "select packed wo_a plan");
    CHECK(dsv4_memory_plan(&c, 65536, 15ull * 1024 * 1024 * 1024,
                           &packed) == 1,
          "15 GiB packed wo_a plan");
    CHECK(packed.wo_a_cache_bytes < dense.wo_a_cache_bytes,
          "packed wo_a consumes fewer planned bytes");
    CHECK(packed.expert_cache_slots > dense.expert_cache_slots,
          "packed wo_a leaves more expert slots");
    CHECK(unsetenv("DSV4_PACKED_WO_A") == 0, "restore decoded wo_a plan");
    /* budget too small must refuse */
    DSV4MemoryPlan bad;
    CHECK(dsv4_memory_plan(&c, 1048576, 2ull * 1024 * 1024 * 1024, &bad) == 0,
          "1M context with 2 GiB refuses");

    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("test_dsv4_config: PASS\n");
    return 0;
}
