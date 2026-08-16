/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_tensor.c - memory planning.
 *
 * The only knob a normal user sees is --memory-gib; this file turns that into a
 * concrete plan. The fixed cost is the runtime working set (1200 MiB), the
 * per-context persistent memory, persistent prefixes of the decoded wo_a
 * projection, and a bounded LRU cache of PACKED experts (never widened). This
 * is what lets a 166.9 GB checkpoint run in a few GiB.
 *
 * Persistent context memory (from the architecture; verified against the tensor
 * shapes in docs/DSV4_ARCHITECTURE.md):
 *
 *   43 * 128 * 512 * 4        sliding-window KV for all 43 layers (fp32)
 *   21 * ceil(C/4)  * 512 * 4 ratio-4 main compressors (21 layers)
 *   21 * ceil(C/4)  * 128 * 4 ratio-4 indexer states
 *   20 * ceil(C/128) * 512 * 4 ratio-128 main compressors (20 layers)
 *
 * One routed expert is 13,369,344 raw bytes (three packed FP4 matrices plus
 * three E8M0 scale tables); each cache slot also carries 4 KiB of alignment
 * padding, so a slot costs 13,369,344 + 4*4096. The plan fails loudly when the
 * budget cannot hold runtime reserve + context + one expert.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "dsv4.h"

#define GIB (1024ull * 1024 * 1024)
#define RUNTIME_RESERVE (1258291200ull)              /* 1200 MiB = 1.171875 GiB     */
#define EXPERT_RAW_BYTES (13369344ull)               /* 3 * 4194304 + 3 * 262144    */
#define EXPERT_SLOT_BYTES (EXPERT_RAW_BYTES + 4 * 4096ull)
#define MIN_EXPERT_CACHE GIB

/* ceil(a/b) for positive 64-bit values, without overflow at a+b-1. */
static uint64_t ceil_div(uint64_t a, uint64_t b)
{
    return a / b + (a % b != 0);
}

uint64_t dsv4_context_bytes(const DSV4Config *c, int context)
{
    int n4 = 0, n128 = 0;
    for (int i = 0; i < c->n_layers; i++) {
        if (c->compress_ratio[i] == 4) n4++;
        else if (c->compress_ratio[i] == 128) n128++;
    }
    const uint64_t C = (uint64_t)context;
    uint64_t ctx = 0;
    ctx += (uint64_t)c->n_layers * (uint64_t)c->window * 512u * 4u;
    ctx += (uint64_t)n4 * ceil_div(C, 4) * 512u * 4u;
    ctx += (uint64_t)n4 * ceil_div(C, 4) * (uint64_t)c->index_dim * 4u;
    ctx += (uint64_t)n128 * ceil_div(C, 128) * 512u * 4u;
    return ctx;
}

static int memory_plan_impl(const DSV4Config *c, int context,
                            uint64_t budget_bytes, DSV4MemoryPlan *plan,
                            int verbose)
{
    if (!plan || !c || context < 0) return 0;
    memset(plan, 0, sizeof *plan);

    if (budget_bytes == 0 || budget_bytes > (1ull << 42)) return 0;

    uint64_t ctx = dsv4_context_bytes(c, context);
    if (ctx < (uint64_t)c->n_layers * (uint64_t)c->window * 512u * 4u) return 0;

    uint64_t wo_a_values = (uint64_t)c->o_lora * (uint64_t)c->n_heads
                         * (uint64_t)c->head_dim;
    uint64_t wo_a_layer = wo_a_values * 2u;
    if (getenv("DSV4_PACKED_WO_A")) {
        uint64_t rows = (uint64_t)c->o_groups * (uint64_t)c->o_lora;
        uint64_t cols = (uint64_t)c->n_heads * (uint64_t)c->head_dim /
                        (uint64_t)c->o_groups;
        wo_a_layer = wo_a_values + ceil_div(rows, 128) * ceil_div(cols, 128);
    }
    uint64_t dspark_persistent = 0;
    if (getenv("DSV4_EXPERIMENTAL_DSPARK"))
        dspark_persistent = (uint64_t)c->dspark_stages * wo_a_layer
            + (uint64_t)c->dspark_stages * (uint64_t)c->window *
              (uint64_t)c->head_dim * sizeof(float);
    uint64_t runtime_reserve = RUNTIME_RESERVE + dspark_persistent;
    if (budget_bytes <= runtime_reserve + ctx) {
        if (verbose)
            fprintf(stderr,
                    "dsv4: memory plan refused: budget %.2f GiB cannot hold runtime "
                    "reserve %.2f GiB + context %.2f GiB + one expert slot (%.2f GiB)\n",
                    (double)budget_bytes / GIB, (double)runtime_reserve / GIB,
                    (double)ctx / GIB,
                    (double)EXPERT_SLOT_BYTES / GIB);
        return 0;
    }
    uint64_t avail = budget_bytes - runtime_reserve - ctx;
    uint64_t wo_a_layers = 0;
    if (wo_a_layer > 0 && avail > MIN_EXPERT_CACHE) {
        wo_a_layers = (avail - MIN_EXPERT_CACHE) / wo_a_layer;
        if (wo_a_layers > (uint64_t)c->n_layers) wo_a_layers = (uint64_t)c->n_layers;
    }
    uint64_t wo_a_bytes = wo_a_layers * wo_a_layer;
    uint64_t slots = (avail - wo_a_bytes) / EXPERT_SLOT_BYTES;
    if (slots < 1) {
        if (verbose)
            fprintf(stderr, "dsv4: memory plan refused: budget leaves room for %llu expert "
                            "slots, need at least 1\n", (unsigned long long)slots);
        return 0;
    }
    if (slots > 1u << 16) slots = 1u << 16;    /* sanity ceiling, far above any real use */

    plan->total_budget_bytes = budget_bytes;
    plan->runtime_reserve_bytes = runtime_reserve;
    plan->context_bytes = ctx;
    plan->wo_a_cache_bytes = wo_a_bytes;
    plan->expert_cache_bytes = slots * EXPERT_SLOT_BYTES;
    plan->planned_bytes = runtime_reserve + ctx + wo_a_bytes +
                          plan->expert_cache_bytes;
    plan->wo_a_cache_layers = (int)wo_a_layers;
    plan->expert_cache_slots = (int)slots;

    if (verbose)
        fprintf(stderr,
                "memory: budget %.2f GiB | runtime reserve %.2f GiB | context %d tokens "
                "(%.2f GiB) | wo_a %d layers (%.2f GiB) | %d expert slots (%.2f GiB) "
                "| planned %.2f GiB\n",
                (double)budget_bytes / GIB, (double)runtime_reserve / GIB,
                context, (double)ctx / GIB,
                plan->wo_a_cache_layers, (double)plan->wo_a_cache_bytes / GIB,
                plan->expert_cache_slots, (double)plan->expert_cache_bytes / GIB,
                (double)plan->planned_bytes / GIB);
    return 1;
}

int dsv4_memory_plan(const DSV4Config *c, int context, uint64_t budget_bytes,
                     DSV4MemoryPlan *plan)
{
    return memory_plan_impl(c, context, budget_bytes, plan, 1);
}

int dsv4_auto_context(const DSV4Config *c, uint64_t budget_bytes)
{
    if (!c || c->max_position < 1 || budget_bytes == 0) return 0;

    int ceiling = c->original_position > 0 ? c->original_position : c->max_position;
    if (ceiling > c->max_position) ceiling = c->max_position;
    if (ceiling < 1) return 0;

    int base_context = ceiling < 4096 ? ceiling : 4096;
    DSV4MemoryPlan base;
    while (base_context > 1 &&
           !memory_plan_impl(c, base_context, budget_bytes, &base, 0))
        base_context /= 2;
    if (!memory_plan_impl(c, base_context, budget_bytes, &base, 0)) return 0;

    int best = base_context;
    while (best < ceiling) {
        int candidate = best <= ceiling / 2 ? best * 2 : ceiling;
        DSV4MemoryPlan next;
        if (!memory_plan_impl(c, candidate, budget_bytes, &next, 0) ||
            next.wo_a_cache_layers != base.wo_a_cache_layers ||
            (int64_t)next.expert_cache_slots * 10 <
                (int64_t)base.expert_cache_slots * 9)
            break;
        best = candidate;
    }
    return best;
}
