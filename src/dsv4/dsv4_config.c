/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_config.c - build a DSV4Config from the official config.json.
 *
 * The released config is flat (no text_config nesting) and every field the engine
 * needs is REQUIRED: an absent field is an error, never a default. The one
 * deliberate exception is that synthetic test checkpoints may omit the DSpark
 * fields. In the released checkpoint, compress_ratios has 46 entries: the first
 * n_layers are the main-model ratios and the trailing zero entries identify the
 * three DSpark stages. The DSpark fields are accepted only as a complete group.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "json.h"

int dsv4_config_load_file(DSV4Config *c, const char *path)
{
    memset(c, 0, sizeof *c);

    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); fclose(f); return 0; }
    long n = ftell(f);
    if (n < 0 || n > (1L << 28)) {
        fprintf(stderr, "dsv4_config: %s: implausible size %ld\n", path, n);
        fclose(f); return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) { perror(path); fclose(f); return 0; }
    char *txt = (char *)malloc((size_t)n + 1);
    if (!txt) { fprintf(stderr, "dsv4_config: OOM reading %s\n", path); fclose(f); return 0; }
    size_t got = fread(txt, 1, (size_t)n, f);
    fclose(f);
    txt[got] = 0;

    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    if (!root) { fprintf(stderr, "dsv4_config: %s is not valid JSON\n", path); free(txt); return 0; }

    const char *missing[48];
    int nmissing = 0;
#define REQ_I(field, key) do { \
        jval *v = json_get(root, key); \
        if (!v || v->t != J_NUM) { missing[nmissing++] = key; } \
        else c->field = (int)v->num; \
    } while (0)
#define REQ_F(field, key) do { \
        jval *v = json_get(root, key); \
        if (!v || v->t != J_NUM) { missing[nmissing++] = key; } \
        else c->field = (float)v->num; \
    } while (0)

    REQ_I(hidden, "hidden_size");
    REQ_I(n_layers, "num_hidden_layers");
    REQ_I(vocab, "vocab_size");
    REQ_I(moe_inter, "moe_intermediate_size");
    REQ_I(n_experts, "n_routed_experts");
    REQ_I(n_shared, "n_shared_experts");
    REQ_I(topk, "num_experts_per_tok");
    REQ_I(n_hash_layers, "num_hash_layers");
    REQ_I(n_heads, "num_attention_heads");
    REQ_I(head_dim, "head_dim");
    REQ_I(q_lora, "q_lora_rank");
    REQ_I(o_groups, "o_groups");
    REQ_I(o_lora, "o_lora_rank");
    REQ_I(window, "sliding_window");
    REQ_I(index_heads, "index_n_heads");
    REQ_I(index_dim, "index_head_dim");
    REQ_I(index_topk, "index_topk");
    REQ_I(hc_mult, "hc_mult");
    REQ_I(hc_iters, "hc_sinkhorn_iters");
    REQ_I(max_position, "max_position_embeddings");
    REQ_F(rope_theta, "rope_theta");
    REQ_F(compress_rope_theta, "compress_rope_theta");
    REQ_F(rms_eps, "rms_norm_eps");
    REQ_F(hc_eps, "hc_eps");
    REQ_F(route_scale, "routed_scaling_factor");
    REQ_F(swiglu_limit, "swiglu_limit");
#undef REQ_I
#undef REQ_F

    c->rope_dim = 64;   /* qk_rope_head_dim; the released model ships 64 */
    {
        jval *v = json_get(root, "qk_rope_head_dim");
        if (v && v->t == J_NUM) c->rope_dim = (int)v->num;
    }
    c->original_position = 65536;

    /* nested rope_scaling */
    jval *rs = json_get(root, "rope_scaling");
    if (rs && rs->t == J_OBJ) {
        jval *v;
        v = json_get(rs, "factor");
        if (v && v->t == J_NUM) c->rope_factor = (float)v->num;
        v = json_get(rs, "beta_fast");
        if (v && v->t == J_NUM) c->beta_fast = (float)v->num;
        v = json_get(rs, "beta_slow");
        if (v && v->t == J_NUM) c->beta_slow = (float)v->num;
        v = json_get(rs, "original_max_position_embeddings");
        if (v && v->t == J_NUM) c->original_position = (int)v->num;
    }

    /* compress_ratios: first n_layers for the main model; a zero-valued tail
     * contains one entry per DSpark stage. */
    jval *cr = json_get(root, "compress_ratios");
    if (!cr || cr->t != J_ARR || cr->len < c->n_layers) {
        missing[nmissing++] = "compress_ratios";
    } else {
        for (int i = 0; i < c->n_layers && i < DSV4_MAX_LAYERS; i++) {
            if (cr->kids[i]->t != J_NUM) { missing[nmissing++] = "compress_ratios[i]"; break; }
            c->compress_ratio[i] = (int)cr->kids[i]->num;
        }
        c->dspark_stages = cr->len - c->n_layers;
        if (c->dspark_stages > DSV4_MAX_DSPARK_STAGES) {
            fprintf(stderr, "dsv4_config: %s has %d DSpark stages (max %d)\n",
                    path, c->dspark_stages, DSV4_MAX_DSPARK_STAGES);
            free(txt); json_free(root); return 0;
        }
        for (int i = c->n_layers; i < cr->len; i++) {
            if (cr->kids[i]->t != J_NUM || (int)cr->kids[i]->num != 0) {
                fprintf(stderr, "dsv4_config: %s: compress_ratios[%d] (past layer %d) is "
                        "nonzero; DSpark stages require uncompressed attention\n", path, i,
                        c->n_layers);
                free(txt); json_free(root); return 0;
            }
        }
    }

    /* DSpark is optional only for small synthetic checkpoints. A real draft
     * configuration must be complete; silently filling one missing member would
     * describe a different speculative model. */
    jval *ds_block = json_get(root, "dspark_block_size");
    jval *ds_noise = json_get(root, "dspark_noise_token_id");
    jval *ds_layers = json_get(root, "dspark_target_layer_ids");
    jval *ds_rank = json_get(root, "dspark_markov_rank");
    int ds_any = ds_block || ds_noise || ds_layers || ds_rank;
    if (ds_any) {
        if (!ds_block || ds_block->t != J_NUM) missing[nmissing++] = "dspark_block_size";
        else c->dspark_block_size = (int)ds_block->num;
        if (!ds_noise || ds_noise->t != J_NUM) missing[nmissing++] = "dspark_noise_token_id";
        else c->dspark_noise_token = (int)ds_noise->num;
        if (!ds_rank || ds_rank->t != J_NUM) missing[nmissing++] = "dspark_markov_rank";
        else c->dspark_markov_rank = (int)ds_rank->num;
        if (!ds_layers || ds_layers->t != J_ARR ||
            ds_layers->len != c->dspark_stages) {
            missing[nmissing++] = "dspark_target_layer_ids";
        } else {
            for (int i = 0; i < ds_layers->len; i++) {
                if (ds_layers->kids[i]->t != J_NUM) {
                    missing[nmissing++] = "dspark_target_layer_ids[i]";
                    break;
                }
                c->dspark_target_layer[i] = (int)ds_layers->kids[i]->num;
            }
        }
    } else if (c->dspark_stages != 0) {
        missing[nmissing++] = "dspark_*";
    }

    if (nmissing) {
        fprintf(stderr, "dsv4_config: %s is missing %d required field(s):\n", path, nmissing);
        for (int i = 0; i < nmissing && i < (int)(sizeof missing / sizeof missing[0]); i++)
            fprintf(stderr, "    %s\n", missing[i]);
        fprintf(stderr, "  refusing to substitute defaults: a half-read config would\n"
                        "  silently produce a DIFFERENT model.\n");
        free(txt); json_free(root); return 0;
    }

    /* structural checks */
    if (c->n_layers <= 0 || c->n_layers > DSV4_MAX_LAYERS) {
        fprintf(stderr, "dsv4_config: %s has %d layers (max %d)\n", path, c->n_layers,
                DSV4_MAX_LAYERS);
        free(txt); json_free(root); return 0;
    }
    if (c->topk <= 0 || c->topk > DSV4_MAX_TOPK) {
        fprintf(stderr, "dsv4_config: %s selects top-%d (max %d)\n", path, c->topk,
                DSV4_MAX_TOPK);
        free(txt); json_free(root); return 0;
    }
    if (c->topk > c->n_experts) {
        fprintf(stderr, "dsv4_config: %s selects %d of %d experts\n",
                path, c->topk, c->n_experts);
        free(txt); json_free(root); return 0;
    }
    if (c->hc_mult <= 0 || c->hc_mult > 8) {
        fprintf(stderr, "dsv4_config: %s hc_mult %d is unsupported\n", path, c->hc_mult);
        free(txt); json_free(root); return 0;
    }
    if (c->window <= 0) {
        fprintf(stderr, "dsv4_config: %s sliding_window %d\n", path, c->window);
        free(txt); json_free(root); return 0;
    }
    if (c->dspark_stages) {
        if (c->dspark_block_size < 2 || c->dspark_block_size > 32) {
            fprintf(stderr, "dsv4_config: %s DSpark block size %d is unsupported\n",
                    path, c->dspark_block_size);
            free(txt); json_free(root); return 0;
        }
        if (c->dspark_noise_token < 0 || c->dspark_noise_token >= c->vocab) {
            fprintf(stderr, "dsv4_config: %s DSpark noise token %d is outside vocab %d\n",
                    path, c->dspark_noise_token, c->vocab);
            free(txt); json_free(root); return 0;
        }
        if (c->dspark_markov_rank <= 0) {
            fprintf(stderr, "dsv4_config: %s DSpark Markov rank %d is invalid\n",
                    path, c->dspark_markov_rank);
            free(txt); json_free(root); return 0;
        }
        for (int i = 0; i < c->dspark_stages; i++) {
            int layer = c->dspark_target_layer[i];
            if (layer < 0 || layer >= c->n_layers ||
                (i > 0 && layer <= c->dspark_target_layer[i - 1])) {
                fprintf(stderr, "dsv4_config: %s DSpark target layer %d at index %d "
                        "is out of range or not strictly increasing\n", path, layer, i);
                free(txt); json_free(root); return 0;
            }
        }
    }

    free(txt);
    json_free(root);
    return 1;
}
