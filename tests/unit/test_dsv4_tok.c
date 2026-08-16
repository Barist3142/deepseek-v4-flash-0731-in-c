/* SPDX-License-Identifier: Apache-2.0 */
/* test_dsv4_tok.c - DeepSeek tokenizer tests against the official
 * tokenizer.json. The full parity suite lives in tools/test_dsv4_tokenizer_
 * parity.py (9,384 samples vs tokenizers==0.22.0); this test pins the oracle
 * prompt and a few tricky cases in C. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tok.h"
#pragma GCC diagnostic pop

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(int argc, char **argv)
{
    const char *model_dir = argc > 1 ? argv[1] : "model/DeepSeek-V4-Flash-0731";
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", model_dir);
    FILE *probe = fopen(path, "rb");
    if (!probe) {
        printf("test_dsv4_tok: SKIPPED (tokenizer.json unavailable at %s)\n",
               path);
        return 0;
    }
    fclose(probe);
    Tok T;
    tok_load(&T, path);

    /* oracle prompt */
    const char *prompt = "<｜begin▁of▁sentence｜><｜User｜>a<｜Assistant｜></think>";
    int ids[32];
    int n = tok_encode(&T, prompt, (int)strlen(prompt), ids, 32);
    CHECK(n == 5, "prompt has 5 tokens");
    if (n == 5) {
        static const int want[5] = { 0, 128803, 67, 128804, 128822 };
        for (int i = 0; i < 5; i++) CHECK(ids[i] == want[i], "oracle prompt ids");
    }

    /* "a" alone */
    n = tok_encode(&T, "a", 1, ids, 32);
    CHECK(n == 1 && ids[0] == 67, "a -> 67");

    /* digits: \p{N}{1,3} */
    n = tok_encode(&T, "123456", 6, ids, 32);
    CHECK(n == 2, "123456 -> 2 pieces");

    /* CJK run */
    n = tok_encode(&T, "你好", (int)strlen("你好"), ids, 32);
    CHECK(n >= 1, "CJK encodes");

    /* EOS added token */
    int eos = tok_id_of(&T, "<｜end▁of▁sentence｜>");
    CHECK(eos == 1, "EOS id 1");

    /* decode roundtrip */
    char out[64];
    int m = tok_decode(&T, ids, n, out, sizeof out);
    CHECK(m > 0, "decode produces text");

    /* raw bytes: byte-level BPE keeps spaces */
    n = tok_encode(&T, "hello world", 11, ids, 32);
    CHECK(n == 2, "hello world -> 2 tokens");
    m = tok_decode(&T, ids, n, out, sizeof out);
    CHECK(strcmp(out, "hello world") == 0, "decode roundtrip hello world");

    /* Streaming one token at a time must produce the same byte stream as a
     * whole-sequence decode, including tokens that split a UTF-8 character. */
    {
        const char *sample = "Hello, 世界!";
        int stream_ids[32];
        char whole[128], streamed[128], piece[128];
        int sn = tok_encode(&T, sample, (int)strlen(sample), stream_ids, 32);
        int wn = tok_decode(&T, stream_ids, sn, whole, (int)sizeof whole - 1);
        int so = 0;
        for (int i = 0; i < sn; i++) {
            int pn = tok_decode(&T, stream_ids + i, 1, piece,
                                (int)sizeof piece - 1);
            memcpy(streamed + so, piece, (size_t)pn);
            so += pn;
        }
        streamed[so] = 0;
        CHECK(so == wn && memcmp(streamed, whole, (size_t)wn) == 0,
              "tokenwise decode equals whole decode");
    }

    /* punctuation symbol split */
    n = tok_encode(&T, "!!!", 3, ids, 32);
    CHECK(n == 1, "!!! is one symbol run");

    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("test_dsv4_tok: PASS\n");
    return 0;
}
