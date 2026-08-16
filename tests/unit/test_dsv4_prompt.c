/* SPDX-License-Identifier: Apache-2.0 */
/* test_dsv4_prompt.c - chat template byte-level tests. The "a" prompt must
 * reproduce the fixed oracle: SHA-256 453281a61e27a36aa728a5028d12b50021cfa5d1
 * 7234a68d81637c73ace37ea3 and token IDs 0 128803 67 128804 128822 (verified in
 * tools/test_dsv4_tokenizer_parity.py against the real tokenizer). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } } while (0)

int main(void)
{
    /* basic "a" prompt, exact bytes */
    const char *want = "<｜begin▁of▁sentence｜><｜User｜>a<｜Assistant｜></think>";
    int need = dsv4_format_prompt(NULL, 0, "a", NULL, NULL);
    CHECK(need == (int)strlen(want), "size query matches");
    char buf[512];
    int n = dsv4_format_prompt(buf, sizeof buf, "a", NULL, NULL);
    CHECK(n == (int)strlen(want), "rendered size");
    CHECK(strcmp(buf, want) == 0, "exact bytes for 'a'");

    /* thinking mode ends with <think> */
    n = dsv4_format_prompt(buf, sizeof buf, "a", NULL, "low");
    CHECK(strstr(buf, "<think>") != NULL, "thinking ends with <think>");
    CHECK(strstr(buf, "</think>") == NULL, "no closing think in thinking mode");

    /* system prompt appears after the BOS + prefix, before <|User|> */
    n = dsv4_format_prompt(buf, sizeof buf, "hi", "sys", NULL);
    CHECK(strstr(buf, "<｜begin▁of▁sentence｜>sys<｜User｜>hi<｜Assistant｜></think>") != NULL,
          "system position");

    /* reasoning effort prefix (high) */
    n = dsv4_format_prompt(buf, sizeof buf, "hi", NULL, "high");
    CHECK(strstr(buf, "Think step by step to solve the problem.") != NULL,
          "high effort prefix present");
    n = dsv4_format_prompt(buf, sizeof buf, "hi", NULL, "low");
    CHECK(strstr(buf, "Think step by step but keep your reasoning concise") != NULL,
          "low effort prefix present");

    /* buffer capacity: truncation must still NUL-terminate */
    char small[8];
    n = dsv4_format_prompt(small, sizeof small, "a", NULL, NULL);
    CHECK(small[sizeof small - 1] == '\0', "truncated buffer NUL-terminated");
    CHECK(n == (int)strlen(want), "needed size still returned");

    /* Later turns keep the existing BOS/system/history and append only the
     * next role markers. The previous assistant EOS is supplied as token 1 by
     * the interactive caller, not duplicated in this byte template. */
    const char *turn = "<｜User｜>next<｜Assistant｜></think>";
    n = dsv4_format_turn(buf, sizeof buf, "next", NULL);
    CHECK(n == (int)strlen(turn), "later turn size");
    CHECK(strcmp(buf, turn) == 0, "later turn exact bytes");
    n = dsv4_format_turn(buf, sizeof buf, "next", "high");
    CHECK(strstr(buf, "<think>") != NULL, "later thinking turn opens think");
    CHECK(strstr(buf, "</think>") == NULL, "later thinking turn has no close");

    /* empty user refuses */
    CHECK(dsv4_format_prompt(buf, sizeof buf, "", NULL, NULL) < 0, "empty user refuses");
    CHECK(dsv4_format_turn(buf, sizeof buf, "", NULL) < 0, "empty later turn refuses");

    if (fails) { printf("%d FAILURES\n", fails); return 1; }
    printf("test_dsv4_prompt: PASS\n");
    return 0;
}
