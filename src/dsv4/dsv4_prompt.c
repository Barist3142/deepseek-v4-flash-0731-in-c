/* SPDX-License-Identifier: Apache-2.0 */
/*
 * dsv4_prompt.c - single-turn chat template.
 *
 * Exact concatenation (verified byte-for-byte against the fixed ModelScope
 * revision; the prompt for user text "a" renders to token IDs
 * 0 128803 67 128804 128822 and its SHA-256 is
 * 453281a61e27a36aa728a5028d12b50021cfa5d17234a68d81637c73ace37ea3):
 *
 *   <|begin_of_sentence|>
 *   + optional reasoning-effort instruction prefix
 *   + optional system prompt
 *   + <|User|>
 *   + user prompt
 *   + <|Assistant|>
 *   + "</think>"                  (ordinary chat)
 *
 * Thinking mode (effort = "low" | "high" | "max") replaces the final "</think>"
 * with "<think>" and prepends the official reasoning instruction for the
 * requested effort. "high" and "max" share the same prefix.
 *
 * The user/assistant markers are the ACTUAL added tokens from the official
 * tokenizer ("<｜User｜>", "<｜Assistant｜>"); this file must stay in byte-level
 * sync with tokenizer.json.
 */
#include <stdio.h>
#include <string.h>

#include "dsv4.h"

#define BOS       "<｜begin▁of▁sentence｜>"
#define USER_TAG  "<｜User｜>"
#define ASSISTANT "<｜Assistant｜>"
#define THINK_END "</think>"
#define THINK_OPEN "<think>"

/* Official reasoning-effort instruction prefixes (DeepSeek-AI released
 * implementation). "max" shares the "high" prefix. */
static const char *effort_prefix(const char *effort)
{
    if (!effort) return "";
    if (!strcmp(effort, "low"))
        return "You are a helpful assistant. Think step by step but keep your "
               "reasoning concise and to the point. Only show your reasoning when "
               "necessary.\n\n";
    if (!strcmp(effort, "high") || !strcmp(effort, "max"))
        return "You are a helpful assistant. Think step by step to solve the "
               "problem. Show your reasoning.\n\n";
    return "";
}

static void append(char *buf, size_t cap, size_t *off, size_t *need,
                   const char *s)
{
    size_t n = strlen(s);
    *need += n;
    if (*off < cap) {
        size_t take = n;
        if (*off + take > cap) take = cap - *off;
        memcpy(buf + *off, s, take);
        *off += take;
        if (cap) buf[*off < cap ? *off : cap - 1] = '\0';
    }
}

int dsv4_format_prompt(char *buf, size_t cap, const char *user,
                       const char *system, const char *effort)
{
    const char *usr = user ? user : "";
    const char *sys = system ? system : "";
    if (!usr[0]) return -1;
    const int thinking = effort && effort[0] && strcmp(effort, "none");

    size_t off = 0, need = 0;
    append(buf, cap, &off, &need, BOS);
    if (thinking) append(buf, cap, &off, &need, effort_prefix(effort));
    append(buf, cap, &off, &need, sys);
    append(buf, cap, &off, &need, USER_TAG);
    append(buf, cap, &off, &need, usr);
    append(buf, cap, &off, &need, ASSISTANT);
    append(buf, cap, &off, &need, thinking ? THINK_OPEN : THINK_END);
    if (buf && cap) buf[off < cap ? off : cap - 1] = '\0';
    return (int)need;
}

int dsv4_format_turn(char *buf, size_t cap, const char *user,
                     const char *effort)
{
    const char *usr = user ? user : "";
    if (!usr[0]) return -1;
    const int thinking = effort && effort[0] && strcmp(effort, "none");

    size_t off = 0, need = 0;
    append(buf, cap, &off, &need, USER_TAG);
    append(buf, cap, &off, &need, usr);
    append(buf, cap, &off, &need, ASSISTANT);
    append(buf, cap, &off, &need, thinking ? THINK_OPEN : THINK_END);
    if (buf && cap) buf[off < cap ? off : cap - 1] = '\0';
    return (int)need;
}
