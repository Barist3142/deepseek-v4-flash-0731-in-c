/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_tok_cli.c - minimal tokenizer CLI for external parity testing.
 *
 * Length-prefixed protocol on stdin (samples may contain newlines, so a plain
 * line protocol would corrupt): for each sample, a decimal byte length on its
 * own line followed by exactly that many raw UTF-8 bytes. Output is one line of
 * space-separated token IDs per sample.
 *
 * All reads are raw read(2); mixing stdio getline with read(2) on one descriptor
 * would let the stdio buffer swallow part of the sample payload.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#include "tok.h"
#pragma GCC diagnostic pop

/* Read until newline into a growable buffer; returns bytes read or -1 on EOF. */
static ssize_t read_line(int fd, char **buf, size_t *cap)
{
    size_t n = 0;
    for (;;) {
        if (n + 1 >= *cap) {
            *cap = *cap ? *cap * 2 : 256;
            *buf = (char *)realloc(*buf, *cap);
            if (!*buf) exit(1);
        }
        char c;
        ssize_t r = read(fd, &c, 1);
        if (r <= 0) return n ? (ssize_t)n : -1;
        if (c == '\n') { (*buf)[n] = 0; return (ssize_t)n; }
        (*buf)[n++] = c;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL_DIR\n", argv[0]); return 2; }
    char path[4096];
    snprintf(path, sizeof path, "%s/tokenizer.json", argv[1]);
    Tok T;
    tok_load(&T, path);

    char *lenbuf = NULL;
    size_t len_cap = 0;
    char *buf = NULL;
    size_t buf_cap = 0;
    int *ids = (int *)malloc(65536 * sizeof(int));
    setvbuf(stdout, NULL, _IOLBF, 0);

    for (;;) {
        ssize_t ln = read_line(STDIN_FILENO, &lenbuf, &len_cap);
        if (ln < 0) break;
        long l = strtol(lenbuf, NULL, 10);
        if (l < 0) l = 0;
        if ((size_t)l + 1 > buf_cap) {
            buf_cap = (size_t)l + 1;
            buf = (char *)realloc(buf, buf_cap);
        }
        size_t got = 0;
        while (got < (size_t)l) {
            ssize_t r = read(STDIN_FILENO, buf + got, (size_t)l - got);
            if (r <= 0) break;
            got += (size_t)r;
        }
        buf[got] = 0;
        int m = tok_encode(&T, buf, (int)got, ids, 65536);
        for (int i = 0; i < m; i++) {
            if (i) printf(" ");
            printf("%d", ids[i]);
        }
        printf("\n");
    }
    free(ids);
    free(lenbuf);
    free(buf);
    return 0;
}
