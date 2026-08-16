#!/usr/bin/env python3
"""test_dsv4_tokenizer_parity.py - verify the C tokenizer against the released
tokenizer.json through tokenizers==0.22.0.

Three coverage sets, all compared token-for-token:
  1. 22 fixed samples (empty/space/CJK/emoji/ZWJ/accents/CRLF/control chars).
  2. 1,000 deterministic mixed-Unicode samples.
  3. every single Unicode P (punctuation) and S (symbol) codepoint.

The C tokenizer is run in batches of 8 samples (never 1,022 argv entries in one
process), reading stdin lines and emitting IDs. tokenizers==0.22.0 is the oracle.

Usage: python3 tools/test_dsv4_tokenizer_parity.py [MODEL_DIR]
"""
import random
import subprocess
import sys
import unicodedata
from tokenizers import Tokenizer

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else "model/DeepSeek-V4-Flash-0731"
CLI = "bin/dsv4_tok_cli"

FIXED = [
    "",
    "a",
    "Hello, world!",
    "   leading and trailing spaces   ",
    "你好，世界。こんにちは",
    "emoji 😀🚀🎉 and more 😃",
    "combining café résumé naïve",
    "CRLF\r\nand\r\nmore\nlines",
    "zero-width\u200bjoiner\u200dsequence",
    "tab\tand\vvertical",
    "CJK punctuation ，。！？；：",
    "acronyms IBM NASA OpenAI",
    "apostrophes don't can't we'll",
    "numbers 1234567890 and 3.14159",
    "mixed 中文 and English and 123",
    "dashes—em and–en and-hyphen",
    "currency $€£¥₹ and %",
    "quotes 'single' \"double\" `backtick`",
    "parentheses (brackets) [square] {curly}",
    "slash / and \\ backslash",
    "newline only\n",
    "the quick brown fox jumps over the lazy dog",
]


def mixed_unicode_samples(n, seed=0):
    random.seed(seed)
    planes = [
        (0x0000, 0x024F),   # Basic Latin + Latin-1
        (0x0400, 0x04FF),   # Cyrillic
        (0x0900, 0x097F),   # Devanagari
        (0x3040, 0x30FF),   # Hiragana + Katakana
        (0x4E00, 0x9FFF),   # CJK Unified
        (0xAC00, 0xD7AF),   # Hangul
        (0x1F300, 0x1F9FF), # emoji
        (0x2000, 0x206F),   # punctuation
    ]
    samples = []
    for _ in range(n):
        k = random.randint(1, 8)
        chars = []
        for _ in range(k):
            lo, hi = random.choice(planes)
            chars.append(chr(random.randint(lo, hi)))
        samples.append("".join(chars))
    return samples


def ps_codepoints():
    for cp in range(0x110000):
        try:
            cat = unicodedata.category(chr(cp))
        except ValueError:
            continue
        if cat and cat[0] in "PS":
            yield chr(cp)


def main():
    oracle = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
    samples = FIXED + mixed_unicode_samples(1000) + list(ps_codepoints())
    total = 0
    errors = 0
    batch = 8
    # Keep ONE resident C process and stream batches over stdin/stdout, so the
    # 6 MB tokenizer is loaded once rather than once per batch.
    p = subprocess.Popen([CLI, MODEL_DIR], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE)
    for i in range(0, len(samples), batch):
        chunk = samples[i:i + batch]
        for text in chunk:
            raw = text.encode("utf-8")
            p.stdin.write(str(len(raw)).encode() + b"\n" + raw)
        p.stdin.flush()
        got = [p.stdout.readline().decode().rstrip("\n") for _ in chunk]
        for text, line in zip(chunk, got):
            expected = oracle.encode(text).ids
            actual = [int(x) for x in line.split()] if line.strip() else []
            total += 1
            if actual != expected:
                errors += 1
                if errors <= 20:
                    print(f"MISMATCH {text!r}\n  expected {expected}\n  actual   {actual}",
                          file=sys.stderr)
    p.stdin.close()
    p.stdout.close()
    p.wait()
    if errors:
        print(f"FAIL: {errors}/{total} samples mismatch", file=sys.stderr)
        return 1
    print(f"tokenizer parity: OK - {total} samples (22 fixed + 1000 mixed + {len(samples) - 1022} P/S) all match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
