#!/usr/bin/env python3
"""make_tiny_dsv4.py - generate a deterministic non-zero 4-layer checkpoint.

The fixture uses the SAME tensor names, dtypes, FP8/FP4 block shapes and
contiguous expert scale/weight layout as the official checkpoint, but at tiny
widths so the whole forward pass is a hand-computable scalar exercise:

  layers 4 | hidden 8 | vocab 16 | heads 2 x head_dim 4 | rope_dim 2
  q_lora 4 | o_groups 2 / o_lora 2 | experts 4 top-2 | shared 1
  moe_inter 4 | hc_mult 2 | ratios 0,0,4,128 | window 4
  index_heads 2 | index_dim 8 | index_topk 4

Weights are nonzero and distinguish row/layer/expert. Output goes to
tests/fixtures/tiny_dsv4/ as model-00001-of-00001.safetensors.

Usage: python3 tools/make_tiny_dsv4.py [outdir]
"""
import json
import math
import os
import struct
import sys

# ------------------------------------------------------------------ config --
def cfg_dict():
    return {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "hidden_size": 8,
        "num_hidden_layers": 4,
        "vocab_size": 16,
        "max_position_embeddings": 512,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "qk_rope_head_dim": 2,
        "q_lora_rank": 4,
        "o_groups": 2,
        "o_lora_rank": 2,
        "sliding_window": 4,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "num_hash_layers": 0,
        "moe_intermediate_size": 4,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-6,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-6,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {
            "factor": 16.0, "beta_fast": 32.0, "beta_slow": 1.0,
            "original_max_position_embeddings": 64,
            "type": "yarn",
        },
        "index_n_heads": 2,
        "index_head_dim": 8,
        "index_topk": 4,
        "compress_ratios": [0, 0, 4, 128],
    }


# ---------------------------------------------------------------- weights --
class Pack:
    def __init__(self, name, dtype, shape):
        self.name = name
        self.dtype = dtype
        self.shape = tuple(shape)
        self.n = 1
        for d in shape:
            self.n *= d

    def assign(self, data):  # data: list/tuple of floats or bytes
        self.data = data
        return self

    def bytes(self):
        if self.dtype == "F32":
            return struct.pack("<%df" % self.n, *self.data)
        if self.dtype == "BF16":
            return b"".join(f2bf16(v) for v in self.data)
        if self.dtype == "F8_E4M3":
            return bytes(self.data)
        if self.dtype == "F8_E8M0":
            return bytes(self.data)
        if self.dtype == "I8":
            return bytes(self.data)
        if self.dtype == "I64":
            return struct.pack("<%dq" % self.n, *self.data)
        raise ValueError(self.dtype)


def f2bf16(v):
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return struct.pack("<H", u >> 16)


def bf16vals(values):
    return [v for v in values]


def fp8_e4m3(v):
    """nearest E4M3 code for a float in [-448, 448]."""
    v = max(-448.0, min(448.0, v))
    if v == 0:
        return 0
    s = 1 if v < 0 else 0
    a = abs(v)
    # exponent
    e = math.floor(math.log2(a))
    # mantissa to 3 bits
    m = a / (2 ** e)
    mi = round(m * 8)
    if mi >= 16:
        mi = 8
        e += 1
    e = max(-9, min(8, e))
    if e == -9:
        code = mi & 7          # subnormal: mantissa only
    else:
        code = ((e + 9) << 3) | (mi & 7)
    return (s << 7) | code


def fp4_e2m1(v):
    """nearest E2M1 nibble for a float in [-6, 6]."""
    vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    v = max(-6.0, min(6.0, v))
    s = 1 if v < 0 else 0
    best = 0
    bd = abs(v)
    for i, x in enumerate(vals):
        d = abs(abs(v) - x)
        if d < bd:
            bd = d
            best = i
    return (s << 3) | best


def pack_fp4(values, cols):
    """values: row-major floats; cols logical columns; returns byte array."""
    out = bytearray()
    for row in values:
        for c in range(0, len(row), 2):
            lo = fp4_e2m1(row[c])
            hi = fp4_e2m1(row[c + 1])
            out.append((hi << 4) | lo)
    return bytes(out)


def scale_e8m0(v):
    """power-of-two E8M0 code: smallest 2^e >= v."""
    if v <= 0:
        return 127
    e = math.ceil(math.log2(v))
    return e + 127


def block_scale_fp8(rows, cols, block=128):
    """E8M0 scale table [ceil(rows/128)][ceil(cols/128)] all ones."""
    out = []
    for r in range((rows + block - 1) // block):
        for c in range((cols + block - 1) // block):
            out.append(scale_e8m0(1.0))
    return out


def block_scale_fp4(rows, cols, group=32):
    out = []
    for r in range(rows):
        for c in range((cols + group - 1) // group):
            out.append(scale_e8m0(1.0))
    return out


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tiny_dsv4"
    os.makedirs(outdir, exist_ok=True)
    H, L, V = 8, 4, 16
    heads, hd, rd, ql, G, ol = 2, 4, 2, 4, 2, 2
    ne, topk, mi = 4, 2, 4
    mult = 2
    mix = (2 + mult) * mult
    ratio_layers = {2: 4, 3: 128}
    index_heads, index_dim, index_topk = 2, 8, 4

    def w(name, dtype, shape, gen):
        p = Pack(name, dtype, shape)
        p.assign(gen(p.n))
        return p

    def lin(rows, cols, base):
        # BF16 matrix with row/layer-distinct values in [-2, 2]
        vals = []
        for r in range(rows):
            for c in range(cols):
                vals.append(((r + 1) * 0.7 + c * 0.11 + base) % 4.0 - 2.0)
        return vals

    def f32vec(n, base):
        return [((i + 1) * 0.37 + base) % 2.0 - 1.0 for i in range(n)]

    tensors = []

    tensors.append(w("embed.weight", "BF16", [V, H], lambda n: lin(V, H, 0.3)))
    tensors.append(w("head.weight", "BF16", [V, H], lambda n: lin(V, H, 1.7)))
    tensors.append(w("norm.weight", "BF16", [H], lambda n: f32vec(H, 0.1)))
    tensors.append(w("hc_head_fn", "F32", [mult, mult * H], lambda n: f32vec(n, 0.9)))
    tensors.append(w("hc_head_base", "F32", [mult], lambda n: f32vec(n, 0.2)))
    tensors.append(w("hc_head_scale", "F32", [1], lambda n: [1.0]))

    for L in range(L):
        base = L * 1.3
        tensors.append(w(f"layers.{L}.attn_norm.weight", "BF16", [H], lambda n, b=base: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.ffn_norm.weight", "BF16", [H], lambda n, b=base + 0.4: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.hc_attn_fn", "F32", [mix, mult * H], lambda n, b=base: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.hc_ffn_fn", "F32", [mix, mult * H], lambda n, b=base + 0.6: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.hc_attn_scale", "F32", [3], lambda n: [1.0, 1.0, 1.0]))
        tensors.append(w(f"layers.{L}.hc_ffn_scale", "F32", [3], lambda n: [1.0, 1.0, 1.0]))
        tensors.append(w(f"layers.{L}.hc_attn_base", "F32", [mix], lambda n, b=base + 0.2: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.hc_ffn_base", "F32", [mix], lambda n, b=base + 0.8: f32vec(n, b)))

        tensors.append(w(f"layers.{L}.attn.attn_sink", "F32", [heads], lambda n, b=base: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.attn.wq_a.weight", "F8_E4M3", [ql, H], lambda n, b=base: [fp8_e4m3((i * 0.29 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.attn.wq_a.scale", "F8_E8M0", [(ql + 127) // 128, (H + 127) // 128], lambda n: block_scale_fp8(ql, H)))
        tensors.append(w(f"layers.{L}.attn.q_norm.weight", "BF16", [ql], lambda n, b=base: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.attn.wq_b.weight", "F8_E4M3", [heads * hd, ql], lambda n, b=base: [fp8_e4m3((i * 0.31 + b) % 3.0 - 1.5) for i in range(n)]))
        tensors.append(w(f"layers.{L}.attn.wq_b.scale", "F8_E8M0", [(heads * hd + 127) // 128, (ql + 127) // 128], lambda n: block_scale_fp8(heads * hd, ql)))
        tensors.append(w(f"layers.{L}.attn.wkv.weight", "F8_E4M3", [hd, H], lambda n, b=base: [fp8_e4m3((i * 0.33 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.attn.wkv.scale", "F8_E8M0", [(hd + 127) // 128, (H + 127) // 128], lambda n: block_scale_fp8(hd, H)))
        tensors.append(w(f"layers.{L}.attn.kv_norm.weight", "BF16", [hd], lambda n, b=base: f32vec(n, b)))
        tensors.append(w(f"layers.{L}.attn.wo_a.weight", "F8_E4M3", [G * ol, heads * hd // G], lambda n, b=base: [fp8_e4m3((i * 0.19 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.attn.wo_a.scale", "F8_E8M0", [(G * ol + 127) // 128, (heads * hd // G + 127) // 128], lambda n: block_scale_fp8(G * ol, heads * hd // G)))
        tensors.append(w(f"layers.{L}.attn.wo_b.weight", "F8_E4M3", [H, G * ol], lambda n, b=base: [fp8_e4m3((i * 0.27 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.attn.wo_b.scale", "F8_E8M0", [(H + 127) // 128, (G * ol + 127) // 128], lambda n: block_scale_fp8(H, G * ol)))

        ratio = ratio_layers.get(L, 0)
        if ratio:
            coff = 2 if ratio == 4 else 1
            tensors.append(w(f"layers.{L}.attn.compressor.ape", "F32", [ratio, coff * hd], lambda n, b=base: f32vec(n, b)))
            tensors.append(w(f"layers.{L}.attn.compressor.wkv.weight", "BF16", [coff * hd, H], lambda n, b=base: lin(coff * hd, H, b)))
            tensors.append(w(f"layers.{L}.attn.compressor.wgate.weight", "BF16", [coff * hd, H], lambda n, b=base + 0.3: lin(coff * hd, H, b)))
            tensors.append(w(f"layers.{L}.attn.compressor.norm.weight", "BF16", [hd], lambda n, b=base: f32vec(n, b)))
            if ratio == 4:
                tensors.append(w(f"layers.{L}.attn.indexer.wq_b.weight", "F8_E4M3", [index_heads * index_dim, ql], lambda n, b=base: [fp8_e4m3((i * 0.23 + b) % 2.0 - 1.0) for i in range(n)]))
                tensors.append(w(f"layers.{L}.attn.indexer.wq_b.scale", "F8_E8M0", [(index_heads * index_dim + 127) // 128, (ql + 127) // 128], lambda n: block_scale_fp8(index_heads * index_dim, ql)))
                tensors.append(w(f"layers.{L}.attn.indexer.weights_proj.weight", "BF16", [index_heads, H], lambda n, b=base: lin(index_heads, H, b)))
                tensors.append(w(f"layers.{L}.attn.indexer.compressor.ape", "F32", [ratio, coff * index_dim], lambda n, b=base: f32vec(n, b)))
                tensors.append(w(f"layers.{L}.attn.indexer.compressor.wkv.weight", "BF16", [coff * index_dim, H], lambda n, b=base: lin(coff * index_dim, H, b)))
                tensors.append(w(f"layers.{L}.attn.indexer.compressor.wgate.weight", "BF16", [coff * index_dim, H], lambda n, b=base + 0.3: lin(coff * index_dim, H, b)))
                tensors.append(w(f"layers.{L}.attn.indexer.compressor.norm.weight", "BF16", [index_dim], lambda n, b=base: f32vec(n, b)))

        tensors.append(w(f"layers.{L}.ffn.gate.weight", "BF16", [ne, H], lambda n, b=base: lin(ne, H, b)))
        tensors.append(w(f"layers.{L}.ffn.gate.bias", "F32", [ne], lambda n, b=base: f32vec(n, b)))
        # shared expert FP8
        tensors.append(w(f"layers.{L}.ffn.shared_experts.w1.weight", "F8_E4M3", [mi, H], lambda n, b=base: [fp8_e4m3((i * 0.41 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.ffn.shared_experts.w1.scale", "F8_E8M0", [(mi + 127) // 128, (H + 127) // 128], lambda n: block_scale_fp8(mi, H)))
        tensors.append(w(f"layers.{L}.ffn.shared_experts.w3.weight", "F8_E4M3", [mi, H], lambda n, b=base + 0.5: [fp8_e4m3((i * 0.41 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.ffn.shared_experts.w3.scale", "F8_E8M0", [(mi + 127) // 128, (H + 127) // 128], lambda n: block_scale_fp8(mi, H)))
        tensors.append(w(f"layers.{L}.ffn.shared_experts.w2.weight", "F8_E4M3", [H, mi], lambda n, b=base + 0.9: [fp8_e4m3((i * 0.37 + b) % 2.0 - 1.0) for i in range(n)]))
        tensors.append(w(f"layers.{L}.ffn.shared_experts.w2.scale", "F8_E8M0", [(H + 127) // 128, (mi + 127) // 128], lambda n: block_scale_fp8(H, mi)))
        # routed experts FP4: scale run (w1,w2,w3) contiguous, weight run (w1,w2,w3)
        for e in range(ne):
            eb = base + e * 0.77
            s1 = block_scale_fp4(mi, H)
            s2 = block_scale_fp4(H, mi)
            s3 = block_scale_fp4(mi, H)
            tensors.append(Pack(f"layers.{L}.ffn.experts.{e}.w1.scale", "F8_E8M0", [mi, (H + 31) // 32]).assign(s1))
            tensors.append(Pack(f"layers.{L}.ffn.experts.{e}.w2.scale", "F8_E8M0", [H, (mi + 31) // 32]).assign(s2))
            tensors.append(Pack(f"layers.{L}.ffn.experts.{e}.w3.scale", "F8_E8M0", [mi, (H + 31) // 32]).assign(s3))
            w1v = [[((r * 7 + c * 3 + e) % 5) * 0.4 - 0.8 for c in range(H)] for r in range(mi)]
            w3v = [[((r * 5 + c * 11 + e) % 5) * 0.4 - 0.8 for c in range(H)] for r in range(mi)]
            w2v = [[((r * 13 + c * 2 + e) % 5) * 0.4 - 0.8 for c in range(mi)] for r in range(H)]
            tensors.append(Pack(f"layers.{L}.ffn.experts.{e}.w1.weight", "I8", [mi, (H + 1) // 2]).assign(pack_fp4(w1v, H)))
            tensors.append(Pack(f"layers.{L}.ffn.experts.{e}.w2.weight", "I8", [H, (mi + 1) // 2]).assign(pack_fp4(w2v, mi)))
            tensors.append(Pack(f"layers.{L}.ffn.experts.{e}.w3.weight", "I8", [mi, (H + 1) // 2]).assign(pack_fp4(w3v, H)))

    # ---- write safetensors ----
    header = {}
    offset = 0
    body = b""
    for t in tensors:
        header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [offset, offset + t.n * {"F32": 4, "BF16": 2, "I64": 8, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}[t.dtype]]}
        offset += t.n * {"F32": 4, "BF16": 2, "I64": 8, "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1}[t.dtype]
        body += t.bytes()
    hdr_bytes = json.dumps(header, separators=(",", ":")).encode()
    with open(os.path.join(outdir, "model-00001-of-00001.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        f.write(body)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(cfg_dict(), f, indent=2)
    print(f"tiny_dsv4 fixture written to {outdir} ({len(tensors)} tensors, {offset} bytes)")


if __name__ == "__main__":
    main()
