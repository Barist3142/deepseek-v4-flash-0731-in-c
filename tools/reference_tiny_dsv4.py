#!/usr/bin/env python3
"""reference_tiny_dsv4.py - independent scalar oracle for the tiny checkpoint.

A slow, pure-Python, standard-library implementation of the full 4-layer graph:
it reads the safetensors fixture itself, implements BF16/FP8/FP4 decode, RMSNorm,
YaRN RoPE, Hadamard, Hyper-Connection, compressor/indexer attention, routing and
experts, and the head. It never calls the C engine or a C library, and it does
not share code with the C kernels.

The semantics match the released inference/model.py + kernel.py at revision
f981a343464c25f82b901e5882716b3b2fa514de:
  - every module boundary rounds to BF16 (round-to-nearest-even)
  - FP8 activations: per-128 amax with 1e-4 floor, power-of-two E8M0 scale,
    values clamped to [-448, 448], E4M3 rounding
  - FP4 activations: per-32 amax with 6*2^-126 floor, power-of-two scale,
    values clamped to [-6, 6], E2M1 rounding
  - KV act quant on the non-rope dims uses group 64
  - compressor pools with per-channel softmax over the block; ratio-4 blocks
    overlap (2*ratio rows), ratio-128 do not
  - the indexer scores compressed positions as sum_h w_h * relu(q_h . k_p_h)

Output: JSON logits for all 130 positions in the fixed token sequence, written
to tests/fixtures/tiny_dsv4/ref_logits.json.

Usage: python3 tools/reference_tiny_dsv4.py [fixture_dir] [out_json]
"""
import json
import math
import os
import struct
import sys
import numpy as np


def f32_fma(a, b, c):
    """Float32 fused multiply-add, with one rounding after the addition."""
    return np.float32(np.float64(np.float32(a)) * np.float64(np.float32(b)) +
                      np.float64(np.float32(c)))


def f32_dot(a, b):
    """Sequential float32 FMA dot product, matching the C kernels."""
    acc = np.float32(0.0)
    for i in range(len(a)):
        acc = f32_fma(a[i], b[i], acc)
    return float(acc)


def f32_sum(a):
    """Naive sequential float32 sum, matching the C kernels."""
    acc = np.float32(0.0)
    for v in a:
        acc = np.float32(acc + np.float32(v))
    return float(acc)


# ------------------------------------------------------------------ io ----
def read_st(path):
    with open(path, "rb") as f:
        raw = f.read()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n].decode())
    data = raw[8 + n:]
    out = {}
    for name, meta in hdr.items():
        if isinstance(meta, dict):
            o0, o1 = meta["data_offsets"]
            out[name] = (meta["dtype"], tuple(meta["shape"]), data[o0:o1])
    return out


# ------------------------------------------------------------- numerics ----
def f2bf16(v):
    u = struct.unpack("<I", struct.pack("<f", float(v)))[0]
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u >> 16


def bf16f(v):
    return struct.unpack("<f", b"\x00\x00" + struct.pack("<H", v))[0]


def round_half_even(x):
    r = math.floor(x)
    frac = x - r
    if frac > 0.5:
        return r + 1.0
    if frac < 0.5:
        return r
    return r if r % 2 == 0 else r + 1.0


def round_e4m3(v):
    """round |v| to the nearest E4M3 grid value (3-bit mantissa),
    round-half-even, matching torch's float->e4m3 cast."""
    if v == 0:
        return 0.0
    a = abs(v)
    s = -1.0 if v < 0 else 1.0
    e = int(np.floor(np.log2(np.float32(a))))
    if e >= -6:
        step = np.float32(2.0 ** (e - 3))
        r = round_half_even(float(np.float32(a) / step))
        if r >= 16:  # carry into the next exponent
            r = 8
            e += 1
            step = np.float32(2.0 ** (e - 3))
        if e > 8:
            return s * 448.0
        return s * r * float(step)
    # subnormal: step 2^-9
    step = np.float32(2.0 ** -9)
    r = round_half_even(float(np.float32(a) / step))
    # Rounding may carry across the subnormal/normal boundary to code 0x08.
    return s * r * float(step)


def act_quant(x, group=128):
    """FP8 QAT: returns (values on the E4M3 grid, scales per group)."""
    n = len(x)
    xa = np.asarray(x, dtype=np.float32)
    q = [0.0] * n
    s = []
    for g in range(0, n, group):
        seg = xa[g:g + group]
        amax = float(np.abs(seg).max())
        if amax < 1e-4:
            amax = 1e-4
        r = np.float32(amax * (1.0 / 448.0))
        e = int(np.ceil(np.log2(r)))
        scale = np.float32(2.0 ** e)
        s.append(float(scale))
        inv = np.float32(1.0 / scale)
        for i, v in enumerate(seg):
            qv = np.float32(max(-448.0, min(448.0, np.float32(v) * inv)))
            q[g + i] = round_e4m3(float(qv))
    return q, s


def fp4_act_quant(x, group=32):
    n = len(x)
    xa = np.asarray(x, dtype=np.float32)
    q = [0.0] * n
    s = []
    vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    for g in range(0, n, group):
        seg = xa[g:g + group]
        amax = float(np.abs(seg).max())
        amax = max(amax, 6 * 2 ** -126)
        r = np.float32(amax * (1.0 / 6.0))
        e = int(np.ceil(np.log2(r)))
        scale = np.float32(2.0 ** e)
        s.append(float(scale))
        inv = np.float32(1.0 / scale)
        for i, v in enumerate(seg):
            qv = np.float32(max(-6.0, min(6.0, np.float32(v) * inv)))
            best = 0
            bd = abs(abs(qv) - vals[0])
            for k in range(1, 8):
                d = abs(abs(qv) - vals[k])
                if d < bd or (d == bd and k % 2 == 0):
                    bd = d
                    best = k
            q[g + i] = (-vals[best] if qv < 0 else vals[best])
    return q, s


def act_quant_inplace(x, group=128, mode=0):
    """FP8 QAT: quantise then dequantise in place (the released inplace=True
    path). mode: 0 = BF16 (indexer q), 1 = FP8 grid value (KV), 2 = float32
    (compressor). group is 64 for the KV/compressor non-rope dims."""
    n = len(x)
    out = [0.0] * n
    for g in range(0, n, group):
        seg = x[g:g + group]
        amax = max(abs(v) for v in seg)
        if amax < 1e-4:
            amax = 1e-4
        r = np.float32(amax * (1.0 / 448.0))
        e = int(np.ceil(np.log2(r)))
        scale = np.float32(2.0 ** e)
        inv = np.float32(1.0 / scale)
        for i, v in enumerate(seg):
            qv = np.float32(max(-448.0, min(448.0, np.float32(v) * inv)))
            dequant = np.float32(round_e4m3(float(qv)) * scale)
            if mode == 2:
                out[g + i] = float(dequant)
            elif mode == 1:
                out[g + i] = round_e4m3(float(dequant))
            else:
                out[g + i] = bf16f(f2bf16(float(dequant)))
    return out


def fp4_act_quant_inplace(x, group=32, mode=0):
    """FP4 QAT: quantise then dequantise in place. mode 0 = BF16 (indexer q),
    mode 1 = float32 (indexer compressor)."""
    n = len(x)
    out = [0.0] * n
    vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    for g in range(0, n, group):
        seg = x[g:g + group]
        amax = max(abs(v) for v in seg)
        amax = max(amax, 6 * 2 ** -126)
        r = np.float32(amax * (1.0 / 6.0))
        e = int(np.ceil(np.log2(r)))
        scale = np.float32(2.0 ** e)
        inv = np.float32(1.0 / scale)
        for i, v in enumerate(seg):
            qv = np.float32(max(-6.0, min(6.0, np.float32(v) * inv)))
            best = 0
            bd = abs(abs(qv) - vals[0])
            for k in range(1, 8):
                d = abs(abs(qv) - vals[k])
                if d < bd or (d == bd and k % 2 == 0):
                    bd = d
                    best = k
            dequant = (-vals[best] if qv < 0 else vals[best]) * float(scale)
            out[g + i] = float(dequant) if mode == 1 else bf16f(f2bf16(dequant))
    return out


def rmsnorm(x, w, eps=1e-6):
    xa = np.asarray(x, dtype=np.float32)
    wa = np.asarray(w, dtype=np.float32)
    ss = f32_sum([float(v) * float(v) for v in xa])
    r = np.float32(1.0 / np.sqrt(np.float32(np.float32(ss) / len(x) + eps)))
    return [bf16f(f2bf16(float(xa[i] * r * wa[i]))) for i in range(len(x))]


def rmsnorm_plain(x, eps=1e-6):
    xa = np.asarray(x, dtype=np.float32)
    ss = f32_sum([float(v) * float(v) for v in xa])
    r = np.float32(1.0 / np.sqrt(np.float32(np.float32(ss) / len(x) + eps)))
    return [float(xa[i] * r) for i in range(len(x))]


def rope_freqs(dim, theta, yarn, factor, beta_fast, beta_slow,
               original_position, position):
    out = []
    for i in range(dim // 2):
        inv = np.float32(1.0 / (theta ** (2.0 * i / dim)))
        if yarn:
            low = int(np.floor(dim * np.log(original_position / (beta_fast * 2 * math.pi)) /
                               (2 * np.log(theta))))
            high = int(np.ceil(dim * np.log(original_position / (beta_slow * 2 * math.pi)) /
                               (2 * np.log(theta))))
            low = max(low, 0)
            high = min(high, dim - 1)
            if high == low:
                high += 0.001
            ramp = np.float32(np.clip((i - low) / (high - low), 0.0, 1.0))
            inv_interp = np.float32(inv / factor)
            inv = np.float32(inv_interp * ramp + inv * (1 - ramp))
        angle = np.float32(position) * inv
        out.append((float(np.float32(math.cos(float(angle)))),
                    float(np.float32(math.sin(float(angle))))))
    return out


def rope_apply(x, freqs):
    """x: [heads][head_dim]; rotate the last len(freqs)*2 dims (float32)."""
    d = len(freqs) * 2
    for h in x:
        base = len(h) - d
        xa = np.asarray(h, dtype=np.float32)
        for i, (c, s) in enumerate(freqs):
            x0 = xa[base + 2 * i]
            x1 = xa[base + 2 * i + 1]
            cf = np.float32(c)
            sf = np.float32(s)
            xa[base + 2 * i] = bf16f(f2bf16(float(np.float32(x0 * cf - x1 * sf))))
            xa[base + 2 * i + 1] = bf16f(f2bf16(float(np.float32(x0 * sf + x1 * cf))))
        h[:] = xa


def rope_apply_inv(x, freqs):
    d = len(freqs) * 2
    for h in x:
        base = len(h) - d
        xa = np.asarray(h, dtype=np.float32)
        for i, (c, s) in enumerate(freqs):
            x0 = xa[base + 2 * i]
            x1 = xa[base + 2 * i + 1]
            cf = np.float32(c)
            sf = np.float32(s)
            xa[base + 2 * i] = bf16f(f2bf16(float(np.float32(x0 * cf + x1 * sf))))
            xa[base + 2 * i + 1] = bf16f(f2bf16(float(np.float32(-x0 * sf + x1 * cf))))
        h[:] = xa


def hadamard(x):
    n = len(x)
    t = np.zeros(n, dtype=np.float32)
    xa = np.asarray(x, dtype=np.float32)
    length = 1
    while length < n:
        t[:] = xa[:]
        for i in range(0, n, length * 2):
            for j in range(length):
                a = t[i + j]
                b = t[i + j + length]
                xa[i + j] = np.float32(a + b)
                xa[i + j + length] = np.float32(a - b)
        length *= 2
    inv = np.float32(1.0 / np.sqrt(np.float32(n)))
    xa *= inv
    x[:] = [bf16f(f2bf16(float(v))) for v in xa]


def sigmoidf_stable(x):
    x = np.float32(x)
    if x >= 0.0:
        e = np.float32(np.exp(-x))
        return float(np.float32(1.0 / (1.0 + e)))
    e = np.float32(np.exp(x))
    return float(np.float32(e / (1.0 + e)))


def hc_split(mixes, hc_scale, hc_base, mult, iters, eps):
    m = mult
    pre = [float(np.float32(sigmoidf_stable(mixes[i] * hc_scale[0] + hc_base[i]) + eps)) for i in range(m)]
    post = [float(np.float32(2.0 * sigmoidf_stable(mixes[m + i] * hc_scale[1] + hc_base[m + i]))) for i in range(m)]
    comb = [[mixes[2 * m + i * m + j] * hc_scale[2] + hc_base[2 * m + i * m + j] for j in range(m)] for i in range(m)]
    # softmax rows + eps (float32)
    for i in range(m):
        row = np.asarray(comb[i], dtype=np.float32)
        mx = np.float32(row.max())
        e = np.exp(row - mx)
        s = np.float32(f32_sum(e))
        comb[i] = [float(np.float32(np.float32(v) / s + eps)) for v in e]
    # one column norm
    for j in range(m):
        s = np.float32(f32_sum([comb[i][j] for i in range(m)]))
        for i in range(m):
            comb[i][j] = float(np.float32(np.float32(comb[i][j]) / (s + eps)))
    for _ in range(iters - 1):
        for i in range(m):
            s = np.float32(f32_sum(comb[i]))
            comb[i] = [float(np.float32(np.float32(v) / (s + eps))) for v in comb[i]]
        for j in range(m):
            s = np.float32(f32_sum([comb[i][j] for i in range(m)]))
            for i in range(m):
                comb[i][j] = float(np.float32(np.float32(comb[i][j]) / (s + eps)))
    return pre, post, comb


# -------------------------------------------------------------- decode ----
def decode_tensor(t, dtype, shape):
    data = t
    n = 1
    for d in shape:
        n *= d
    if dtype == "F32":
        vals = list(struct.unpack("<%df" % n, data[:4 * n]))
    elif dtype == "BF16":
        raw = struct.unpack("<%dH" % n, data[:2 * n])
        vals = [bf16f(v) for v in raw]
    elif dtype == "F8_E4M3":
        vals = list(data[:n])
    elif dtype == "F8_E8M0":
        vals = list(data[:n])
    elif dtype == "I8":
        vals = list(data[:n])
    elif dtype == "I64":
        vals = list(struct.unpack("<%dq" % n, data[:8 * n]))
    else:
        raise ValueError(dtype)
    if len(shape) > 1:
        # reshape row-major
        out = []
        stride = n // shape[0]
        for r in range(shape[0]):
            out.append(vals[r * stride:(r + 1) * stride])
        return out
    return vals


def e4m3f(code):
    s = -1.0 if (code >> 7) & 1 else 1.0
    e = (code >> 3) & 15
    m = code & 7
    if e == 0:
        return s * (m * 2.0 ** -9)
    return s * ((8 + m) * 2.0 ** (e - 10))


def e8m0f(code):
    if code == 255:
        return float('nan')
    return 2.0 ** (code - 127)


def fp4_nibble(nib):
    vals = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    s = -1.0 if nib & 8 else 1.0
    return s * vals[nib & 7]


# ------------------------------------------------------------- model ----
class TinyModel:
    def __init__(self, t, cfg):
        self.t = t
        self.c = cfg
        self.H = cfg["hidden_size"]
        self.L = cfg["num_hidden_layers"]
        self.V = cfg["vocab_size"]
        self.heads = cfg["num_attention_heads"]
        self.hd = cfg["head_dim"]
        self.rd = cfg["qk_rope_head_dim"]
        self.ql = cfg["q_lora_rank"]
        self.G = cfg["o_groups"]
        self.ol = cfg["o_lora_rank"]
        self.win = cfg["sliding_window"]
        self.ne = cfg["n_routed_experts"]
        self.topk = cfg["num_experts_per_tok"]
        self.mi = cfg["moe_intermediate_size"]
        self.mult = cfg["hc_mult"]
        self.ratios = cfg["compress_ratios"]
        self.ih = cfg["index_n_heads"]
        self.idim = cfg["index_head_dim"]
        self.itopk = cfg["index_topk"]
        self.eps = cfg["rms_norm_eps"]
        self.hc_eps = cfg["hc_eps"]
        self.hc_iters = cfg["hc_sinkhorn_iters"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.limit = cfg["swiglu_limit"]
        self.rope_theta = cfg["rope_theta"]
        self.compress_theta = cfg["compress_rope_theta"]
        self.rf = cfg["rope_scaling"]["factor"]
        self.bf = cfg["rope_scaling"]["beta_fast"]
        self.bs = cfg["rope_scaling"]["beta_slow"]
        self.original_position = cfg["rope_scaling"]["original_max_position_embeddings"]

        # per-layer runtime
        self.kv_cache = {}      # layer -> [n][hd]
        self.kv_state = {}      # layer -> dict of slot -> [coff*hd]
        self.score_state = {}
        self.comp_count = {}
        self.idx_cache = {}
        self.idx_count = {}
        self.state = None       # [mult][H]

    def T(self, name):
        return decode_tensor(self.t[name][2], self.t[name][0], self.t[name][1])

    def gemv_bf16(self, w, x, out_bf16=True):
        rows = len(w)
        xa = np.asarray(x, dtype=np.float32)
        y = []
        for r in range(rows):
            wa = np.asarray(w[r], dtype=np.float32)
            acc = np.float32(0.0)
            for c in range(len(x)):
                acc = f32_fma(wa[c], xa[c], acc)
            y.append(float(acc))
        if out_bf16:
            y = [bf16f(f2bf16(v)) for v in y]
        return y

    def gemv_fp8(self, wcode, wscale_code, x, out_bf16=True):
        rows = len(wcode)
        cols = len(wcode[0])
        cb = (cols + 127) // 128
        rb = (rows + 127) // 128
        wscale = [[e8m0f(wscale_code[rb_i][cb_i]) for cb_i in range(cb)] for rb_i in range(rb)]
        qx, xs = act_quant(x)
        qa = np.asarray(qx, dtype=np.float32)
        y = []
        for r in range(rows):
            wrow = np.asarray([e4m3f(c) for c in wcode[r]], dtype=np.float32)
            acc = np.float32(0.0)
            for cb_i in range(cb):
                c0 = cb_i * 128
                c1 = min(c0 + 128, cols)
                partial = np.float32(0.0)
                for c in range(c0, c1):
                    partial = f32_fma(wrow[c], qa[c], partial)
                scale = np.float32(np.float32(xs[cb_i]) * np.float32(wscale[r // 128][cb_i]))
                acc = f32_fma(partial, scale, acc)
            y.append(float(acc))
        if out_bf16:
            y = [bf16f(f2bf16(v)) for v in y]
        return y

    def gemv_fp4(self, wbytes, wscale_code, x, out_bf16=True):
        # wbytes: rows of packed bytes; x decoded E4M3 values + scales
        rows = len(wbytes)
        cols = len(x)
        gcols = (cols + 31) // 32
        wscale = [[e8m0f(wscale_code[r][c]) for c in range(gcols)] for r in range(rows)]
        qx, xs = act_quant(x)
        qa = np.asarray(qx, dtype=np.float32)
        y = []
        for r in range(rows):
            acc = np.float32(0.0)
            for c0 in range(0, cols, 32):
                c1 = min(c0 + 32, cols)
                partial = np.float32(0.0)
                for c in range(c0, c1):
                    b = wbytes[r][c // 2]
                    nib = (b >> 4) if (c & 1) else (b & 0xF)
                    partial = f32_fma(fp4_nibble(nib), qa[c], partial)
                scale = np.float32(np.float32(xs[c0 // 128]) * np.float32(wscale[r][c0 // 32]))
                acc = f32_fma(partial, scale, acc)
            y.append(float(acc))
        if out_bf16:
            y = [bf16f(f2bf16(v)) for v in y]
        return y

    # ------------------------------------------------------------ steps ----
    def compressor_step(self, layer, x, pos, head_dim, wkv, wgate, ape, norm,
                        ratio, rotate, key=None):
        key = key if key is not None else layer
        coff = 2 if ratio == 4 else 1
        cd = coff * head_dim
        if key not in self.kv_state:
            self.kv_state[key] = [[np.float32(0.0)] * cd for _ in range(coff * ratio)]
            self.score_state[key] = [[np.float32(-np.inf)] * cd for _ in range(coff * ratio)]
        kv = self.gemv_bf16(wkv, x, out_bf16=False)
        sc = self.gemv_bf16(wgate, x, out_bf16=False)
        for i in range(cd):
            sc[i] = np.float32(np.float32(sc[i]) + np.float32(ape[pos % ratio][i]))
        slot = (pos % ratio) + (ratio if coff == 2 else 0)
        self.kv_state[key][slot] = [np.float32(v) for v in kv]
        self.score_state[key][slot] = [np.float32(v) for v in sc]
        if (pos + 1) % ratio != 0:
            return
        rows = ratio * coff
        bkv = []
        bsc = []
        for i in range(rows):
            if coff == 2 and i >= ratio:
                bkv.append(self.kv_state[key][i][head_dim:])
                bsc.append(self.score_state[key][i][head_dim:])
            else:
                bkv.append(self.kv_state[key][i][:head_dim])
                bsc.append(self.score_state[key][i][:head_dim])
        if coff == 2:
            for i in range(ratio):
                self.kv_state[key][i] = self.kv_state[key][ratio + i]
                self.score_state[key][i] = self.score_state[key][ratio + i]
        # per-channel softmax over rows (float32 accumulation)
        comp = [0.0] * head_dim
        for ch in range(head_dim):
            col = np.asarray([bsc[i][ch] for i in range(rows)], dtype=np.float32)
            mx = float(col.max())
            e = np.exp(col - mx)
            inv = np.float32(1.0 / f32_sum(e))
            bkv_col = np.asarray([bkv[i][ch] for i in range(rows)], dtype=np.float32)
            acc = np.float32(0.0)
            for i in range(rows):
                weight = np.float32(e[i] * inv)
                acc = f32_fma(weight, bkv_col[i], acc)
            comp[ch] = float(acc)
        comp = [bf16f(f2bf16(v)) for v in comp]
        comp = rmsnorm(comp, norm, self.eps)
        # rope at block start
        blk = pos + 1 - ratio
        freqs = rope_freqs(self.rd, self.compress_theta, True, self.rf, self.bf,
                           self.bs, self.original_position, blk)
        tail = comp[-self.rd:]
        for i, (c, s) in enumerate(freqs):
            x0, x1 = tail[2 * i], tail[2 * i + 1]
            tail[2 * i] = bf16f(f2bf16(x0 * c - x1 * s))
            tail[2 * i + 1] = bf16f(f2bf16(x0 * s + x1 * c))
        comp[-self.rd:] = tail
        if rotate:
            hadamard(comp)
            comp = fp4_act_quant_inplace(comp, 32, 0)
        else:
            comp = act_quant_inplace(comp[:-self.rd], 64, 0) + comp[-self.rd:]
        if rotate:
            self.idx_cache.setdefault(layer, []).append([np.float32(v) for v in comp])
            self.idx_count[layer] = self.idx_count.get(layer, 0) + 1
        else:
            self.kv_cache.setdefault(layer, []).append([np.float32(v) for v in comp])
            self.comp_count[layer] = self.comp_count.get(layer, 0) + 1

    def indexer_step(self, layer, x, qr, pos):
        ratio = 4
        # q
        q = self.gemv_fp8(self.T(f"layers.{layer}.attn.indexer.wq_b.weight"),
                          self.T(f"layers.{layer}.attn.indexer.wq_b.scale"), qr)
        q = [q[i * self.idim:(i + 1) * self.idim] for i in range(self.ih)]
        freqs = rope_freqs(self.rd, self.compress_theta, True, self.rf, self.bf,
                           self.bs, self.original_position, pos)
        rope_apply(q, freqs)
        for h in q:
            hadamard(h)
            h[:] = fp4_act_quant_inplace(h, 32, 0)
        # compressor (consumes the FULL hidden state x, not the q_lora proj qr)
        self.compressor_step(layer, x, pos, self.idim,
                             self.T(f"layers.{layer}.attn.indexer.compressor.wkv.weight"),
                             self.T(f"layers.{layer}.attn.indexer.compressor.wgate.weight"),
                             self.T(f"layers.{layer}.attn.indexer.compressor.ape"),
                             self.T(f"layers.{layer}.attn.indexer.compressor.norm.weight"),
                             ratio, True, key=(layer, "idx"))
        # weights
        wts = self.gemv_bf16(self.T(f"layers.{layer}.attn.indexer.weights_proj.weight"), x)
        scl = np.float32(np.float32(self.idim) ** np.float32(-0.5)) * np.float32(np.float32(self.ih) ** np.float32(-0.5))
        wts = [bf16f(f2bf16(v * scl)) for v in wts]
        keys = self.idx_cache.get(layer, [])
        scores = []
        for kp in keys:
            kpa = np.asarray(kp, dtype=np.float32)
            acc = np.float32(0.0)
            for h in range(self.ih):
                qh = np.asarray(q[h], dtype=np.float32)
                dot = np.float32(f32_dot(qh, kpa))
                if dot < 0:
                    dot = np.float32(0.0)
                acc = f32_fma(dot, wts[h], acc)
            scores.append(float(acc))
        topk = min(self.itopk, len(scores))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [self.win + i for i in order]

    def attn_step(self, layer, x, pos):
        c = self.c
        d, rd = self.hd, self.rd
        ratio = self.ratios[layer]
        # q
        qr = self.gemv_fp8(self.T(f"layers.{layer}.attn.wq_a.weight"), self.T(f"layers.{layer}.attn.wq_a.scale"), x)
        qr = rmsnorm(qr, self.T(f"layers.{layer}.attn.q_norm.weight"), self.eps)
        q = self.gemv_fp8(self.T(f"layers.{layer}.attn.wq_b.weight"),
                          self.T(f"layers.{layer}.attn.wq_b.scale"), qr)
        q = [q[i * d:(i + 1) * d] for i in range(self.heads)]
        for h in q:
            ss = 0.0
            for v in h:
                ss += float(v) * float(v)
            r = 1.0 / math.sqrt(ss / d + self.eps)
            h[:] = [bf16f(f2bf16(v * r)) for v in h]
        yarn = 1 if ratio else 0
        theta = self.compress_theta if ratio else self.rope_theta
        freqs = rope_freqs(rd, theta, yarn, self.rf, self.bf, self.bs,
                           self.original_position, pos)
        rope_apply(q, freqs)
        # kv
        kv = self.gemv_fp8(self.T(f"layers.{layer}.attn.wkv.weight"), self.T(f"layers.{layer}.attn.wkv.scale"), x)
        kv = rmsnorm(kv, self.T(f"layers.{layer}.attn.kv_norm.weight"), self.eps)
        rope_apply([kv], freqs)
        kv = act_quant_inplace(kv[:-rd], 64) + kv[-rd:]
        # window
        kvc = self.kv_cache.setdefault(layer, [None] * (self.win))
        kvc[pos % self.win] = [np.float32(v) for v in kv]
        # topk indices
        win_idx = []
        if pos >= self.win - 1:
            s = pos % self.win
            win_idx = [(s + 1 + i) % self.win for i in range(self.win)]
        elif pos > 0:
            win_idx = list(range(pos + 1)) + [-1] * (self.win - pos - 1)
        else:
            win_idx = [0] + [-1] * (self.win - 1)
        comp_idx = []
        if ratio:
            self.compressor_step(layer, x, pos, d,
                                 self.T(f"layers.{layer}.attn.compressor.wkv.weight"),
                                 self.T(f"layers.{layer}.attn.compressor.wgate.weight"),
                                 self.T(f"layers.{layer}.attn.compressor.ape"),
                                 self.T(f"layers.{layer}.attn.compressor.norm.weight"),
                                 ratio, False)
            if ratio == 4:
                comp_idx = self.indexer_step(layer, x, qr, pos)
            else:
                cnt = self.comp_count.get(layer, 0)
                comp_idx = [self.win + i for i in range(cnt)]
        # sparse attn
        scale = np.float32(np.float32(d) ** np.float32(-0.5))
        sink = self.T(f"layers.{layer}.attn.attn_sink")
        o = []
        for h in range(self.heads):
            qh = np.asarray(q[h], dtype=np.float32)
            topk_idx = win_idx + comp_idx
            # two-pass softmax (pass 1: scores + max, pass 2: denominator,
            # pass 3: value accumulation) -- matches the C engine.
            scores = []
            mx = np.float32(-1e30)
            for t in topk_idx:
                if t < 0:
                    scores.append(np.float32(-1e30))
                    continue
                kvv = np.asarray(kvc[t], dtype=np.float32)
                s = np.float32(np.float32(f32_dot(qh, kvv)) * np.float32(scale))
                scores.append(s)
                if s > mx:
                    mx = s
            denom = np.float32(np.exp(np.float32(sink[h]) - mx))
            for s in scores:
                if s > np.float32(-1e20):
                    denom = np.float32(denom + np.exp(s - mx))
            oh = np.zeros(d, dtype=np.float32)
            for t, s in zip(topk_idx, scores):
                if t < 0:
                    continue
                a = bf16f(f2bf16(float(np.exp(s - mx))))   # BF16 weight
                kvv = np.asarray(kvc[t], dtype=np.float32)
                for i in range(d):
                    oh[i] = f32_fma(a, kvv[i], oh[i])
            o.append([bf16f(f2bf16(float(v / denom))) for v in oh])
        rope_apply_inv(o, freqs)
        # o proj
        og = [o[i] for i in range(self.heads)]
        gdim = self.heads * d // self.G
        oall = []
        for g in range(self.G):
            seg = og[g * (gdim // d):(g + 1) * (gdim // d)]
            flat = [v for h in seg for v in h]
            wcode = self.T(f"layers.{layer}.attn.wo_a.weight")[g * self.ol:(g + 1) * self.ol]
            ws = self.T(f"layers.{layer}.attn.wo_a.scale")
            wrows = self.ol
            wcblk = (gdim + 127) // 128
            wsc = [[e8m0f(ws[(g * self.ol) // 128 + ri][ci]) for ci in range(wcblk)]
                   for ri in range((self.ol + 127) // 128)]
            # wo_a: BF16(FP8 weight x scale), no input quantisation
            for r in range(wrows):
                acc = np.float32(0.0)
                for c in range(gdim):
                    wv = bf16f(f2bf16(e4m3f(wcode[r][c]) * wsc[r // 128][c // 128]))
                    acc = f32_fma(wv, flat[c], acc)
                oall.append(bf16f(f2bf16(acc)))
        out = self.gemv_fp8(self.T(f"layers.{layer}.attn.wo_b.weight"), self.T(f"layers.{layer}.attn.wo_b.scale"), oall)
        return out

    def moe_step(self, layer, x, token_id):
        c = self.c
        gate = self.gemv_bf16(self.T(f"layers.{layer}.ffn.gate.weight"), x, out_bf16=False)
        bias = self.T(f"layers.{layer}.ffn.gate.bias")
        score = [float(np.sqrt(np.float32(np.log1p(np.exp(np.float32(min(z, 20.0))))))) if z < 20 else float(np.sqrt(np.float32(z))) for z in gate]
        choice = [score[i] + bias[i] for i in range(self.ne)]
        order = sorted(range(self.ne), key=lambda i: choice[i], reverse=True)[:self.topk]
        # weights from unbiased scores, normalised
        ssum = f32_sum([score[i] for i in order])
        wts = [score[i] / ssum * self.route_scale for i in order]
        order, wts = zip(*sorted(zip(order, wts)))
        acc = [0.0] * self.H
        for e, w in zip(order, wts):
            xb = x
            # w1/w3 fp4
            w1b = self.T(f"layers.{layer}.ffn.experts.{e}.w1.weight")
            w1s = self.T(f"layers.{layer}.ffn.experts.{e}.w1.scale")
            w3b = self.T(f"layers.{layer}.ffn.experts.{e}.w3.weight")
            w3s = self.T(f"layers.{layer}.ffn.experts.{e}.w3.scale")
            w2b = self.T(f"layers.{layer}.ffn.experts.{e}.w2.weight")
            w2s = self.T(f"layers.{layer}.ffn.experts.{e}.w2.scale")
            g = self.gemv_fp4(w1b, w1s, xb)
            u = self.gemv_fp4(w3b, w3s, xb)
            y = []
            for i in range(self.mi):
                gg = min(g[i], self.limit)
                uu = max(-self.limit, min(self.limit, u[i]))
                silu = float(np.float32(np.float32(gg) / (1.0 + np.exp(np.float32(-gg)))))
                y.append(silu * uu * w)
            yb = [bf16f(f2bf16(v)) for v in y]
            down = self.gemv_fp4(w2b, w2s, yb)
            for i in range(self.H):
                acc[i] += down[i]
        # shared expert
        sh1 = self.T(f"layers.{layer}.ffn.shared_experts.w1.weight")
        sh1s = self.T(f"layers.{layer}.ffn.shared_experts.w1.scale")
        sh3 = self.T(f"layers.{layer}.ffn.shared_experts.w3.weight")
        sh3s = self.T(f"layers.{layer}.ffn.shared_experts.w3.scale")
        sh2 = self.T(f"layers.{layer}.ffn.shared_experts.w2.weight")
        sh2s = self.T(f"layers.{layer}.ffn.shared_experts.w2.scale")
        g = self.gemv_fp8(sh1, sh1s, x)
        u = self.gemv_fp8(sh3, sh3s, x)
        y = []
        for i in range(self.mi):
            gg = min(g[i], self.limit)
            uu = max(-self.limit, min(self.limit, u[i]))
            silu = float(np.float32(np.float32(gg) / (1.0 + np.exp(np.float32(-gg)))))
            y.append(silu * uu)
        yb = [bf16f(f2bf16(v)) for v in y]
        down = self.gemv_fp8(sh2, sh2s, yb)
        for i in range(self.H):
            acc[i] += down[i]
        return [bf16f(f2bf16(v)) for v in acc]

    def hc_pre(self, x, fn, scale3, base):
        mult = self.mult
        flat = np.asarray([v for h in x for v in h], dtype=np.float32)
        ss = np.float32(0.0)
        for v in flat:
            ss = np.float32(ss + np.float32(v) * np.float32(v))
        rsqrt = np.float32(1.0 / np.sqrt(np.float32(ss / len(flat) + self.eps)))
        mixes = []
        for j in range((2 + mult) * mult):
            row = np.asarray(fn[j], dtype=np.float32)
            mixes.append(float(np.float32(f32_dot(row, flat)) * rsqrt))
        pre, post, comb = hc_split(mixes, scale3, base, mult, self.hc_iters, self.hc_eps)
        bin_ = [0.0] * self.H
        for j in range(self.H):
            acc = np.float32(0.0)
            for i in range(mult):
                acc = f32_fma(pre[i], x[i][j], acc)
            bin_[j] = bf16f(f2bf16(float(acc)))
        return bin_, post, comb

    def hc_post(self, x, branch_out, post, comb):
        mult = self.mult
        new = []
        for i in range(mult):
            row = []
            for j in range(self.H):
                y = np.float32(np.float32(post[i]) * np.float32(branch_out[j]))
                for k in range(mult):
                    y = f32_fma(comb[i][k], x[k][j], y)
                row.append(y)
            new.append(row)
        return [[bf16f(f2bf16(float(v))) for v in row] for row in new]

    def forward_token(self, token_id, pos):
        # embed
        emb = self.T("embed.weight")[token_id]
        state = [[emb[i] for i in range(self.H)] for _ in range(self.mult)]
        self.state = state
        for layer in range(self.L):
            fn = self.T(f"layers.{layer}.hc_attn_fn")
            scale3 = self.T(f"layers.{layer}.hc_attn_scale")
            base = self.T(f"layers.{layer}.hc_attn_base")
            bin_, post, comb = self.hc_pre(state, fn, scale3, base)
            bin_ = rmsnorm(bin_, self.T(f"layers.{layer}.attn_norm.weight"), self.eps)
            out = self.attn_step(layer, bin_, pos)
            state = self.hc_post(state, out, post, comb)
            fn = self.T(f"layers.{layer}.hc_ffn_fn")
            scale3 = self.T(f"layers.{layer}.hc_ffn_scale")
            base = self.T(f"layers.{layer}.hc_ffn_base")
            bin_, post, comb = self.hc_pre(state, fn, scale3, base)
            bin_ = rmsnorm(bin_, self.T(f"layers.{layer}.ffn_norm.weight"), self.eps)
            out = self.moe_step(layer, bin_, token_id)
            state = self.hc_post(state, out, post, comb)
        # head
        flat = np.asarray([v for h in state for v in h], dtype=np.float32)
        ss = np.float32(0.0)
        for v in flat:
            ss = np.float32(ss + np.float32(v) * np.float32(v))
        rsqrt = np.float32(1.0 / np.sqrt(np.float32(ss / len(flat) + self.eps)))
        hc_fn = self.T("hc_head_fn")
        hc_base = self.T("hc_head_base")
        hc_scale = self.T("hc_head_scale")[0]
        pre = []
        for i in range(self.mult):
            row = np.asarray(hc_fn[i], dtype=np.float32)
            mixes = np.float32(f32_dot(row, flat) * rsqrt)
            pre.append(np.float32(1.0 / (1.0 + np.exp(-(mixes * np.float32(hc_scale) + np.float32(hc_base[i])))) + self.hc_eps))
        h = [0.0] * self.H
        for j in range(self.H):
            acc = np.float32(0.0)
            for i in range(self.mult):
                acc = f32_fma(pre[i], state[i][j], acc)
            h[j] = bf16f(f2bf16(float(acc)))
        h = rmsnorm(h, self.T("norm.weight"), self.eps)
        head = self.T("head.weight")
        ha = np.asarray(h, dtype=np.float32)
        logits = []
        for v in range(self.V):
            logits.append(f32_dot(np.asarray(head[v], dtype=np.float32), ha))
        return logits


def main():
    fdir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/tiny_dsv4"
    outjson = sys.argv[2] if len(sys.argv) > 2 else os.path.join(fdir, "ref_logits.json")
    cfg = json.load(open(os.path.join(fdir, "config.json")))
    t = read_st(os.path.join(fdir, "model-00001-of-00001.safetensors"))
    m = TinyModel(t, cfg)

    # fixed deterministic token sequence (16 tokens, cycles)
    seq = [3, 7, 1, 14, 5, 9, 2, 11, 0, 13, 6, 4, 12, 8, 15, 10]
    results = {}
    for pos in range(130):
        tok = seq[pos % len(seq)]
        logits = m.forward_token(tok, pos)
        results[str(pos)] = logits
    with open(outjson, "w") as f:
        json.dump(results, f)
    print(f"reference logits written to {outjson}")


if __name__ == "__main__":
    main()
