#!/usr/bin/env python3
"""verify_dsv4_checkpoint.py - validate the full tensor contract of the fixed
DeepSeek-V4-Flash-0731 revision.

Reads config.json + model.safetensors.index.json, then scans the HEADER of every
safetensors shard (no tensor data) to verify that all 67,612 main-model tensors
have the expected name, dtype and shape, and that the 4,705 optional mtp.*
tensors are reported (not rejected).

Usage: python3 tools/verify_dsv4_checkpoint.py MODEL_DIR

Exit 0 only when every main tensor matches the contract.
"""
import json
import os
import struct
import sys

# ----------------------------------------------------------------- contract --
H = 4096
V = 129280
L = 43
HEADS = 64
HEAD_DIM = 512
QL = 1024
G = 8
OL = 1024
MI = 2048
NE = 256
TOPK = 6
MULT = 4
MIX = (2 + MULT) * MULT
INDEX_HEADS = 64
INDEX_DIM = 128


def ceil_div(a, b):
    return (a + b - 1) // b


def fp8(rows, cols):
    return ("F8_E4M3", [rows, cols]), ("F8_E8M0", [ceil_div(rows, 128), ceil_div(cols, 128)])


def fp4(rows, cols):
    return ("I8", [rows, (cols + 1) // 2]), ("F8_E8M0", [rows, ceil_div(cols, 32)])


def bf16(*shape):
    return ("BF16", list(shape))


def f32(*shape):
    return ("F32", list(shape))


def i64(*shape):
    return ("I64", list(shape))


def expected_tensors(cfg):
    ratios = cfg["compress_ratios"]
    out = {}
    # global
    out["embed.weight"] = bf16(V, H)
    out["head.weight"] = bf16(V, H)
    out["norm.weight"] = bf16(H)
    out["hc_head_fn"] = f32(MULT, MULT * H)
    out["hc_head_scale"] = f32(1)
    out["hc_head_base"] = f32(MULT)

    for layer in range(L):
        ratio = ratios[layer] if layer < len(ratios) else 0
        coff = 2 if ratio == 4 else 1
        p = f"layers.{layer}"
        out[f"{p}.attn_norm.weight"] = bf16(H)
        out[f"{p}.ffn_norm.weight"] = bf16(H)
        out[f"{p}.hc_attn_fn"] = f32(MIX, MULT * H)
        out[f"{p}.hc_attn_scale"] = f32(3)
        out[f"{p}.hc_attn_base"] = f32(MIX)
        out[f"{p}.hc_ffn_fn"] = f32(MIX, MULT * H)
        out[f"{p}.hc_ffn_scale"] = f32(3)
        out[f"{p}.hc_ffn_base"] = f32(MIX)
        out[f"{p}.attn.q_norm.weight"] = bf16(QL)
        out[f"{p}.attn.kv_norm.weight"] = bf16(HEAD_DIM)
        out[f"{p}.attn.attn_sink"] = f32(HEADS)
        w, s = fp8(QL, H)
        out[f"{p}.attn.wq_a.weight"], out[f"{p}.attn.wq_a.scale"] = w, s
        w, s = fp8(HEADS * HEAD_DIM, QL)
        out[f"{p}.attn.wq_b.weight"], out[f"{p}.attn.wq_b.scale"] = w, s
        w, s = fp8(HEAD_DIM, H)
        out[f"{p}.attn.wkv.weight"], out[f"{p}.attn.wkv.scale"] = w, s
        w, s = fp8(G * OL, HEADS * HEAD_DIM // G)
        out[f"{p}.attn.wo_a.weight"], out[f"{p}.attn.wo_a.scale"] = w, s
        w, s = fp8(H, G * OL)
        out[f"{p}.attn.wo_b.weight"], out[f"{p}.attn.wo_b.scale"] = w, s
        out[f"{p}.ffn.gate.weight"] = bf16(NE, H)
        if layer < cfg["num_hash_layers"]:
            out[f"{p}.ffn.gate.tid2eid"] = i64(V, TOPK)
        else:
            out[f"{p}.ffn.gate.bias"] = f32(NE)
        w, s = fp8(MI, H)
        out[f"{p}.ffn.shared_experts.w1.weight"], out[f"{p}.ffn.shared_experts.w1.scale"] = w, s
        w, s = fp8(MI, H)
        out[f"{p}.ffn.shared_experts.w3.weight"], out[f"{p}.ffn.shared_experts.w3.scale"] = w, s
        w, s = fp8(H, MI)
        out[f"{p}.ffn.shared_experts.w2.weight"], out[f"{p}.ffn.shared_experts.w2.scale"] = w, s
        for e in range(NE):
            w, s = fp4(MI, H)
            out[f"{p}.ffn.experts.{e}.w1.weight"], out[f"{p}.ffn.experts.{e}.w1.scale"] = w, s
            w, s = fp4(MI, H)
            out[f"{p}.ffn.experts.{e}.w3.weight"], out[f"{p}.ffn.experts.{e}.w3.scale"] = w, s
            w, s = fp4(H, MI)
            out[f"{p}.ffn.experts.{e}.w2.weight"], out[f"{p}.ffn.experts.{e}.w2.scale"] = w, s
        if ratio:
            out[f"{p}.attn.compressor.ape"] = f32(ratio, coff * HEAD_DIM)
            out[f"{p}.attn.compressor.wkv.weight"] = bf16(coff * HEAD_DIM, H)
            out[f"{p}.attn.compressor.wgate.weight"] = bf16(coff * HEAD_DIM, H)
            out[f"{p}.attn.compressor.norm.weight"] = bf16(HEAD_DIM)
            if ratio == 4:
                out[f"{p}.attn.indexer.wq_b.weight"], out[f"{p}.attn.indexer.wq_b.scale"] = fp8(INDEX_HEADS * INDEX_DIM, QL)
                out[f"{p}.attn.indexer.weights_proj.weight"] = bf16(INDEX_HEADS, H)
                out[f"{p}.attn.indexer.compressor.ape"] = f32(ratio, coff * INDEX_DIM)
                out[f"{p}.attn.indexer.compressor.wkv.weight"] = bf16(coff * INDEX_DIM, H)
                out[f"{p}.attn.indexer.compressor.wgate.weight"] = bf16(coff * INDEX_DIM, H)
                out[f"{p}.attn.indexer.compressor.norm.weight"] = bf16(INDEX_DIM)
    return out


def shard_headers(model_dir, shards):
    for shard in shards:
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise RuntimeError(f"{shard}: too short for header length")
            n = struct.unpack("<Q", raw)[0]
            f.seek(0)
            raw = f.read(8 + n)
        yield shard, json.loads(raw[8:].decode())


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model/DeepSeek-V4-Flash-0731"
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    wm = idx["weight_map"]
    total_size = idx.get("metadata", {}).get("total_size", 0)

    expected = expected_tensors(cfg)
    main = set(k for k in wm if not k.startswith("mtp."))
    mtp = set(k for k in wm if k.startswith("mtp."))
    if len(main) != 67612:
        print(f"FAIL: {len(main)} main tensors, expected 67612", file=sys.stderr)
        return 1
    if len(mtp) != 4705:
        print(f"FAIL: {len(mtp)} mtp tensors, expected 4705", file=sys.stderr)
        return 1
    if total_size != 166878536440:
        print(f"FAIL: total_size {total_size}", file=sys.stderr)
        return 1
    if len(expected) != 67612:
        print(f"FAIL: expected contract has {len(expected)} entries, expected 67612", file=sys.stderr)
        return 1

    shards = sorted({v for v in wm.values()})
    errors = 0
    seen = set()
    for shard, header in shard_headers(model_dir, shards):
        for name, meta in header.items():
            if isinstance(meta, dict):
                if name.startswith("mtp."):
                    continue
                seen.add(name)
                want = expected.get(name)
                if want is None:
                    print(f"FAIL: unexpected tensor {name}", file=sys.stderr)
                    errors += 1
                    continue
                wdtype, wshape = want
                if meta["dtype"] != wdtype or meta["shape"] != wshape:
                    print(f"FAIL: {name} dtype/shape {meta['dtype']} {meta['shape']} != {wdtype} {wshape}", file=sys.stderr)
                    errors += 1
    missing = main - seen
    for name in sorted(missing):
        print(f"FAIL: missing tensor {name}", file=sys.stderr)
        errors += 1
    if errors:
        print(f"FAIL: {errors} contract errors", file=sys.stderr)
        return 1
    print(f"verify_dsv4_checkpoint: OK - {len(seen)} main tensors verified, {len(mtp)} mtp.* reported, total {total_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
