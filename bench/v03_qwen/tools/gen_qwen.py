#!/usr/bin/env python3


import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "src" / "data"
SAFE = HERE / "model.safetensors"
CONF = HERE / "config.json"

MAXPOS = 128
CHUNK = 8192


def cfg() -> dict:
    c = json.loads(CONF.read_text())
    expected = {
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": True,
    }
    for key, value in expected.items():
        if c.get(key) != value:
            raise ValueError(
                f"unsupported model config: {key}={c.get(key)!r}; expected {value!r}"
            )
    return dict(
        hidden=c["hidden_size"],
        inter=c["intermediate_size"],
        layers=c["num_hidden_layers"],
        heads=c["num_attention_heads"],
        kv=c["num_key_value_heads"],
        hd=c["hidden_size"] // c["num_attention_heads"],
        theta=c["rope_theta"],
        eps=c["rms_norm_eps"],
        vocab=c["vocab_size"],
    )


def safetensors_index(path: Path) -> dict:
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    return {
        k: (base + v["data_offsets"][0], v["dtype"], v["shape"])
        for k, v in hdr.items()
        if k != "__metadata__"
    }


def read_tensor(idx: dict, name: str) -> np.ndarray:
    off, dtype, shape = idx[name]
    n = int(np.prod(shape))
    if dtype == "BF16":
        u = np.memmap(SAFE, dtype="<u2", mode="r", offset=off, shape=n)
        f = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    elif dtype == "F16":
        f = np.memmap(SAFE, dtype="<f2", mode="r", offset=off, shape=n).astype(
            np.float64
        )
    elif dtype == "F32":
        f = np.memmap(SAFE, dtype="<f4", mode="r", offset=off, shape=n).astype(
            np.float64
        )
    else:
        raise ValueError(f"{name}: unsupported dtype {dtype}")
    return f.reshape(shape)


def quant_rows(w: np.ndarray, wf) -> np.ndarray:

    out, inn = w.shape
    scale = np.empty(out, dtype=np.float64)
    for r0 in range(0, out, CHUNK):
        r1 = min(r0 + CHUNK, out)
        blk = w[r0:r1]
        s = np.abs(blk).max(axis=1) / 127.0
        s[s == 0] = 1.0
        q = np.rint(blk / s[:, None]).astype(np.int16) + 128
        q.astype(np.uint8).tofile(wf)
        scale[r0:r1] = s
    return scale


def write_f64(path: Path, v: np.ndarray) -> None:
    np.ascontiguousarray(v, dtype="<f8").tofile(path)


def main() -> int:
    if not SAFE.exists():
        print(
            "model.safetensors missing; place Qwen2.5-0.5B-Instruct assets in tools/ (see README.md)"
        )
        return 1
    DATA.mkdir(parents=True, exist_ok=True)
    c = cfg()
    L, H, HD, KV = c["layers"], c["hidden"], c["hd"], c["kv"]
    idx = safetensors_index(SAFE)
    print(f"config: {c}")

    inv = 1.0 / (c["theta"] ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    ang = np.arange(MAXPOS, dtype=np.float64)[:, None] * inv[None, :]
    write_f64(DATA / "rope.bin", np.concatenate([np.cos(ang), np.sin(ang)]))

    wte = read_tensor(idx, "model.embed_tokens.weight")
    with open(DATA / "wte.bin", "wb") as wf:
        swte = quant_rows(wte, wf)
    write_f64(DATA / "swte.bin", swte)

    tables = {
        "q": ("self_attn.q_proj.weight", "wq.bin", "sq.bin"),
        "k": ("self_attn.k_proj.weight", "wk.bin", "sk.bin"),
        "v": ("self_attn.v_proj.weight", "wv.bin", "sv.bin"),
        "o": ("self_attn.o_proj.weight", "wo.bin", "so.bin"),
        "g": ("mlp.gate_proj.weight", "wg.bin", "sg.bin"),
        "u": ("mlp.up_proj.weight", "wu.bin", "su.bin"),
        "d": ("mlp.down_proj.weight", "wd.bin", "sd.bin"),
    }
    n1 = np.empty((L, H), dtype=np.float64)
    n2 = np.empty((L, H), dtype=np.float64)
    bq = np.empty((L, H), dtype=np.float64)
    bkv = np.empty((L, 2, KV * HD), dtype=np.float64)
    for i in range(L):
        for key, (suffix, wname, sname) in tables.items():
            w = read_tensor(idx, f"model.layers.{i}.{suffix}")
            with open(DATA / wname, "ab" if i else "wb") as wf:
                s = quant_rows(w, wf)
            with open(DATA / sname, "ab" if i else "wb") as sf:
                np.asarray(s, dtype="<f8").tofile(sf)
        bq[i] = read_tensor(idx, f"model.layers.{i}.self_attn.q_proj.bias")
        bkv[i, 0] = read_tensor(idx, f"model.layers.{i}.self_attn.k_proj.bias")
        bkv[i, 1] = read_tensor(idx, f"model.layers.{i}.self_attn.v_proj.bias")
        n1[i] = read_tensor(idx, f"model.layers.{i}.input_layernorm.weight")
        n2[i] = read_tensor(idx, f"model.layers.{i}.post_attention_layernorm.weight")
        if i in (0, L - 1) or (i + 1) % 8 == 0:
            print(f"layer {i + 1}/{L}")
    write_f64(DATA / "norm1.bin", n1)
    write_f64(DATA / "norm2.bin", n2)
    write_f64(DATA / "normf.bin", read_tensor(idx, "model.norm.weight"))
    write_f64(DATA / "bq.bin", bq)
    write_f64(DATA / "bk.bin", bkv[:, 0])
    write_f64(DATA / "bv.bin", bkv[:, 1])

    (DATA / "benchspec.txt").write_text("qwen")
    total = sum(p.stat().st_size for p in DATA.glob("*.bin"))
    print(f"lumps written: {total / 1e6:.0f} MB -> {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
