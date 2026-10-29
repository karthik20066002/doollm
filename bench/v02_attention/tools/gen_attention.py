#!/usr/bin/env python3


import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "src" / "data"
SAFE = HERE / "gpt2.safetensors"

TOKENS = [464, 2068, 7586, 21831]
FX = 1 << 40
D, QKV = 768, 2304


def load_safetensors(path: Path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    base = 8 + hlen
    out = {}
    for name, meta in hdr.items():
        if name == "__metadata__":
            continue
        assert meta["dtype"] == "F32", meta
        a0, a1 = meta["data_offsets"]
        arr = np.memmap(
            path, dtype="<f4", mode="r", offset=base + a0, shape=tuple(meta["shape"])
        )
        out[name] = np.array(arr, dtype=np.float64)
    return out


def quant_rows(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    scale = np.abs(w).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    q = np.rint(w / scale).astype(np.int64).astype(np.int8)
    return q, scale


def write_fx(path: Path, v: np.ndarray) -> None:

    np.asarray(v, dtype="<f8").tofile(path)


def write_u8(path: Path, q: np.ndarray) -> None:
    (q.astype(np.int16) + 128).astype(np.uint8).tofile(path)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    t = load_safetensors(SAFE)

    emb = t["wte.weight"][TOKENS]
    write_fx(DATA / "emb.bin", emb)
    write_fx(DATA / "lng.bin", t["h.0.ln_1.weight"])
    write_fx(DATA / "lnb.bin", t["h.0.ln_1.bias"])
    print(f"emb {emb.shape} range [{emb.min():.3f},{emb.max():.3f}]")

    wq, sq = quant_rows(t["h.0.attn.c_attn.weight"])
    write_u8(DATA / "wqkv.bin", wq)
    write_fx(DATA / "sqkv.bin", sq)
    write_fx(DATA / "bqkv.bin", t["h.0.attn.c_attn.bias"])
    nz = (wq != 0).mean()
    print(
        f"wqkv {wq.shape} nonzero={nz:.2%} scale range [{sq.min():.5f},{sq.max():.5f}]"
    )

    wp, sp = quant_rows(t["h.0.attn.c_proj.weight"])
    write_u8(DATA / "wproj.bin", wp)
    write_fx(DATA / "sproj.bin", sp)
    write_fx(DATA / "bproj.bin", t["h.0.attn.c_proj.bias"])
    print(f"wproj {wp.shape} nonzero={(wp != 0).mean():.2%}")

    (DATA / "benchspec.txt").write_text("attn")
    print("lumps written to", DATA)


if __name__ == "__main__":
    sys.exit(main())
