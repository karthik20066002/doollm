#!/usr/bin/env python3

import hashlib
import json
import numpy as np
import torch
from safetensors import safe_open
from common import MODEL, DATA, CONFIG, ROPE_TABLE_POSITIONS, write_json


def main():
    torch.set_num_threads(4)
    DATA.mkdir(parents=True, exist_ok=True)
    specs = []
    with safe_open(MODEL / "model.safetensors", framework="pt", device="cpu") as f:
        for i, name in enumerate(sorted(f.keys())):
            tensor = f.get_tensor(name)
            shape = list(tensor.shape)
            quant = len(shape) == 2
            stem = f"t{i:03d}"
            path = DATA / (stem + ".bin")
            scales = []
            with path.open("wb") as out:
                if quant:
                    for start in range(0, shape[0], 256):
                        w = tensor[start : start + 256].double().numpy()
                        s = np.abs(w).max(axis=1) / 127.0
                        s[s == 0] = 1.0
                        q = np.rint(w / s[:, None]).astype(np.int16) + 128
                        q.astype("u1").tofile(out)
                        scales.extend(s)
                    np.asarray(scales, dtype="<f8").tofile(DATA / (stem + ".scale"))
                else:
                    tensor.double().numpy().astype("<f8").tofile(out)
            specs.append(
                {
                    "name": name,
                    "stem": stem,
                    "shape": shape,
                    "format": "u8-row" if quant else "f64",
                    "bytes": path.stat().st_size,
                }
            )
            print(name, shape, specs[-1]["format"], flush=True)
    for kind, params in CONFIG["rope_parameters"].items():
        dim = CONFIG["head_dim"]
        angles = int(dim * params.get("partial_rotary_factor", 1.0)) / 2
        inv = np.zeros(dim // 2, dtype=np.float64)
        inv[: int(angles)] = params["rope_theta"] ** (
            -np.arange(0, int(angles) * 2, 2, dtype=np.float64) / dim
        )
        a = np.arange(ROPE_TABLE_POSITIONS)[:, None] * inv
        np.concatenate([np.cos(a).ravel(), np.sin(a).ravel()]).astype("<f8").tofile(
            DATA / f"rope_{kind}.bin"
        )
    write_json(
        DATA / "weights.json",
        {
            "format_version": 1,
            "precision": "W8 with floating activations and caches",
            "config": CONFIG,
            "rope_table_positions": ROPE_TABLE_POSITIONS,
            "tensors": specs,
        },
    )
    print("Export bytes", sum(p.stat().st_size for p in DATA.glob("*")), flush=True)


if __name__ == "__main__":
    main()
