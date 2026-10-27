#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "src" / "data"


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [256]
    DATA.mkdir(parents=True, exist_ok=True)
    for n in sizes:
        rng = np.random.default_rng(1000 + n)
        w = rng.integers(-128, 128, size=(n, n), dtype=np.int8)
        x = rng.integers(-128, 128, size=(n,), dtype=np.int8)

        wb = (w.astype(np.int16) + 128).astype(np.uint8)
        xb = (x.astype(np.int16) + 128).astype(np.uint8)
        (DATA / f"w{n}.bin").write_bytes(wb.tobytes(order="C"))
        (DATA / f"x{n}.bin").write_bytes(xb.tobytes(order="C"))
        print(f"gen size={n} w_bytes={wb.nbytes} x_bytes={xb.nbytes}")


if __name__ == "__main__":
    main()
