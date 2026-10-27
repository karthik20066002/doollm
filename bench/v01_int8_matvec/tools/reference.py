#!/usr/bin/env python3

import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "src" / "data"


def main() -> None:
    sizes = [int(a) for a in sys.argv[1:]] or [256]
    for n in sizes:
        wb = np.frombuffer((DATA / f"w{n}.bin").read_bytes(), dtype=np.uint8)
        x = (np.frombuffer((DATA / f"x{n}.bin").read_bytes(), dtype=np.uint8)
             .astype(np.int32) - 128)

        y = np.empty(n, dtype=np.int32)
        for r0 in range(0, n, 4096):
            r1 = min(r0 + 4096, n)
            y[r0:r1] = (wb[r0 * n:r1 * n].astype(np.int32) - 128).reshape(-1, n) @ x

        checksum = int(np.sum(y, dtype=np.int64)) % (1 << 32)
        head = " ".join(str(int(v)) for v in y[:16])
        print(f"Y size={n} sum={checksum} head={head}")

        def matvec() -> None:
            for r0 in range(0, n, 4096):
                r1 = min(r0 + 4096, n)
                y[r0:r1] = (wb[r0 * n:r1 * n].astype(np.int32) - 128).reshape(-1, n) @ x

        est = 0.0
        for _ in range(3):
            t0 = time.perf_counter()
            matvec()
            est += time.perf_counter() - t0
        iters = max(10, min(100000, int(0.3 / max(est / 3, 1e-9))))
        t0 = time.perf_counter()
        for _ in range(iters):
            matvec()
        ms = (time.perf_counter() - t0) / iters * 1000
        print(f"NUMPY size={n} ms_per_iter={ms:.6f}")


if __name__ == "__main__":
    main()
