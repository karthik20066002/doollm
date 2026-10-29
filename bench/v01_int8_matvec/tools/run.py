#!/usr/bin/env python3


import argparse
import re
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
TOOLS = ROOT / "tools"

DEFAULT_TOKENS = ["256", "512", "1024", "2048", "4096", "8192", "16384:1"]


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def parse_block(pattern: str, txt: str, ngroup: int = 1):
    return {int(m.group(ngroup)): m.group(0) for m in re.finditer(pattern, txt, re.M)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "tokens",
        nargs="*",
        default=DEFAULT_TOKENS,
        help="e.g. 256 (both modes) or 256:1 (uint8 only)",
    )
    ap.add_argument("--iwad", default="/usr/share/games/doom/DOOM2.WAD")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    pairs = []
    for tok in args.tokens:
        size_s, _, mode_s = tok.partition(":")
        size = int(size_s)
        if mode_s:
            pairs.append((size, int(mode_s)))
        else:
            pairs.extend([(size, 0), (size, 1)])
    sizes = sorted({n for n, _ in pairs})

    BUILD.mkdir(exist_ok=True)

    print(f"[1/4] generating INT8 data for sizes {sizes}")
    sh([sys.executable, str(TOOLS / "gen_data.py"), *map(str, sizes)])
    (SRC / "data" / "benchspec.txt").write_text(" ".join(f"{n}{m}" for n, m in pairs))

    pk3 = BUILD / "matmul_v01.pk3"
    with zipfile.ZipFile(pk3, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(SRC).as_posix())
    print(f"[2/4] packaged {pk3.name} ({pk3.stat().st_size} bytes)")

    ref = sh([sys.executable, str(TOOLS / "reference.py"), *map(str, sizes)]).stdout
    ref_y = {
        int(m.group(1)): (m.group(2), m.group(3))
        for m in re.finditer(r"^Y size=(\d+) sum=(\d+) head=(.*)$", ref, re.M)
    }
    numpy_ms = {
        int(m.group(1)): float(m.group(2))
        for m in re.finditer(r"^NUMPY size=(\d+) ms_per_iter=([\d.]+)", ref, re.M)
    }
    print(f"[3/4] numpy reference computed ({len(ref_y)} sizes)")

    log = BUILD / "gzdoom.log"
    log.unlink(missing_ok=True)
    cmd = [
        "gzdoom",
        "-config",
        str(BUILD / "gzdoom.ini"),
        "-iwad",
        args.iwad,
        "-file",
        str(pk3),
        "+logfile",
        str(log),
        "+map",
        "MAP01",
        "-nomonsters",
        "-nosound",
        "-nomusic",
    ]
    print(f"[4/4] running gzdoom (log: {log})")
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    done = resumed = False
    try:
        deadline = time.time() + args.timeout
        while time.time() < deadline and proc.poll() is None:
            txt = log.read_text(errors="replace") if log.exists() else ""
            if "Script error" in txt:
                print("ZScript compile error; see", log)
                return 1
            if "MATMUL_DONE" in txt:
                done = True
                break
            time.sleep(0.25)
        if done:
            t0 = time.time()
            while time.time() - t0 < 5:
                txt = log.read_text(errors="replace")
                if "TICK_AFTER_BENCH" in txt:
                    resumed = True
                    break
                time.sleep(0.25)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    txt = log.read_text(errors="replace") if log.exists() else ""
    z_y = {
        (int(m.group(1)), int(m.group(2))): (m.group(3), m.group(4))
        for m in re.finditer(
            r"^Y size=(\d+) mode=(\d+) sum=(\d+) head=(.*)$", txt, re.M
        )
    }
    result = {
        (int(m.group(1)), int(m.group(2))): m.groupdict()
        for m in re.finditer(
            r"^RESULT size=(?P<size>\d+) mode=(?P<mode>\d+) iters=(?P<iters>\d+) "
            r"warmup_ms=(?P<warmup>[\d.]+) total_ms=(?P<total>[\d.]+) "
            r"ms_per_iter=(?P<per>[\d.]+) macs_per_sec=(?P<mps>[\d.]+)",
            txt,
            re.M,
        )
    }
    load = {
        (int(m.group(1)), int(m.group(2))): float(m.group(3))
        for m in re.finditer(
            r"^LOAD size=(\d+) mode=(\d+) bytes=\d+ ms=([\d.]+)", txt, re.M
        )
    }

    print("\n=== RESULTS ===")
    hdr = f"{'size':>6} {'mode':>4} {'ms/iter':>10} {'MACs/s':>12} {'load ms':>9} {'vs numpy':>9}  checksum"
    print(hdr)
    print("-" * len(hdr))
    ok = done and resumed
    mode_name = {0: "i32", 1: "u8"}
    for n, mode in pairs:
        r = result.get((n, mode))
        match = "MISSING"
        if (n, mode) in z_y and n in ref_y:
            match = "PASS" if z_y[(n, mode)] == ref_y[n] else "FAIL"
        if match != "PASS":
            ok = False
        if r:
            per = float(r["per"])
            mps = float(r["mps"])
            vs = f"{per / numpy_ms[n]:.0f}x" if n in numpy_ms else "?"
            print(
                f"{n:>6} {mode_name[mode]:>4} {per:>10.4f} {mps:>12.1f} "
                f"{load.get((n, mode), 0):>9.3f} {vs:>9}  {match}"
            )
        else:
            print(
                f"{n:>6} {mode_name[mode]:>4} {'--':>10} {'--':>12} {'--':>9} {'--':>9}  {match}"
            )
    print("-" * len(hdr))
    print(f"MATMUL_DONE seen: {done}   TICK_AFTER_BENCH (gameplay resumed): {resumed}")
    if not done:
        print("ERROR: benchmark did not complete; see", log)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
