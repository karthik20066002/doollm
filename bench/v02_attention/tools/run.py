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

EXACT_KEYS = {"XQ", "QKVINT", "ATTINT", "PROJINT"}
FLOAT_KEYS = {"LNO", "P", "Y", "YSUM"}
LINE_RE = re.compile(r"^(XQ|QKVINT|ATTINT|PROJINT|LNO|P|Y|YSUM) (\S+) (\S+)$", re.M)
TOL = 1e-9


def sh(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def parse(txt: str) -> dict:
    out = {}
    for m in LINE_RE.finditer(txt):
        key, idx, val = m.group(1), m.group(2), m.group(3)
        if key in EXACT_KEYS:
            out[(key, idx)] = int(val)
        else:
            out[(key, idx)] = float(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iwad", default="/usr/share/games/doom/DOOM2.WAD")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    BUILD.mkdir(exist_ok=True)

    print("[1/4] extracting GPT-2 attention weights")
    sh([sys.executable, str(TOOLS / "gen_attention.py")])

    pk3 = BUILD / "attention_v02.pk3"
    with zipfile.ZipFile(pk3, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(SRC).as_posix())
    print(f"[2/4] packaged {pk3.name} ({pk3.stat().st_size} bytes)")

    ref_txt = sh([sys.executable, str(TOOLS / "reference_attention.py")]).stdout
    ref = parse(ref_txt)
    print(f"[3/4] numpy reference computed ({len(ref)} values)")

    log = BUILD / "gzdoom.log"
    log.unlink(missing_ok=True)
    cmd = ["gzdoom", "-config", str(BUILD / "gzdoom.ini"), "-iwad", args.iwad,
           "-file", str(pk3), "+logfile", str(log),
           "+map", "MAP01", "-nomonsters", "-nosound", "-nomusic"]
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
    z = parse(txt)
    time_m = re.search(r"^TIME (.*)$", txt, re.M)
    load_m = re.search(r"^LOAD (.*)$", txt, re.M)

    print("\n=== RESULTS ===")
    if load_m:
        print("LOAD", load_m.group(1))
    if time_m:
        print("TIME", time_m.group(1))

    fails = 0
    checked = 0
    for key in sorted(EXACT_KEYS | FLOAT_KEYS):
        idxs = sorted({idx for (k, idx) in ref if k == key},
                      key=lambda s: (len(s), s))
        for idx in idxs:
            checked += 1
            rk, zk = ref.get((key, idx)), z.get((key, idx))
            if zk is None:
                print(f"MISSING  {key} {idx}")
                fails += 1
            elif key in EXACT_KEYS:
                if rk != zk:
                    print(f"FAIL     {key} {idx}: ref={rk} zscript={zk}")
                    fails += 1
            else:
                if abs(zk - rk) > TOL * max(1.0, abs(rk)):
                    print(f"FAIL     {key} {idx}: ref={rk:.17g} zscript={zk:.17g}")
                    fails += 1
    ok = fails == 0 and done and resumed and checked == len(ref)
    print(f"\nchecked {checked} reference values, {fails} mismatches")
    print(f"MATMUL_DONE: {done}   gameplay resumed: {resumed}")
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
