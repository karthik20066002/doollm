#!/usr/bin/env python3

import argparse
import math
import re
import signal
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from prompts import DEFAULT_SYSTEM, MAXTOK, tokenize_prompt, tokenizer

ROOT = Path(__file__).resolve().parent.parent
SRC, BUILD, TOOLS = ROOT / "src", ROOT / "build", ROOT / "tools"
TOL = 1e-9
INT_KEYS = {"ESUM", "LQ", "LK", "LV", "LO", "LG", "LU", "LD", "TOK"}


def parse(txt):
    ints, values = {}, {}

    def put(dst, key, value):
        if key in dst:
            raise ValueError(f"duplicate record: {key}")
        dst[key] = value

    for line in txt.splitlines():
        key = line.partition(" ")[0]
        if key in INT_KEYS:
            m = re.fullmatch(r"(\w+) (\d+) (\d+)", line)
            if not m:
                raise ValueError(f"malformed record: {line!r}")
            put(ints, (key, m[2]), int(m[3]))
        elif key == "LOGIT0":
            fields = line.split()
            if len(fields) != 4:
                raise ValueError("malformed LOGIT0")
            put(ints, ("LOGIT0_ID", "-"), int(fields[1]))
            put(ints, ("LOGIT0_SUM", "-"), int(fields[3]))
            value = float(fields[2])
            if not math.isfinite(value):
                raise ValueError("non-finite LOGIT0")
            put(values, ("LOGIT0_VAL", "-"), value)
        elif key == "TTXT":
            m = re.fullmatch(r"TTXT (\d+) (.*)", line)
            if not m:
                raise ValueError("malformed TTXT")
            put(values, ("TTXT", m[1]), m[2])
        elif key in {"TEXT", "PIDS"}:
            if not line.startswith(key + " "):
                raise ValueError(f"malformed {key}")
            put(values, (key, "-"), line[len(key) + 1 :])
    return ints, values


def validate_records(ints, values):
    for key in ("LOGIT0_ID", "LOGIT0_SUM"):
        if (key, "-") not in ints:
            raise ValueError(f"missing {key}")
    for key in ("LOGIT0_VAL", "TEXT", "PIDS"):
        if (key, "-") not in values:
            raise ValueError(f"missing {key}")
    pids = values["PIDS", "-"].split()
    for idx in range(min(3, len(pids))):
        if ("ESUM", str(idx)) not in ints:
            raise ValueError("missing embedding checksum")
    for key in ("LQ", "LK", "LV", "LO", "LG", "LU", "LD"):
        for layer in (0, 12, 23, 99):
            if (key, str(layer)) not in ints:
                raise ValueError(f"missing {key} layer {layer}")
    indices = sorted(int(idx) for key, idx in ints if key == "TOK")
    if not indices or indices != list(range(len(indices))):
        raise ValueError("missing or non-contiguous TOK records")
    if {idx for key, idx in values if key == "TTXT"} != set(map(str, indices)):
        raise ValueError("missing or extra token text")


def compare(reference, actual):
    for records in (reference, actual):
        validate_records(*records)
    errors = []
    for ref, got in zip(reference, actual):
        if ref.keys() != got.keys():
            errors.append(
                f"record keys differ: missing={ref.keys() - got.keys()}, extra={got.keys() - ref.keys()}"
            )
        for key, rv in ref.items():
            gv = got.get(key)
            if isinstance(rv, float):
                equal = (
                    isinstance(gv, (int, float))
                    and math.isfinite(gv)
                    and abs(gv - rv) <= TOL * max(1.0, abs(rv))
                )
            else:
                equal = gv == rv
            if not equal:
                errors.append(f"{key}: ref={rv!r}, zscript={gv!r}")
    return errors


def sh(cmd, timeout=None):
    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(f"{cmd[0]} exited {result.returncode}:\n{result.stdout}")
    return result.stdout


def package(pk3):
    pk3.parent.mkdir(parents=True, exist_ok=True)
    tmp = pk3.with_suffix(".tmp.pk3")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(SRC.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(SRC).as_posix())
    tmp.replace(pk3)


def main():
    ap = argparse.ArgumentParser(description="Run the Qwen benchmark in GZDoom.")
    ap.add_argument("--iwad", default="/usr/share/games/doom/DOOM2.WAD")
    ap.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="timeout per inference process, seconds",
    )
    ap.add_argument(
        "--skip-weights", action="store_true", help="reuse generated weight lumps"
    )
    ap.add_argument("--query", help="verify a custom console prompt")
    ap.add_argument(
        "--max-tokens", type=int, default=MAXTOK, choices=range(1, MAXTOK + 1)
    )
    args = ap.parse_args()
    BUILD.mkdir(exist_ok=True)
    try:
        if not args.skip_weights:
            print("[1/5] generating weights", flush=True)
            print(sh([sys.executable, str(TOOLS / "gen_qwen.py")]))
        print("[2/5] preparing tokenizer tables", flush=True)
        print(sh([sys.executable, str(TOOLS / "prep_prompt.py")]))
        prompt = SRC / "data" / "prompt_0.bin"
        if args.query is not None:
            ids = tokenize_prompt(tokenizer(), DEFAULT_SYSTEM, args.query)
            prompt = BUILD / "verify_prompt.bin"
            prompt.write_bytes(struct.pack(f"<{len(ids) + 1}i", len(ids), *ids))
        pk3 = BUILD / "qwen_v03.pk3"
        print("[3/5] packaging", flush=True)
        package(pk3)
        print("[4/5] NumPy reference", flush=True)
        ref_txt = sh(
            [
                sys.executable,
                str(TOOLS / "reference_qwen.py"),
                "--prompt",
                str(prompt),
                "--max-tokens",
                str(args.max_tokens),
            ],
            timeout=args.timeout,
        )
        (BUILD / "reference.txt").write_text(ref_txt)
        reference = parse(ref_txt)
        validate_records(*reference)
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
            "-stdout",
            "+map",
            "MAP01",
            "-nomonsters",
            "-nosound",
            "-nomusic",
            "+qwen_idx",
            "0",
            "+qwen_prompt",
            args.query or "",
            "+qwen_system",
            DEFAULT_SYSTEM,
            "+qwen_max_tokens",
            str(args.max_tokens),
            "+qwen_tokenize_only",
            "false",
            "+qwen_autorun",
            "true",
        ]
        print("[5/5] GZDoom inference", flush=True)
        done = resumed = False
        with (BUILD / "stdout.log").open("wb") as sout:
            proc = subprocess.Popen(cmd, stdout=sout, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    txt = log.read_text(errors="replace") if log.exists() else ""
                    done = "MATMUL_DONE" in txt
                    resumed = "TICK_AFTER_BENCH" in txt
                    if (
                        (done and resumed)
                        or "QWEN_ERROR" in txt
                        or "Script error" in txt
                        or proc.poll() is not None
                    ):
                        break
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    proc.send_signal(signal.SIGTERM)
                    try:
                        proc.wait(5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        actual = parse(log.read_text(errors="replace") if log.exists() else "")
        errors = compare(reference, actual)
        for error in errors:
            print("FAIL", error)
        ok = not errors and done and resumed
        print(
            f"checked {sum(map(len, reference))} records; MATMUL_DONE={done} resumed={resumed}"
        )
        print("reply:", actual[1].get(("TEXT", "-"), ""))
        print("VERDICT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"FAIL: {e}", file=sys.stderr)
        print(f"Diagnostics: {BUILD}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
