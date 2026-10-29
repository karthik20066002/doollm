#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import psutil
from common import ROOT, BUILD, DEFAULT_SYSTEM, write_json


def run_engine(
    query="",
    max_tokens=32,
    budget=4,
    timeout=900,
    label="engine",
    compile_only=False,
    tokenize_only=False,
    extra_files=(),
    debug=True,
    system=DEFAULT_SYSTEM,
    package=False,
    thinking=False,
    profile=False,
):
    if not 1 <= budget <= 8:
        raise ValueError(
            "GZDoom work budget must be between 1 and 8 ms to keep rendering responsive"
        )
    BUILD.mkdir(exist_ok=True)
    log = BUILD / (label + ".log")
    stdout = BUILD / (label + ".stdout")
    log.unlink(missing_ok=True)
    cmd = [
        "gzdoom",
        "-iwad",
        "/usr/share/games/doom/DOOM2.WAD",
        "-file",
        str(BUILD / "lfm3.pk3" if package else ROOT / "src"),
        *map(str, extra_files),
        "-config",
        str(BUILD / (label + ".ini")),
        "-stdout",
        "-nosound",
        "-nomusic",
        "-nomonsters",
        "-window",
        "-width",
        "640",
        "-height",
        "480",
        "+vid_activeinbackground",
        "true",
        "+vid_lowerinbackground",
        "false",
        "+vid_maxfps",
        "120",
        "+vid_vsync",
        "false",
        "+logfile",
        str(log),
        "+llm_prompt",
        query,
        "+llm_system",
        system,
        "+llm_max_tokens",
        str(max_tokens),
        "+llm_budget_ms",
        str(budget),
        "+llm_thinking",
        str(thinking).lower(),
        "+llm_timeout_s",
        "0",
        "+llm_profile",
        str(profile).lower(),
        "+llm_debug",
        str(debug).lower(),
        "+llm_tokenize_only",
        str(tokenize_only).lower(),
    ]
    if compile_only:
        cmd += ["-norun"]
    else:
        cmd += ["+map", "MAP01", "+llm_autorun", "true" if query else "false"]
    start = time.monotonic()
    peak = 0
    text = ""
    status = "timeout"
    with stdout.open("w") as out:
        proc = subprocess.Popen(
            cmd, stdout=out, stderr=subprocess.STDOUT, start_new_session=True
        )
        handle = psutil.Process(proc.pid)
        try:
            while time.monotonic() - start < timeout:
                text = log.read_text(errors="replace") if log.exists() else ""
                try:
                    peak = max(peak, handle.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if any(
                    marker in text
                    for marker in (
                        "Script error",
                        "LLM_TIMEOUT",
                        "LLM_ERROR",
                        "VM execution aborted",
                        "Attempt to change scope",
                        "errors while parsing",
                    )
                ):
                    status = "error"
                    break
                if "LLM_DONE" in text or "TOKENIZE_DONE" in text:
                    status = "done"
                    break
                if proc.poll() is not None:
                    status = (
                        "compiled"
                        if compile_only and "script parsing took" in text
                        else "exited"
                    )
                    break
                time.sleep(0.1)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
    result = {
        "status": status,
        "elapsed_s": time.monotonic() - start,
        "peak_rss_bytes": peak,
        "log": str(log),
    }
    write_json(BUILD / (label + ".metrics.json"), result)
    print(json.dumps(result), flush=True)
    if status not in ("done", "compiled"):
        print(text[-6000:] or stdout.read_text()[-6000:])
        raise RuntimeError(f"GZDoom {status}; see {log}")
    return text, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--budget", type=float, default=4)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--label", default="engine")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tokenize", action="store_true")
    ap.add_argument("--package", action="store_true")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    run_engine(
        args.query,
        args.max_tokens,
        args.budget,
        args.timeout,
        args.label,
        args.compile,
        args.tokenize,
        package=args.package,
        thinking=args.thinking,
        profile=args.profile,
        debug=not args.profile,
    )


if __name__ == "__main__":
    main()
