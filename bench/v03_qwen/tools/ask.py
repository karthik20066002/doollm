#!/usr/bin/env python3


import argparse
import subprocess
import sys
from run import package
from prompts import DEFAULT_SYSTEM
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
TOOLS = ROOT / "tools"

IWAD = "/usr/share/games/doom/DOOM2.WAD"


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask Qwen2.5 inside Doom")
    ap.add_argument("query", nargs="+", help="user query")
    ap.add_argument("--iwad", default=IWAD, help="path to IWAD")
    ap.add_argument("--no-launch", action="store_true",
                    help="only repackage; don't launch GZDoom")
    args = ap.parse_args()

    query = " ".join(args.query)
    print(f"[1/3] tokenizing query: {query!r}")
    subprocess.run([sys.executable, str(TOOLS / "set_prompt.py"), query, "--idx", "0"],
                   check=True)

    subprocess.run([sys.executable, str(TOOLS / "prep_prompt.py")], check=True)
    if not (SRC / "data" / "wte.bin").exists():
        subprocess.run([sys.executable, str(TOOLS / "gen_qwen.py")], check=True)
    print("[2/3] repackaging pk3")
    pk3 = BUILD / "qwen_v03.pk3"
    package(pk3)

    if args.no_launch:
        print(f"[3/3] done ({pk3})")
        return 0

    print("[3/3] launching GZDoom")
    cmd = ["gzdoom", "-config", str(BUILD / "gzdoom.ini"), "-iwad", args.iwad,
           "-file", str(pk3), "+map", "MAP01", "-nomonsters", "-nosound", "-nomusic",
           "+qwen_idx", "0", "+qwen_prompt", query, "+qwen_system", DEFAULT_SYSTEM,
           "+qwen_max_tokens", "24", "+qwen_tokenize_only", "false", "+qwen_autorun", "true"]
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    sys.exit(main())
