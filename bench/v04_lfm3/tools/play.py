#!/usr/bin/env python3

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch Doom with the LFM3 console.")
    parser.add_argument("--iwad", default="/usr/share/games/doom/DOOM2.WAD")
    parser.add_argument("--query", default="")
    args = parser.parse_args()
    build = Path(__file__).resolve().parents[1] / "build"
    package = build / "lfm3.pk3"
    if not package.is_file():
        parser.error("Build lfm3.pk3 with tools/package.py first.")
    command = [
        "gzdoom",
        "-iwad",
        args.iwad,
        "-file",
        str(package),
        "-config",
        str(build / "play.ini"),
        "-window",
        "-width",
        "1280",
        "-height",
        "720",
        "+vid_activeinbackground",
        "true",
        "+vid_lowerinbackground",
        "false",
        "+vid_maxfps",
        "120",
        "+vid_vsync",
        "false",
        "+llm_budget_ms",
        "4",
        "+llm_timeout_s",
        "0",
        "+llm_max_tokens",
        "0",
        "+llm_thinking",
        "true",
        "+llm_debug",
        "false",
        "+llm_tokenize_only",
        "false",
        "+llm_prompt",
        args.query,
        "+llm_autorun",
        "true" if args.query else "false",
        "+logfile",
        str(build / "play.log"),
        "+map",
        "MAP01",
        "+toggleconsole",
    ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
