#!/usr/bin/env python3

import argparse
from prompts import DEFAULT_SYSTEM, load_lib, save_lib, tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('query')
    ap.add_argument('--idx', type=int, default=0)
    ap.add_argument('--system', default=DEFAULT_SYSTEM)
    args = ap.parse_args()
    lib = load_lib()
    if not 0 <= args.idx <= len(lib):
        ap.error(f'slot must be between 0 and {len(lib)}')
    entry = {'system': args.system, 'query': args.query}
    if args.idx == len(lib):
        lib.append(entry)
    else:
        lib[args.idx] = entry
    try:
        save_lib(lib, tokenizer())
    except ValueError as e:
        ap.error(str(e))
    print(f'Updated prompt slot {args.idx}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
