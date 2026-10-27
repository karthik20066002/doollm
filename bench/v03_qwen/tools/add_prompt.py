#!/usr/bin/env python3

import argparse
from prompts import DEFAULT_SYSTEM, load_lib, save_lib, tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('query', nargs='?')
    g.add_argument('--list', action='store_true')
    g.add_argument('--clear', action='store_true')
    args = ap.parse_args()
    lib = load_lib()
    if args.clear:
        lib = []
    elif args.query is not None:
        lib.append({'system': DEFAULT_SYSTEM, 'query': args.query})
    if not args.list:
        try:
            save_lib(lib, tokenizer())
        except ValueError as e:
            ap.error(str(e))
    print(f'prompt library: {len(lib)} entries')
    for i, e in enumerate(lib):
        print(f"  {i}: {e['query']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
