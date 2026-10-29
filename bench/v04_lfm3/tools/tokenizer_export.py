#!/usr/bin/env python3


import json
import struct
from pathlib import Path

from tokenizers import Regex, pre_tokenizers
from common import DATA, MODEL

VOCAB = 128000


def byte_alphabet():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = list(bs)
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(bs, map(chr, chars)))


def write_tables(data=DATA):
    config = json.loads((MODEL / "tokenizer.json").read_text())
    vocab = config["model"]["vocab"]
    alphabet = byte_alphabet()
    inverse = {c: b for b, c in alphabet.items()}
    table = [b""] * VOCAB
    for token, tid in vocab.items():
        table[tid] = bytes(inverse[c] for c in token)
    for token in config["added_tokens"]:
        table[token["id"]] = token["content"].encode("utf-8")
    with (data / "decode.bin").open("wb") as f:
        for b in table:
            if len(b) > 255:
                raise ValueError("decode token too long for format")
            f.write(bytes([len(b)]) + b)

    merges = config["model"]["merges"]
    with (data / "bpe.bin").open("wb") as f:
        f.write(struct.pack("<256i", *(vocab[alphabet[b]] for b in range(256))))
        f.write(struct.pack("<i", len(merges)))
        for merge in merges:
            a, b = merge.split(" ") if isinstance(merge, str) else merge
            f.write(struct.pack("<3i", vocab[a], vocab[b], vocab[a + b]))

    ranges = []
    for start, end in [(0, 0xD800), (0xE000, 0x110000)]:
        chars = "".join(map(chr, range(start, end)))
        for kind, pattern in enumerate([r"\p{L}+", r"\p{N}+", r"\s+"], 1):
            split = pre_tokenizers.Split(
                Regex(pattern), behavior="removed", invert=True
            )
            for _, (a, b) in split.pre_tokenize_str(chars):
                ranges.append((start + a, start + b - 1, kind))
    with (data / "unicode.bin").open("wb") as f:
        for row in sorted(ranges):
            f.write(struct.pack("<3i", *row))

    (data / "special.txt").write_text(
        "\n".join(t["content"] for t in config["added_tokens"])
    )
    (data / "special.bin").write_bytes(
        struct.pack(
            f"<{len(config['added_tokens'])}i",
            *(t["id"] for t in config["added_tokens"]),
        )
    )
    (data / "benchspec.txt").write_text("lfm3")
    return table


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    table = write_tables()
    print("Tokenizer ready:", len(table), "IDs, max bytes", max(map(len, table)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
