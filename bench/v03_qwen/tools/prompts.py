
import json
import struct
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "src" / "data"
DEFAULT_SYSTEM = "You are a helpful assistant. Answer in one short sentence."
DEFAULT_QUERY = "Give one survival tip for the UAC Mars base."
MAXPOS, MAXTOK, VOCAB = 128, 24, 151936
MAXPROMPT = MAXPOS - MAXTOK + 1


def tokenizer():
    return Tokenizer.from_file(str(HERE / "tokenizer.json"))


def tokenize_prompt(tok, system, user):
    text = (f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n")
    ids = tok.encode(text, add_special_tokens=False).ids
    if tok.decode(ids, skip_special_tokens=False) != tok.normalizer.normalize_str(text):
        raise ValueError("prompt does not round-trip through the tokenizer")
    validate_ids(ids)
    return ids


def validate_ids(ids):
    if not 1 <= len(ids) <= MAXPROMPT:
        raise ValueError(f"prompt has {len(ids)} tokens; limit is {MAXPROMPT} including chat template")
    if any(t < 0 or t >= VOCAB for t in ids):
        raise ValueError("invalid prompt token ID")


def load_lib(data=DATA):
    path = data / "prompt_lib.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_lib(lib, tok, data=DATA):

    encoded = [tokenize_prompt(tok, e.get("system", DEFAULT_SYSTEM), e["query"]) for e in lib]
    data.mkdir(parents=True, exist_ok=True)
    keep = set()
    for idx, ids in enumerate(encoded):
        blob = struct.pack(f"<{len(ids) + 1}i", len(ids), *ids)
        name = f"prompt_{idx}.bin"
        (data / name).write_bytes(blob)
        keep.add(name)
        if idx == 0:
            (data / "prompt.bin").write_bytes(blob)
    for path in data.glob("prompt_*.bin"):
        if path.name not in keep:
            path.unlink()
    if not lib:
        (data / "prompt.bin").unlink(missing_ok=True)
    (data / "prompt_lib.json").write_text(json.dumps(lib, indent=2, ensure_ascii=False) + "\n")
