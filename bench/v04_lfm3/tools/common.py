
from pathlib import Path
import json
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL, DATA, BUILD = ROOT/'model', ROOT/'src/data', ROOT/'build'
CONFIG = json.loads((MODEL/'config.json').read_text())
ROPE_TABLE_POSITIONS = 512
DEFAULT_SYSTEM = 'You are a helpful assistant. Answer briefly and accurately.'


def tokenizer():
    return Tokenizer.from_file(str(MODEL/'tokenizer.json'))


def prompt_text(query, system=DEFAULT_SYSTEM, thinking=False):
    return ('<|startoftext|>' + (f'<|im_start|>system\n{system}<|im_end|>\n' if system else '')
            + f'<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n<think>'
            + ('' if thinking else '</think>'))


def prompt_ids(query, system=DEFAULT_SYSTEM, thinking=False):
    ids = tokenizer().encode(prompt_text(query, system, thinking), add_special_tokens=False).ids
    return ids


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
