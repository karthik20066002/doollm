#!/usr/bin/env python3

import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model'
REVISION = 'ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7'
REPO = 'LiquidAI/LFM3-1.7B'
FILES = ['config.json', 'configuration_lfm3.py', 'modeling_lfm3.py', 'lfm3_conv.py',
         'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja',
         'generation_config.json', 'LICENSE', 'tokenizer.json', 'model.safetensors']


def main():
    MODEL.mkdir(parents=True, exist_ok=True)
    manifest = {'repo': REPO, 'revision': REVISION, 'files': {}}
    for name in FILES:
        dest = MODEL / name
        if not dest.exists():
            print('Fetching', name, flush=True)
            request = urllib.request.Request(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}')
            temp = dest.with_suffix(dest.suffix + '.part')
            with urllib.request.urlopen(request, timeout=90) as r, temp.open('wb') as f:
                while block := r.read(8 * 1024 * 1024):
                    f.write(block)
            temp.replace(dest)
        with dest.open('rb') as f:
            sha = hashlib.file_digest(f, 'sha256').hexdigest()
        manifest['files'][name] = {'bytes': dest.stat().st_size, 'sha256': sha}
        print(name, dest.stat().st_size, sha, flush=True)
    (ROOT / 'assets.lock.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
