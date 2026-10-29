#!/usr/bin/env python3


import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = __import__("pathlib").Path(__file__).resolve().parent
MODEL = HERE

SYSTEM = "You are a helpful assistant. Answer in one short sentence."
USER = "Give one survival tip for the UAC Mars base."
prompt = (
    f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    f"<|im_start|>user\n{USER}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

tok = AutoTokenizer.from_pretrained(str(MODEL))
ids = tok.encode(prompt, add_special_tokens=False)
print("ids:", ids[:14], "len", len(ids))

model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.float32)
model.eval()

with torch.no_grad():
    out = model(input_ids=torch.tensor([ids]))
logits = out.logits[0, -1].numpy()
top = np.argsort(logits)[-5:][::-1]
for t in top:
    print(f"TOP {t} {logits[t]:.4f} {tok.decode([int(t)])!r}")

with torch.no_grad():
    gen = model.generate(
        torch.tensor([ids]),
        max_new_tokens=24,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
print("reply:", repr(tok.decode(gen[0, len(ids) :], skip_special_tokens=False)))


hooks = []


def mk(i):
    def fn(m, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        print(f"HLN {i} {float(h[..., -1, :].norm()):.6f}")

    return fn


for i in range(4):
    hooks.append(model.model.layers[i].register_forward_hook(mk(i)))
with torch.no_grad():
    model(input_ids=torch.tensor([ids[:3]]))
for h in hooks:
    h.remove()
