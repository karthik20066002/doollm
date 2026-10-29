#!/usr/bin/env python3

import json
import struct
from common import MODEL, ROOT, CONFIG


def main():
    with (MODEL / "model.safetensors").open("rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    names = sorted(k for k in header if k != "__metadata__")
    stems = {name: f"t{i:03d}" for i, name in enumerate(names)}

    def matrix(target, name):
        nr, nc = header[name]["shape"]
        return f'        {target}=Matrix("{stems[name]}",{nr},{nc});'

    def vector(target, name):
        count = 1
        for n in header[name]["shape"]:
            count *= n
        return f'        Vector("{stems[name]}",{target},{count});'

    lines = [
        "class LFMModel : LFMRuntime {",
        "    override bool Configure() {",
        matrix("embedding", "model.embed_tokens.weight"),
        vector("finalNorm", "model.embedding_norm.weight"),
        '        Vector("rope_full_attention",ropeFull,65536);',
        '        Vector("rope_sliding_attention",ropeSliding,65536);',
        "        layers.Resize(24);",
    ]
    for i, kind in enumerate(CONFIG["layer_types"]):
        p = f"model.layers.{i}."
        v = f"layers[{i}]"
        lines.extend(
            [
                f'        {v}=new("LFMBlock");',
                f"        {v}.kind={dict(conv=0, full_attention=1, sliding_attention=2)[kind]};",
            ]
        )
        for target, suffix in [
            ("norm", "operator_norm"),
            ("postNorm", "post_operator_norm"),
            ("ffnNorm", "ffn_norm"),
            ("postFfnNorm", "post_ffn_norm"),
        ]:
            lines.append(vector(v + "." + target, p + suffix + ".weight"))
        for target, suffix in [("w1", "w1"), ("w2", "w2"), ("w3", "w3")]:
            lines.append(
                matrix(v + "." + target, p + "feed_forward." + suffix + ".weight")
            )
        if kind == "conv":
            lines.extend(
                [
                    matrix(v + ".ip", p + "conv.in_proj.weight"),
                    matrix(v + ".op", p + "conv.out_proj.weight"),
                    vector(v + ".kernel", p + "conv.conv.weight"),
                ]
            )
        else:
            for target, suffix in [
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("op", "o_proj"),
                ("gate", "gate_proj"),
            ]:
                lines.append(
                    matrix(v + "." + target, p + "self_attn." + suffix + ".weight")
                )
            lines.extend(
                [
                    vector(v + ".qNorm", p + "self_attn.q_norm.weight"),
                    vector(v + ".kNorm", p + "self_attn.k_norm.weight"),
                ]
            )
    lines.extend(["        return !error;", "    }", "}"])
    (ROOT / "src/model.zs").write_text("\n".join(lines) + "\n")
    print("Layout:", len(names), "tensors")


if __name__ == "__main__":
    main()
