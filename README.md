# doollm

The native **LFM3-1.7B** experiment is in [bench/v04_lfm3](bench/v04_lfm3/README.md).
It supports custom console prompts and cancellable inference. See its measured
latency and limitations before testing; the instructions below describe Qwen.

Qwen2.5-0.5B-Instruct inference inside GZDoom's ZScript VM. Python prepares
quantized weights and provides a numerical reference; inference and console
prompt tokenization run inside the engine.

The experiments progress from `bench/v01_int8_matvec` (integer matrix-vector
multiplication), through `bench/v02_attention` (attention), to `bench/v03_qwen`
(the full model).

## Ask a question in Doom

With the v03 PK3 loaded and a map running, open the console with `~`:

```text
qwen_prompt "What is the BFG 9000?"
qwen
```

The reply appears after `Qwen:`. Change `qwen_prompt` and run `qwen` again for
another question. Weights stay loaded, while generation state starts fresh;
questions are independent, with no conversation history.

Optional settings:

```text
qwen_system "You are a helpful assistant. Answer in one short sentence."
qwen_max_tokens 24
```

The context budget is **128 tokens**, including the chat template and generated
reply. With the default 24-token generation budget, prompts may contain at most
105 tokens including the template. Oversized prompts produce a console error.
Generation is synchronous and pauses gameplay until the reply is ready.

To use a previously prepared prompt slot:

```text
qwen_prompt ""
qwen_idx 0
qwen
```

`qwen_tokenize` prints prompt token IDs without running inference.

## Setup and launch

Requires GZDoom 4.14+, a Doom-compatible IWAD, Python, and:

```sh
python3 -m pip install -r bench/v03_qwen/requirements.txt
```

Place the Qwen2.5-0.5B-Instruct model's `model.safetensors`, `config.json`, and
`tokenizer.json` in `bench/v03_qwen/tools/`. These large/downloaded assets and
generated data are git-ignored. The implementation is specific to this model's
architecture; it uses greedy decoding and int8 weights, activations, and KV cache.

```sh
python3 bench/v03_qwen/tools/ask.py "Give me a Doom survival tip."
```

This prepares the tokenizer, generates weights if absent, packages
`bench/v03_qwen/build/qwen_v03.pk3`, and launches the game. Use `--iwad PATH`
for a different IWAD or `--no-launch` to build only. The default IWAD is
`/usr/share/games/doom/DOOM2.WAD`.

To launch an already built PK3 without automatically asking a question:

```sh
gzdoom -iwad /usr/share/games/doom/DOOM2.WAD \
  -file bench/v03_qwen/build/qwen_v03.pk3 +map MAP01
```

`prep_prompt.py` preserves existing saved prompts. `set_prompt.py QUERY --idx N`
updates a slot, `add_prompt.py QUERY` appends, and `add_prompt.py --clear` removes
both the saved library and generated prompt binaries.

## Benchmark

```sh
python3 bench/v03_qwen/tools/run.py --skip-weights --query "Say hello." --max-tokens 3
```

The benchmark needs a working graphical session and closes its own GZDoom
process. It compares prompt IDs, generated IDs, raw text bytes,
first logits, and projection checksums against NumPy, then checks that gameplay
resumes. Remove `--skip-weights` to regenerate weights; omit `--max-tokens` for
the default generation budget. Logs and reference output go to the v03 `build`
directory.

The native tokenizer uses exported BPE ranks, the tokenizer library's Unicode
character classes, and NFC tables generated with Python's Unicode database
(version recorded in `src/data/unicode_version.txt`).
