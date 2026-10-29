# doollm

LFM3-1.7B running inside Doom. The matrix multiplies, attention, tokenizer and
text generation are all ZScript, running in GZDoom. Python turns the model
weights into lumps beforehand. There is no inference server.

This started with an int8 matrix multiply, then attention, then Qwen2.5-0.5B.
LFM3 is the current version. You can ask it something from the console and keep
playing while it works through the answer.

It is slow. One short prompt with eight output tokens took about 142 seconds,
including loading. That run matched the quantized reference token for token.
It does not mean the answers are good. It has confidently told me nonsense
about the BFG 9000.

## Run it

You need GZDoom 4.14+, Python 3.11 or newer, and your own Doom IWAD. The launcher
uses Doom II by default. Model weights and built PK3s are not in this repo.

From the repository root:

```sh
python3 -m venv bench/v04_lfm3/.venv
. bench/v04_lfm3/.venv/bin/activate
pip install numpy torch safetensors tokenizers psutil

python bench/v04_lfm3/tools/fetch.py
python bench/v04_lfm3/tools/export.py
python bench/v04_lfm3/tools/tokenizer_export.py
python bench/v04_lfm3/tools/layout.py
python bench/v04_lfm3/tools/package.py
python bench/v04_lfm3/tools/play.py
```

The download is several GB. Fetching uses the model revision recorded in the
source; the downloaded files and their hashes are listed in `assets.lock.json`.
For an IWAD somewhere else, pass `--iwad /path/to/DOOM2.WAD` to `play.py`.

Open the console with `~`:

```text
llm_prompt "What is the BFG 9000?"
llm
```

Tokens print as they arrive. The weights stay loaded between questions, but
each question starts fresh. There is no chat history.

```text
llm_thinking false
llm_cancel
```

The first command turns thinking off for the next question. The second stops
the current request and prints what it has produced so far. Thinking is on by
default. You can also change the instructions with `llm_system "..."`.

There is no input or output token cap imposed by the mod, and no default
request timeout. Generation stops at the model's end token or when you cancel.
Long prompts still cost memory and time.

## What's in here

- `bench/v01_int8_matvec`: the first matrix multiply.
- `bench/v02_attention`: attention and a Python reference.
- [bench/v03_qwen](bench/v03_qwen/README.md): Qwen2.5-0.5B, including its older console commands.
- [bench/v04_lfm3](bench/v04_lfm3/README.md): LFM3, measurements and profiling instructions.

The LFM3 runtime uses int8 weights and double-precision activations and caches.
Weights are stored in lumps and loaded into the VM. Work is split across small
slices so inference can run alongside the game; some initial reads and
allocations can still hitch.

In the single-token profile, dense matrix-vector products took 99.47% of the
measured stage time. GZDoom's VM JIT was enabled. That number includes indexing,
conversions and memory access as well as the arithmetic; it doesn't isolate
interpreter overhead.

[Liquid AI's LFM3](https://huggingface.co/LiquidAI/LFM3-1.7B) made this a fun
thing to try. Its model license is included in the generated PK3. Doom's game
data is not distributed here.
