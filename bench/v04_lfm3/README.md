# Native LFM3 experiment

LFM3-1.7B runs inside GZDoom 4.14's ZScript VM with per-row int8 weights,
double activations/caches, greedy decoding, and a native tokenizer. Python
prepares assets and validates results; it does not serve the game requests.

## Launch and prompt

With the generated `build/lfm3.pk3` present:

```sh
python3 bench/v04_lfm3/tools/play.py
```

Open the game console with `~`, then enter:

```text
llm_prompt "What is Doom (1993)?"
llm
```

`llm_cancel` stops the current request. Set `llm_system` to customize its
instructions. Requests are independent; weights stay loaded for subsequent
requests. `llm_tokenize` prints prompt IDs without inference.

The launcher uses its own `build/play.ini`, enables background rendering,
disables background frame throttling and VSync, and caps rendering at 120 FPS. These
settings matter because inference runs in bounded slices on UI ticks and
rendered frames. Suppressing or throttling rendering slows inference.

There is **no mod-imposed input or output token cap**, and no request timeout
by default. `llm_max_tokens 0` and `llm_timeout_s 0` mean unlimited; positive
values are optional limits for benchmarks. Generation continues until the
model emits its end-of-response token or you enter `llm_cancel`.

Thinking is enabled by default (`llm_thinking true`): the prompt leaves
`<think>` open, and model-generated thinking, its `</think>` marker, and answer
text stream to the console as tokens arrive. GZDoom's ZScript console API adds
a newline per print, so streamed pieces appear on separate lines. Incomplete
UTF-8 characters wait for their remaining bytes. The final print includes the
complete generated text. `llm_thinking false` restores the non-thinking template.

Full attention caches grow in 256-position pages; sliding attention retains
its native 256-position window. Positional encoding is computed on demand
beyond the precomputed prefix. The model configuration declares 262,144
positions; removing this mod's checks does not make model context, engine
strings, or available RAM infinite. No prompt tokens are silently truncated.
Long full-attention contexts require more memory and compute.

The scheduler still uses 4 ms work slices (maximum 8 ms), including incremental
attention reductions, to preserve responsiveness. Progress includes an
approximate time to the first token. Longer reasoning can take substantial time.

## Measured status

This remains a slow experiment. Before the uncapped/thinking change, the scheduler completed
the 34-token prompt `What is planet Mars's colour?` plus eight output tokens
in **142.18 seconds**, including startup and loading. Its truncated output was
`Mars has a reddish appearance due to iron`. All eight IDs matched the
independent W8 reference; selected layer errors were below 1e-12.

In that 640x480 MAP01 test without monsters, 13,077 inference render intervals
averaged 10.13 ms and peaked at 14.34 ms. Loading included a 119 ms slice:
initial large lump reads/allocations are not yet fully incremental. These
measurements do not establish performance on every map, resolution, or prompt.
The previous UI-only scheduler could take many minutes and is superseded.

Evidence is in `build/mars_background.metrics.json`, `.log`, and `.parity.json`.
After the cache changes, an eight-token thinking-enabled `Hi.` check matched
the independent reference in every generated token and selected intermediate
vector (maximum error below 1.4e-13). It took 172.45 seconds in that background
test; streaming does not eliminate inference latency. Evidence is in
`build/thinking_stream_vsync.log` and `.parity.json`. Test runs deliberately use
finite output budgets; the interactive launcher defaults to no cutoff.

Tokenizer validation covered 114 cases; isolated original-model block checks
covered convolution boundaries and the 256-token sliding-window boundary.
Native cache checks now cover positions 255, 256, 511, 512, and 1025 against
independent NumPy attention. Positional math is checked through position 262143,
and a 10,000-byte / 2,027-token prompt matches the reference tokenizer. These
are component checks; broad answer-quality evaluation and a native full-model
long-context generation test are still unfinished. Model answers can be wrong.

## Rebuild and profile

The local `.venv` contains the tooling dependencies. The pinned source revision
and asset hashes are in `assets.lock.json`. Package after changing ZScript:

```sh
bench/v04_lfm3/.venv/bin/python bench/v04_lfm3/tools/package.py
bench/v04_lfm3/.venv/bin/python bench/v04_lfm3/tools/profile_engine.py
```

The profiler measures one raw BOS token through all layers and the vocabulary
projection, with a cold load. It writes `build/forward_profile.json` and closes
only its own GZDoom process. It requires a graphical session. This is a short
context measurement, not a conversational throughput benchmark.

For a normal request, enable `llm_profile true` before `llm`. The report contains
per-stage elapsed timers and the active VM JIT setting. Profiling is off by
default and adds timing overhead. The measured single-forward run spent 99.47%
of its stage time in dense matrix-vector products, with JIT enabled; this does
not separate memory traffic, indexing, conversions, and arithmetic costs.

The packaged model includes its source license.
