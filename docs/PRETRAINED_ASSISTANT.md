# Local pretrained 8B assistant

Silivra provides a separate practical inference backend using **Qwen3-8B** from
the Qwen team, converted to 4-bit MLX format by MLX Community. Its architecture has
**8,190,735,360 parameters**. Quantization reduces storage and arithmetic precision;
it does not turn the model into a 14M model or enlarge the original Silivra weights.
The pretrained backend and the original research model have distinct provenance.

## Run

```sh
uv sync --locked
uv run silivra setup-assistant
uv run silivra
```

Setup downloads 4,623,794,744 bytes across weights, tokenizer, metadata and license,
verifies every file, and selects the model for the current local user/project.
It requires free download space plus 512 MiB headroom, refuses unrelated run
artifacts, limits each invocation to a 15-minute transfer session, and preserves
partial transfers for a repeated invocation. It never changes original checkpoints.
The marker is written only after all pinned artifacts verify. A size/hash mismatch
is reported; a corrupt `.partial` must be inspected or removed before retrying.

Use an interactive session for the lowest ongoing latency: the first ordinary
message loads and verifies the weights, and later messages reuse that loaded model.
Each one-shot `chat` starts a new process and pays loading/verification again.
The backend reuses one exact input-token prefix in memory, capped at 1,024 tokens
and 256 MiB of allocated KV storage. It never retains a generated tail as input,
always recomputes at least one token, and drops reusable state on cancellation,
invalid output or an exhausted generation budget. Token accounting still uses
the complete prompt. Nothing is persisted and no model server is started.

Setup selects `calculate` if the saved tool selection is empty. Pretrained one-shot
chat also defaults to this tool; `--no-tools` disables it. Interactive `/tools`
preferences remain saved. For file-backed calculations select both `read_file` and
`calculate`; explicit tool lists replace the one-shot default.

```sh
uv run silivra chat "Explain the difference between a list and a tuple in Python."
uv run silivra chat --tool read_file "Read pyproject.toml and tell me the package version."
# Original-model research remains available:
uv run silivra --model runs/silivra-assistant-v2
```

## Identity and operating limits

- External model: `mlx-community/Qwen3-8B-4bit`.
- Immutable revision: `545dc4251c05440727734bcd94334791f6ab0192`.
- Original model/tokenizer: `Qwen/Qwen3-8B`; Apache-2.0 license.
- Original license revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Inference: Apple MLX/Metal through pinned `mlx-lm==0.32.2`.
- Runtime context: 8,192 tokens, including policy, schemas, history, memory,
  current-turn evidence and output reservation. Qwen's upstream configured maximum
  is larger; this runtime deliberately uses a smaller memory budget.
- Thinking: disabled through Qwen's actual chat template. Output defaults to 768
  tokens, configurable up to 4,096 subject to the context budget.
- Model allocation guideline: 8 GiB, with a 256 MiB allocator cache guideline.
  MLX's limit is advisory; OS/apps and transient allocations also consume memory.
  MLX-LM temporarily changes its process wired allocation limit during generation;
  setup changes no system settings or sysctls.
- Inference needs neither an API key nor network. Missing/tampered model files fail
  visibly. Local tokenizer loading disables downloads and remote Python code.
- An Outlines-core automaton masks tokens to one JSON action with string finals,
  enabled tool names and typed argument objects. Semantic limits and permissions
  remain runtime checks. Constrained syntax is not proof of answer correctness.
- A labelled synthetic calculator exchange demonstrates the action/observation
  protocol when that tool is enabled. These prompt tokens count toward context;
  they are neither user facts nor weight updates and are never stored as memory.
- Tool writes, network and trusted command execution remain disabled by default.
  Read tools must be selected explicitly for model use. Context overflow does not
  silently discard policy, current requests or current-turn tool evidence.
- Durable memory still requires explicit user operations. No chats, tool observations
  or private memories are automatically used for training.

The arithmetic tool interprets at most 512 characters, 64 AST nodes, 16 nesting
levels and 64 arithmetic operations. It accepts decimal literals, parentheses and
`+ - * / // %` with unary signs, using exact fractions. Numeric magnitudes are
bounded at `1e100`, rational storage at 4,096 bits, and output is decimal text at
50 significant digits with rounding identified. It cannot execute Python, access
files or make network requests. Unaided model arithmetic remains a separate skill.

See [third-party notices](../THIRD_PARTY_NOTICES.md) and the local manifest for
license, hashes and sizes. The downloaded weights/tokenizer remain ignored artifacts.

## What changed in learning

No weight updates were performed for this integration. Qwen's existing pretrained
capabilities provide the larger model. Original Silivra pretraining, tokenizer,
instruction tuning, v2/v3/v4 checkpoints and measured failures remain intact.
Full FP32 weights alone for this architecture would occupy about 32.8 GB before
gradients or optimizer state. Original 8B pretraining needs a separately planned
compute/data program; it is not established by connecting tools or downloading
external weights. No hosted model, paid compute, or paid generation is involved.

## Local validation

Measured on the M4 16 GB target on 2026-10-24: the 8B backend passed
**15/18** development tasks, **8/12** separate diagnostic tasks, and **4/4**
calculator-assisted tasks. Median decode throughput was approximately 24 tokens/s;
median first-token latency was 0.62–0.97 seconds across the diagnostic runs with
prefix reuse. These timings exclude model loading and hash verification.
See the full [development](results/pretrained-8b.json),
[separate diagnostic](results/pretrained-8b-fresh.json),
[calculator](results/pretrained-8b-calculator.json), and
[live cache/cancellation/output-budget](results/pretrained-8b-runtime.json) reports.
All 18 runtime checks passed. Original unaided arithmetic and character-case
mistakes remain visible; a file-listing task also failed in the separate set.
The first separate-set report rejected valid `+=` syntax; the bounded checker was
extended to support local augmented assignment, without changing frozen prompts.
Repeated diagnostics informed adapter changes and are regression evidence, not
untouched general-intelligence benchmarks.

The source currently passes 308 Mac tests. The preserved
[249-test Linux report](results/pretrained-linux.json) covers its recorded source
hashes before the subsequent 9B profile extension; it is not verification of that
extension on Linux. No paid inference or training was used.

An optional **Qwen3.5-9B text-only profile** has pinned download/architecture support
and portable tests, but its local download was interrupted at the user's request
to save and publish the current state. **No native 9B inference, quality, memory or
speed result has been established, and it was not promoted.** The default remains
the tested Qwen3-8B backend. A future explicit trial can resume with
`uv run silivra setup-assistant --profile qwen3.5-9b --no-select`.
Its approximately 5.98 GB artifacts are separate from both existing model lineages.

The frozen [pretrained-v1 diagnostic](../examples/evaluations/pretrained-v1.md)
checks instruction following, mathematics, stable facts, two small Python functions,
real file tools and observation interpretation, untrusted file content, missing-file
grounding, conversation recall, and explicit memory recall/correction/isolation/deletion.
Python functions are checked by a bounded AST interpreter for the requested subset;
this is not a general execution sandbox or a broad coding benchmark.

Reproduce with a fresh output path:

```sh
uv run python scripts/measure-pretrained.py \
  --run runs/silivra-qwen3-8b --output docs/results/pretrained-8b.json
```

The report records raw public synthetic outputs, actual tool observations, objective
grades, source hashes, cold model verification/loading, per-generation latency,
decode throughput and memory. Tests mock inference/network separately. These
eighteen tasks are a small diagnostic; they cannot establish general intelligence,
reliability on arbitrary software projects, long-context quality, or absence of
upstream pretraining contamination. Historical original-model scores used other
suites and must not be presented as a direct accuracy comparison.
