# Local 8B and 9B comparison

On 2026-10-25, the pinned Qwen3.5-9B download completed and verified. Its
text-only MLX backend has 8,953,803,264 parameters; the existing Qwen3-8B backend
has 8,190,735,360. Both are external pretrained models. No weights were trained
or changed, and the original Silivra checkpoints/tokenizer remain separate.

## First comparison on new tasks

The [15-case v3 fixture](../examples/evaluations/pretrained-v3.md) was frozen
before either model ran. Both used identical adapter source hashes, temperature
zero, 384 output tokens per action, five actions per turn, the same read-only
tools and scoped-memory fixtures, and the same M4/16 GB Mac. The fixture was
not used as training data. Expected answers and task-specific examples were not
inserted into system/adapter prompts or decoder rules.

| Initial measurement | Qwen3-8B | Qwen3.5-9B |
| --- | ---: | ---: |
| Tasks passed | 11/15 | 14/15 |
| Coding | 2/2 | 2/2 |
| Tool-dependent tasks | 3/5 | 5/5 |
| Conversation and memory | 3/3 | 3/3 |
| Median first-token latency | 0.91 s | 2.99 s |
| Median decode throughput | 21.73 tokens/s | 20.58 tokens/s |
| Median complete case time | 2.91 s | 5.99 s |
| All 15 cases, excluding loading | 49.84 s | 117.74 s |
| Model loading and verification | 5.17 s | 7.54 s |
| Peak MLX allocation | 4.824 GiB | 5.255 GiB |

Raw [8B](results/pretrained-8b-v3.json) and
[initial 9B](results/pretrained-9b-v3.json) reports retain all outputs,
observations, grades, timings, model provenance and source hashes. These are
single local runs, not controlled thermal or operating-system benchmarks.
macOS swap use rose during the 9B trial; MLX allocation is not total system
memory use. Timing excludes downloads, and the summary latencies exclude loading.
Case times include actual tools and may include several actions or conversation
turns; the models need not take the same number of actions.

8B incorrectly replaced uppercase/substrings, returned an otherwise correct
dependency order as invalid inner JSON, failed to recover from an empty literal
search, and obeyed an answer-hijack instruction inside an untrusted file. Its
ordering failure was formatting, rather than incorrect dependency reasoning.

9B's only failure was an extra terminal period in an exact-format answer. It
recovered from its initially empty search, read the relevant file, and answered
the one adversarial-file task correctly. Both invoice passes were manually
audited: each model actually read both source files and used the real calculator.
Passing this one adversarial task does not establish general injection resistance.

The initial 8B run reused bounded input-prefix KV state. The initial hybrid 9B
run deliberately prefills every prompt because its recurrent layers cannot be
trimmed back from a generated tail. Its stronger task results therefore came
with substantially higher first-token and tool-loop latency. The reports preserve
that performance cost rather than treating parameter count as an improvement.

## Final default and cache validation

Qwen3.5-9B is the new setup/discovery default after the live checks below. Existing
saved model choices remain intact until explicit setup or `/model` changes them;
Qwen3-8B remains available with `setup-assistant --profile qwen3-8b`.

The hybrid cache now copies only an evaluated stable input prefix, capped at
1,024 tokens and 256 MiB. Its entire token prefix must match before reuse;
recurrent state is never trimmed and generated answers never become input cache
entries. Prefill chunks remain at most 256 tokens. Failed/cancelled generations
discard pending state. See the [architecture](ARCHITECTURE.md) for the boundaries.

The [cached v3 repeat](results/pretrained-9b-v3-cached.json) retained **14/15**.
Median first-token latency fell from 2.99 to **1.89 seconds**, total suite time
from 117.74 to **82.83 seconds**, and median case time from 5.99 to **3.94 seconds**.
Median decoding was **21.04 tokens/s**, model loading/verification took 5.62 s,
and peak MLX allocation was **5.311 GiB**. This is about a 30% reduction in that
suite's wall time across two individual runs, not a controlled general benchmark.
9B still has higher first-token and tool-loop latency than 8B in these local
measurements; its default selection prioritizes the observed task-quality gain.

Repeated development diagnostics retain their original prompts and all failures:

| Diagnostic | Earlier 8B | Final 9B |
| --- | ---: | ---: |
| v1 | 15/18 | [16/18](results/pretrained-9b.json) |
| v2 | 8/12 | [9/12](results/pretrained-9b-v2.json) |
| Calculator-assisted | 4/4 | [4/4](results/pretrained-9b-calculator.json) |

9B still makes unaided arithmetic mistakes, including interpretation of numeric
file contents when only a read tool is available. The calculator-assisted suite
passes all four tasks. One-shot pretrained chat and an empty-selection setup
therefore expose `calculate` by default. Explicit tool selections replace the
default, so file-backed calculations should select both `read_file` and `calculate`.
The older 8B measurements are from 2026-10-24; use the paired initial v3 runs for
the same-source comparison rather than treating this table as a controlled ranking.

All **20 live runtime checks** pass for
[9B](results/pretrained-9b-runtime.json) and
[8B](results/pretrained-8b-runtime-20261025.json). They check cached/cold equality
for text, code and calculator actions, changed-question equality, actual expected
cache reuse, streaming, cancellation/recovery and output-budget rejection. The
probe distinguishes native trimmability from hybrid snapshot support. These checks
validate those runtime paths; they do not establish correct answers on arbitrary
tasks. The final source passes **328 Mac tests** and
[281 actual Linux tests](results/pretrained-linux-20261025.json). The interactive
CLI identified the selected external model, streamed a correct Python explanation,
and used the actual default calculator to return 139 for `19.5 * 8 - 17`.
Model and Docker processes were closed after verification; no server is installed.

## Evidence boundaries

Fifteen synthetic tasks are limited evidence, not a standardized benchmark or
proof of general intelligence. Exact checks can fail semantically correct answers
for punctuation; the small Python checker supports only the requested AST subset.
The earlier v1/v2/calculator suites informed adapter changes and are regression
checks. Repeating v3 after implementation changes is also regression evidence.
Upstream pretrained-data overlap cannot be comprehensively audited.

Any later performance result must retain the initial reports, identify its source
snapshot, and verify action/tool behavior, cancellation, output limits and memory.
Cache changes must never turn a generated answer into trusted prompt input.
