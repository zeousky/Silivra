# Assistant foundation results — 2026-10-22

The assistant runtime, tool and memory system, instruction data pipeline and
bounded MLX fine-tuning are implemented. **The current model is not a useful
general assistant.** Training improved JSON formatting but did not produce correct
held-out task responses, and it regressed original story-language loss. The
original pretrained checkpoint is preserved unchanged.

## Engineering verification

100 local tests pass on Apple M4/macOS/Python 3.12/MLX 0.32.4. They cover original
model numerics/pretraining, instruction masking and resume, data bounds and leakage,
strict actions, permission denial and budgets, file conflicts/path confinement,
network address/redirect limits, memory scopes/revisions/deletion/backup, CLI
operations, and curriculum correctness. Scripted backends test orchestration;
they are not evidence that the native model can perform the same workflows.

Live public Wikipedia search and HTTPS article fetching succeeded. Brave search
was tested with fixtures; no live Brave/paid-provider request was made. Command
execution is explicitly opt-in trusted host execution, without OS sandboxing.

## Native model experiment

The base is the original 14,162,304-parameter TinyStories model at step 2,500,
with its original 4,096-token tokenizer and 512-token maximum context.
`scripts/build-curriculum.py`, seed 17, generated 400 original AI-assisted MIT
training records and 100 validation records. Ten families cover literal copying,
exact answers, arithmetic, Python expressions/functions and single-tool proposals.
Validation reserves separate words, filenames and arithmetic operands; templates
remain shared. It is a narrow validation set, not an independent general benchmark.

The separate instruction run completed 1,000 AdamW updates using
`configs/finetune-curriculum.json`: batch 4, learning rate 0.0002, 25 sampled
validation batches at each 100-step checkpoint. It used 86,406 supervised
assistant/EOS token exposures and 1,056,962 nonpadding input-label exposures.
The training loop took 146.99 seconds; peak MLX allocation was 1.5434 GiB.
That allocation is not total system memory usage.

The lowest sampled validation loss selected **step 400**, independently of exact
task scores. Validation batches use a fixed seed and may repeat records. Initial
assistant-only validation loss was 8.34198; selected loss was 0.80323. The final
step-1,000 loss was 0.94593. These values are not comparable to unmasked story loss.

| Measure | Original base | Selected instruction checkpoint |
| --- | ---: | ---: |
| Valid JSON actions, curriculum validation | 0 / 100 | 100 / 100 |
| Exact correct actions/answers, curriculum validation | 0 / 100 | 0 / 100 |
| Valid JSON actions, fixed assistant diagnostic | 0 / 6 | 5 / 6 |
| Exact correct actions/answers, fixed assistant diagnostic | 0 / 6 | 0 / 6 |
| Original story loss, fixed 12-batch sample | 2.38112 | 6.15055 |

All 30 curriculum tool proposals named the single advertised tool correctly, but
all supplied incorrect arguments. For example, a request to read `owl.py` proposed
`cup.py`; copying `map map` returned `bed`. Such outputs must not be mistaken for
successful coding or research. The evaluation never executes model proposals.
The fixed six diagnostics use separate prompts; underlying skill/problem overlap
with curriculum examples has not been exhaustively excluded.

The initial seven-conversation format fixture was also trained for a short pipeline
check in a separate local run. It similarly learned formatting without solving the
six diagnostics. It is not a model release or evidence of general improvement.

## Reproducibility and limits

The public [machine-readable records](results/reproduction.json) identify source
hashes, data-manifest hash and checkpoint selection. Full raw diagnostic outputs
are retained in [curriculum results](results/curriculum-instruct.json) and
[assistant results](results/assistant-instruct.json), alongside the corresponding
base outputs. [Training records](results/training-run.json),
[summary](results/training-curriculum.json),
[data generation](results/curriculum-generation.json), and
[language regression](results/story-regression.json) document the experiment.
Local machine paths are removed from these public copies; private chats/memory,
raw datasets and weights are excluded from Git.

The selected instruction weight SHA-256 is
`57b2f702e59dfb9cdcb47df0dca5b9d44134b762081e819f3fd5df4f0a85e71c`.
The original base remains
`633efc5b513b7de90de6d1e55d37f45948852d1c19728078a54f7502a6a7cf40`.
Floating-point training reductions can vary between runs. Reproduction should
compare outcomes and provenance; it need not recreate an identical trained hash.

The next model milestone is broader licensed language/code pretraining and a
substantially more varied instruction curriculum with language-retention checks.
Longer-context training is necessary for useful multi-step code, observations,
memory and history. Keep the original checkpoint as the language baseline rather
than replacing it with this format-focused experiment. The shared runtime and
memory APIs can evolve while this model work proceeds.
