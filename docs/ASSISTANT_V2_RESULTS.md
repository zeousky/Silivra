# Assistant v2 results — 2026-10-22

The existing assistant v2 run resumed from checkpoint 500 and completed all
8,000 updates with the original configuration and prepared data. **Instruction
loss and JSON formatting improved; independent task accuracy did not.** The
checkpoint remains unsuitable as a reliable general or coding assistant. The
interactive CLI, direct tools and explicit memory operations work independently
of that learned capability.

## Model measurements

All three models used the unchanged original 4,096-token tokenizer. Generation
was greedy, with 64 output tokens reserved and no tools executed. The independent
15-case suite covers instruction following, Python, supplied-source grounding,
tool selection/observations and supplied-memory recall. Its longest prompt plus
output reservation is 506 tokens, fitting the original 512-token model as well
as both 1,024-token instruction checkpoints.

| Measure | Original base, step 2,500 | Before resume, SFT step 500 | Completed/selected SFT step 8,000 |
| --- | ---: | ---: | ---: |
| V2 assistant-only validation loss, lower is better | Not comparable at 512 context | 3.73719 | 2.98620 |
| Independent valid JSON actions | 0 / 15 | 11 / 15 | 12 / 15 |
| Independent exact correct actions/answers | 0 / 15 | 0 / 15 | 0 / 15 |
| Historical diagnostic valid JSON actions | 0 / 6 | 4 / 6 | 5 / 6 |
| Historical diagnostic exact correct actions/answers | 0 / 6 | 0 / 6 | 0 / 6 |
| Synthetic validation valid JSON actions | 0 / 48 | 41 / 48 | 48 / 48 |
| Synthetic validation exact correct actions/answers | 0 / 48 | 3 / 48 | 5 / 48 |
| Original story validation loss, lower is better | 2.38112 | 2.96005 | 3.06644 |

The resumed run reduced instruction loss by 0.75098 (20.1%). It worsened story
loss by 0.10638 compared with checkpoint 500, and by 0.68532 compared with the
original language model. Instruction loss masks prompts/padding and is not
numerically comparable to unmasked story loss.

Each independent category has only three cases:

| Category | Valid actions at step 500 | Valid actions at step 8,000 | Exact successes at step 8,000 |
| --- | ---: | ---: | ---: |
| Instruction following | 3 / 3 | 2 / 3 | 0 / 3 |
| Python coding | 2 / 3 | 2 / 3 | 0 / 3 |
| Supplied-source factual grounding | 2 / 3 | 3 / 3 | 0 / 3 |
| Tool selection and observations | 3 / 3 | 3 / 3 | 0 / 3 |
| Supplied-memory recall | 1 / 3 | 2 / 3 | 0 / 3 |

Formatting is not task success. The five synthetic successes were two uncertainty
responses, two counting cases and one comparison. No sampled coding, grounded
fact, memory or tool-argument case passed exactly. The 48 cases are a fixed,
seeded sample of two examples from each of 24 original synthetic validation
families, with shared generator templates. This small sample does not establish
broad generalization. Six base-model synthetic cases exceeded its context budget;
those count as failed admissions, so native-context coverage differs in that
48-case comparison. The shared independent 15 cases all fit every model.
Exact grading can reject semantically equivalent wording;
these fixtures generally request exact text, code or action arguments.

Grounding cases test supplied source extraction, not broad factual knowledge.
Memory cases test model interpretation of quoted supplied memories, not learned
cross-session storage or retrieval. The runtime's storage, correction, isolation
and retrieval tests are separate. Proposed tools and generated code were never
executed by the evaluator; these are not end-to-end coding or research scores.

## Training and provenance

The model retains the original 14,162,304-parameter from-scratch lineage and
original tokenizer. The base at pretraining step 2,500 and interrupted SFT
checkpoint 500 were checksum-verified before and after the run and preserved.
No trainer was running before resumption, and the exclusive run lock was checked.
A historical unsaved step-600 log entry remains in place; updates resumed from
the complete step-500 optimizer/sampler/RNG checkpoint.

The unchanged `configs/assistant-v2.json` specifies 8,000 steps, batch four,
1,024-token context, 64-token padding buckets, learning rate 0.0002, 200 warmup
steps and cosine decay to a 0.1 ratio. The separate run completed with 2,494,990
supervised assistant/EOS token exposures and 11,680,945 nonpadding input-label
exposures. The resumed loop took 2,064.97 seconds; cumulative recorded training
time was 2,152.75 seconds. Peak MLX allocation was 3.38856 GiB, below the 6 GiB
run budget. This is not total system memory usage.

The existing data manifest contains 24,908 training and 1,311 validation
conversations (26,314 and 1,390 assistant-turn samples). Sources are pinned
Dolly-15k, original Codex-authored MIT exercises and original TinyStories training
story replay. See [attribution and licensing](../THIRD_PARTY_NOTICES.md).
Story-replay validation has previous pretraining exposure, so the independent
original validation token stream is used for language regression. The tokenizer
was not refitted. Training and validation artifact hashes and exact normalized
split separation passed verification. The 21 independent/historical diagnostic
prompts have zero exact normalized training-prefix overlaps; semantic/template
and corpus-wide pretraining contamination have not been exhaustively excluded.

The historical corpus manifest's generator hash differs from the recovered
unfinished generator source. Existing prepared artifacts and their hashes were
preserved. The current generator must not be claimed to reproduce those old
bytes exactly. Resume remains reproducible from the saved local artifacts and
configuration, subject to floating-point rounding differences. The public source
repository excludes weights and datasets, so cloning it alone cannot resume this
local experiment.

The trainer started with unfinished local source changes that were subsequently
reviewed and committed. Recorded source-file hashes identify the actual trained
code; the source snapshot commit is not a claim about when training started.

Step 8,000 is both `latest.json` and `best.json`, selected solely by the lowest
fixed 24-batch instruction validation loss, without consulting task scores. Full
1,024-context v2 loss is measured for both instruction checkpoints. It is omitted
for the 512-context base rather than silently extrapolating its context. Story
loss uses the same original 12 batches, batch 16, sequence length 256, seed 12345
for all three models.

## CLI verification and use

187 local tests passed, including MLX numerics, warmup/decay resume equivalence,
exclusive trainer locking, strict actions, budgets, streaming/cancellation,
context preservation, private state, data bounds, tool permissions and failures.
The [Linux portable CI run](https://github.com/zeousky/Silivra/actions/runs/39837975331)
passed, including real prompt-toolkit startup, multiline input and Ctrl+C tests.
Wheel/source builds passed and were checked to exclude private artifacts.
Network checks use offline fixtures; no new live network/provider test was run.

A real PTY demonstrated startup, help, explicit file reading, scoped memory
remember/recall and exit. A native step-8,000 turn loaded the checkpoint, streamed
a draft and accepted the complete valid JSON action, but its greeting was
nonsensical. This verifies the UI/model integration, not conversational quality.
The terminal keeps writes, network and command execution disabled by default.

From the checkout containing the local weights:

```sh
uv sync --locked
uv run silivra --model runs/silivra-assistant-v2
```

Use `/help` for controls, `/new` for a fresh conversation, `/tools read_file` to
select one model tool, and `/tool read_file {"path":"pyproject.toml"}` for an
explicit read that does not rely on learned tool selection. `/memory remember
language This project uses Python.` stores an explicitly directed fact; `/exit`
quits. The [terminal guide](CLI.md) covers sessions, settings, permissions,
multiline input and memory operations.

## Reproduce the measurement

With the same ignored artifacts present and the trainer stopped:

```sh
uv run python scripts/measure-assistant-v2.py \
  --output runs/silivra-assistant-v2/comparison-repeat.json \
  --public-output runs/silivra-assistant-v2/comparison-repeat-public.json
```

Output paths must be absent. The script verifies manifests, data artifact hashes,
lineage and exact prompt overlap, loads one model at a time, and evaluates the
same bounded fixtures/settings. It never runs proposed tools, trains weights,
downloads data or sends prompts to a provider.

[Machine-readable training evidence](results/assistant-v2-training.json) retains
configuration, hashes, validation history, resources and checks. The
[comparison record](results/assistant-v2-comparison.json) contains per-category
scores, full bounded diagnostic generations and the MIT synthetic validation
fixtures; it excludes imported Dolly rows, private chat logs and machine paths.
