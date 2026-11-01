# Native English v6 experiment

**Status: data and final grading are frozen; development-led training experiments continue.**
The three initial token-weighted pilots all scored 0/96 on the current original
development sample. Their older diagnostic scores were 5/85, 4/85 and 3/85 for the
conservative, balanced and higher learning rates. Conservative and balanced passed
both retention limits; the higher-rate pilot failed the story-retention limit.
These results motivated an optional assistant-turn-weighted objective before
candidate fresh evaluation. This adaptation was based on development failures;
it was **not** a preregistered training-objective comparison.

At that protocol change, four training sessions had consumed 1,900.795 seconds
(31.68 minutes) of the same 7,200-second child-wall budget. The fourth session was
an intentional, gracefully interrupted conservative resume with a complete
step-3,556 checkpoint. It was not a failed or incomplete weight artifact. The
original token-loss compatibility check has passed; final candidate selection
and fresh evaluation remain pending.

This experiment tests whether a more varied original English curriculum and much
stronger rehearsal can improve Silivra's own model without repeating v5's severe
language and instruction regressions. It does not establish broad intelligence or
reliable assistant ability.

## Original model and data

The initial pilots start separately from Silivra assistant v2 step 8,000,
preserving its original 14,162,304-parameter architecture, 4,096-token tokenizer
and 1,024-token context. Later runs must identify their actual verified native
starting checkpoint and parent hashes. Original checkpoints remain immutable.
No alternative model, hosted inference, tokenizer fitting, new downloads, private
chats or saved user memories are used.

The new exercises are original AI-authored synthetic material under MIT. They
include varied conversational answers, multiple English-definition phrasings and
examples, grammar identification/edits, exact copying, supplied facts, uncertainty,
synthetic memory revisions, simple Python output, arithmetic, and tool arguments.
Ordinary tasks include zero-, one- and two-tool schema contexts. Tool proposals
copy expressions or queries from the request. Supplied calculator observations
use the real pure arithmetic implementation; mocked tests check calculator and
search wrappers against the actual tool adapters. Search observations are
fictional, use reserved `example.com` URLs, and are not evidence of live retrieval.

The rehearsal mixer selects existing local v2 records without changing any
retained record's messages, source, revision or license. Databricks Dolly records
retain CC BY-SA 3.0, original v2 synthetic exercises retain MIT, and TinyStories
training-split replay retains CDLA-Sharing-1.0. The MIT selection code does not
relicense imported data or assign a license to derived weights. See
[third-party notices](../THIRD_PARTY_NOTICES.md).

## Actual mixture and context limits

Selection uses deterministic hash ordering and proportional source/family quotas
measured in assistant decisions. A multi-turn conversation can contribute more
than one decision. Rehearsal is not simulated by repeating the same records.
Both splits must contain at least 70% rehearsal by assistant-decision count **and**
supervised label-token count; preparation fails if either measure is too low.

| Measure | Training | Development validation |
| --- | ---: | ---: |
| Conversations | 28,033 | 1,653 |
| Assistant decisions | 30,000 | 1,758 |
| Rehearsal decisions | 22,000 | 1,390 |
| New original decisions | 8,000 | 368 |
| Rehearsal share of decisions | 73.33% | 79.07% |
| Supervised label tokens | 1,911,372 | 114,865 |
| Rehearsal label tokens | 1,740,105 | 107,618 |
| New original label tokens | 171,267 | 7,247 |
| Rehearsal share of label tokens | 91.04% | 93.69% |
| Maximum complete sample, next-token positions | 1,024 | 1,007 |
| Prepared padded cells | 30,750,000 | 1,801,950 |

The existing preparation limit is 32 million padded cells per split. Including
all v2 records plus all new exercises would exceed that limit, so the training
mixer selects 22,000 rehearsal and 8,000 original decisions. The actual token
shares are reported because conversation counts alone would hide how strongly
longer rehearsal answers dominate supervision. Mixed validation loss can improve
while new English tasks remain poor, and the reverse can also happen.

## Separation and evaluation roles

The data builder accepts one JSON string containing a user message per exclusion
line. It rejects evaluation objects or answer fields. The frozen fresh artifact
contains 52 user-message strings; the existing development artifact contains 94.
A message count is not an evaluation-case count because a case can include more
than one message.

Every excluded message is removed from training and development validation,
including licensed rehearsal. Literal arithmetic operand/operator matches are
also excluded, treating addition and multiplication operand order equivalently.
Training excludes every individual user message in the combined development
validation split. Independent artifact review confirmed no excluded user-message
matches and disjoint train/validation individual user-message sets.

These checks do not establish absence of semantic or historical pretraining
contamination. New validation uses reserved words/entities and request wording,
but many skill templates and English concepts are shared with training. Imported
rehearsal validation measures retention, not fresh language generalization.

The measurements have distinct roles:

- **Within-run checkpoint selection (`best.json`)** minimizes validation loss
  under that run's configured objective. Token-normalized validation has 93.69%
  rehearsal labels; assistant-turn-normalized validation gives each nonempty
  supervised decision equal objective weight, with 79.07% rehearsal decisions in
  this split. These configured-loss scalars are not directly comparable across
  objectives and cannot rank token- versus turn-trained candidates by themselves.
- **External mixed-data token loss** is recomputed using the original token
  reduction and fixed evaluation protocol, regardless of the training objective.
  It is comparable across those candidates with the same prepared data and
  sampling settings. It remains a mixed-distribution loss rather than a score
  for English fluency or reasoning.
- **Current original development tasks** grade a fixed, balanced subset of the
  new validation decisions. Their shared templates make them a development signal
  for pilot selection, rather than an independent benchmark.
- **The existing 85 diagnostic tasks** track earlier instruction, conversation,
  grounding, coding, tools, memory and uncertainty behavior. They have informed
  previous development and cannot be relabelled as a new blind evaluation.
- **Prior v2 instruction loss and original story validation loss** measure
  retention on unchanged samples using the original token-normalized reductions.
  They remain comparable across training objectives and separate from current
  mixed loss. Remeasuring the unchanged original checkpoint after the optional
  objective implementation gave exactly the same token-loss baseline values;
  lower configured-objective loss still does not excuse degraded retention.
- **The frozen fresh suite** is reserved until one candidate is fixed. Its answer
  material is not consumed by the data builder or pilot selection. Final grading
  and retention gates must be applied together, with failures retained.

Valid JSON, correct task outcomes, proposed tool arguments and actual runtime
execution are separate claims. These model diagnostics do not execute proposed
tools. Exact/finitely declared answer matching can reject a differently worded
correct explanation, so raw diagnostic answers remain available for review.
Fresh acceptance criteria must not be loosened after seeing candidate answers.

## Bounded training method

Initial pilot configurations use batch four, a 6 GiB advisory MLX allocation
budget, 1,024-token context, 64-token padding buckets, 100 warmup updates, cosine
decay and an 8,000-update schedule. The initial token-weighted learning-rate
profiles are `1e-5`, `3e-5` and `1e-4`. Separate 24,000-update continuation
configurations also exist; configuration availability is not evidence that a run
was trained or evaluated.

### Development-led objective adaptation

The frozen mixture gives rehearsal 73.33% of training assistant decisions but
91.04% of supervised label tokens. Under the original token-normalized objective,
longer rehearsal answers therefore dominate the label population. All three
initial pilots failed the current 96-case development sample despite some older
loss or diagnostic gains. This observation motivated testing an explicit
`loss_normalization: "assistant_turn"` option with `3e-5` and `1e-4` pilot
configurations. The dataset, tokenizer, frozen user exclusions, development cases
and final grading criteria remain unchanged.

The default remains `token`: sum masked cross-entropy across labels, divided by
the total supervised-label count. The optional assistant-turn objective first
averages masked cross-entropy within each nonempty decision and then averages
those decision means. Prompt, observation and padding labels remain excluded.
This prevents answer length alone from giving a decision more total objective
weight. Its population mixing weights correspond to the decision shares
(73.33% rehearsal in training), rather than the label-token shares. These are
objective weights, not measured gradient magnitudes or evidence of better skills.

Changing normalization requires a separate run and declared objective. Such a
run starts from its identified native checkpoint with a fresh optimizer and
learning-rate schedule; it is not a continuation of the old optimizer history.
An exact `--resume` retains the original normalization, data, optimizer, sampler,
RNG and schedule and rejects a changed objective. New run/checkpoint metadata
records the normalization and trainer implementation hash. The initial and
adapted phases use distinct trainer revisions, identified in the provenance table
below; the original token formula remains the default in the adapted revision.

A recheck of the unchanged v2 checkpoint returned prior instruction loss
`2.9862034865382174` and story loss `3.066438396771749`, exactly matching the
pre-change measurements: maximum absolute difference was zero. The compatibility
record identifies the old and new trainer hashes, unchanged model weights and no
fresh-suite execution. All 278 local tests passed at this implementation check.
This verifies the fixed token-loss comparison and trainer behavior; it does not
make the training objectives equivalent or demonstrate improved model skills.

This is an adaptive development experiment. It does not change the fresh-suite
thresholds, permit another candidate after inspecting fresh results, or establish
that per-turn weighting will improve English or reasoning.

Pilots and any continuation share at most 7,200 seconds of supervised child
wall time. This accounting includes initialization, checkpoint/evaluation grace
and unsaved work, making it stricter than counting training-loop time alone.
A durable reservation is charged before starting a child; a supervisor crash
leaves the reservation charged. An exclusive supervisor lock rejects concurrent
supervised trainers. Numerical inference/evaluation is kept separate from active
training. The 6 GiB MLX guideline is monitored between steps and is not an OS
memory cap.

Pilot choice uses development outcomes and retention only. An exact resume of a
chosen pilot must preserve its run, tokenizer, prepared-data hashes, objective,
optimizer, sampler, RNG, total-step schedule and training configuration. A changed
objective or schedule uses a separately declared run with its own initialization
and optimizer lineage. Only a complete verified checkpoint is resumable. Logged
updates without a published checkpoint and saved state must not be counted as a
finished artifact. A checkpoint selected by mixed
loss is not automatically promoted to the general default.

## Frozen provenance

| Artifact | SHA-256 |
| --- | --- |
| Trainer before optional assistant-turn objective | `4e5f433e8045b458034061dcbb3e5f30c2c5410d389905a79fa4f5f48bb97d0e` |
| Trainer with optional assistant-turn objective | `e1a7c7c2e8f9be276b89cd136e6a7752f4cbc129230ab42262bf8b080c8bd24b` |
| V2 base weights, step 8,000 | `efb8d54eb4f0bda52567e50e41398e9d7793a7ae3c6e2d650bf3770321187a96` |
| Original tokenizer | `1640a5301686bd4970d6bd5c1b27926f71227bfd11b9e2d59375cbe33c035269` |
| V2 prepared input manifest | `5f87cbc17af6afd4cfd8f9258bf31aa03aea1a15cf28f678c37ed04c9477d51c` |
| V6 generator | `bd5d67c4dea4e4688a344d4ff804bb5993877f1c3595a2ec422b612eadc70151` |
| V6 selection/corpus manifest | `8b4820bd3d67607e605b5466709fd33eaacaf45ea6064b64f169cd5530269f9a` |
| V6 prepared manifest | `cc4d91db514750b8ed7fda852434b6331aebda309b292d726e05adce1c87c115` |
| Fresh user-message exclusions | `aaaf48bcaca980b23603ee34d22c3c2844d4a0d59588460e52f3bb542a8f5add` |
| Existing development user-message exclusions | `b51e2e191ea07416045ec53f66ff110e143ae91a49dc5156eb8d028e0830a8ae` |

The local corpus manifest also binds tool implementation hashes, each selected
source split, per-source licensing, exact sample lengths, quotas and exclusions.
The prepared manifest binds that corpus manifest. Raw datasets and weights stay
outside the public repository.

To reproduce preparation from existing local v2 artifacts, choose new destinations
if the shown outputs already exist:

```sh
uv run python scripts/build-native-english-v6.py \
  --old data/instructions-v2 \
  --output data/native-english-v6-source --prepare data/native-english-v6 \
  --exclude-prompts examples/evaluations/native-english-v6-prompts.jsonl \
  --exclude-prompts examples/evaluations/native-development-prompts.jsonl
```

Eight portable data tests cover deterministic generation, diverse targets and
schema contexts, executable Python fixture answers, real calculator/search
adapter shapes, prompt-only exclusion parsing, user-message/arithmetic separation,
actual assistant-decision quotas, attribution, atomic preparation and overwrite
refusal. Passing these tests demonstrates data-pipeline behavior, not learned
model competence. Initial development outcomes and completed token-loss compatibility
checks are reported above. Adapted-objective development measurements continue;
final candidate selection, fresh scores and any promotion decision remain pending.
