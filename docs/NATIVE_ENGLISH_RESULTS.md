# Native English v5 results — evaluated 2026-10-26

The original Silivra model learned some narrow conversational and uncertainty
patterns. It did **not** become a reliable English, reasoning, coding, or research
assistant. The general default remains v2. V5 is available for explicit
experimental testing with the limits below.

## Training and preserved lineage

This is real assistant-only weight training of Silivra's own
14,162,304-parameter model, starting from immutable v2 step 8,000. It retains the
original 4,096-token tokenizer and 1,024-token context. No external model weights,
tokenizer, hosted inference, private chats, or live memories were used.

The original AI-authored MIT curriculum contains 2,454 training and 371 validation
conversations, producing 3,036 and 441 assistant decisions. It covers short
conversation, elementary English definitions, copying and supplied facts,
uncertainty, addition/comparison, calculator actions, and web-search actions.
Numeric pairs, copied words, search topics, and conversational paraphrases are
reserved before generation. English definitions share concepts but reserve
question wording. Tool observations are fictional fixtures; calculator values
are computed by the real pure arithmetic implementation. Mocked tests verify
that their result wrappers match the actual calculator and search adapters.
The reserved `example.com` search URLs are training placeholders, never evidence
of live retrieval.

The configuration permits 2,000 updates, batch eight, a saved warmup/cosine
schedule, and a 6 GiB advisory MLX allocation budget. The recorded invocation
had a 13-minute loop limit. The process stopped after logging step 1,200 without
a final run summary. The latest complete, hash-verified resumable checkpoint is
step 1,000, containing 543.766 saved training seconds. Later unsaved updates
are not a resumable artifact. Peak logged MLX allocation was 4.691 GiB; this
is not total system memory.

Fixed validation loss selected step 500 before diagnostic generations. Its
257.566 saved training seconds cover 100,602 supervised and 1,688,737 nonpadding
input-token exposures. Validation loss was 0.30418, compared with 0.30907 at
step 1,000. All original checkpoints and data remain unchanged. No additional
training was performed to finish this evaluation.

Selected weights SHA-256:
`ca85e201eefed4b555b447db8cdad358b5c289ff57be08fe2df7be8e1523f35d`.

## Frozen diagnostic outcomes

Both checkpoints use greedy native generation with a 96-token output limit.
Tools are proposed, never executed by these suites. Scores require exact parsed
actions, except the suites' predeclared finite whole-answer alternatives.
The [complete report](results/native-english-v5.json) retains public diagnostic
generations, failures, artifact hashes and the original-tokenizer checks.

| Measure | V2 step 8,000 | V5 selected step 500 |
| --- | ---: | ---: |
| Historical assistant diagnostic | 0/6 | 0/6 |
| Earlier independent assistant diagnostic | 0/15 | 0/15 |
| Practical quality-v1 | 3/40 | 1/40 |
| Practical quality-v2 | 0/24 | 2/24 |
| Combined four diagnostics | 3/85 | 3/85 |
| Narrow current-curriculum validation sample | 5/94 | 31/94 |
| Valid actions, four diagnostics | 72/85 | 84/85 |
| Valid actions, narrow sample | 58/94 | 93/94 |
| Current instruction validation loss | 2.40137 | 0.30418 |
| Prior v2 instruction validation loss | 2.98620 | 4.24804 |
| Original unseen story validation loss | 3.06644 | 3.84780 |

Lower loss is better. Prior instruction loss worsened **42.26%** and story loss
**25.48%**, exceeding the earlier 5% retention limits. Better JSON formatting and
lower current-curriculum loss do not compensate for unchanged broad accuracy and
lost earlier skills. V5 fails general promotion criteria.

The narrow sample takes up to eight decisions per family in fixed record-ID
order, with six network-unavailable cases. It shares training templates and is
not an independent intelligence benchmark. Zero exact normalized user-prefix
overlaps were found with current or prior v2 training for the four diagnostic
suites and sampled validation decisions. This does not establish absence of
semantic or pretraining contamination. The existing practical suites have informed
earlier development and are development diagnostics, not a fresh blind benchmark.

## What improved, and what failed

| Narrow family | V2 | V5 |
| --- | ---: | ---: |
| Conversation | 0/8 | 5/8 |
| Missing-information honesty | 0/8 | 8/8 |
| Search unavailable | 0/6 | 6/6 |
| Calculator action and correct operands | 0/8 | 1/8 |
| Reading supplied calculator observations | 0/8 | 6/8 |
| Numeric comparison | 5/8 | 5/8 |
| Unaided addition | 0/8 | 0/8 |
| English-definition paraphrases | 0/8 | 0/8 |
| Exact copying | 0/8 | 0/8 |
| Supplied-fact grounding | 0/8 | 0/8 |
| Search action and correct query | 0/8 | 0/8 |
| Reading supplied search observations | 0/8 | 0/8 |

Wrong calculator operands and memorized search queries remain common. For the
held-out definition wording, the model incorrectly asked for web search instead
of answering. Search observations often produced a remembered training topic
rather than the supplied topic. These are substantive failures, even when the
JSON is valid.

A separate native smoke check produced a relevant greeting and a correct noun
definition for short familiar wording. It also answered `What is 31 + 48?` with
`3`. Those development observations are not extra scored successes. Web retrieval
and exact arithmetic remain useful explicit runtime tools, with their results
labelled as tool output. They do not establish that this checkpoint has learned
to choose tools or interpret evidence reliably.

## Reproduce and select explicitly

Use the [training and evaluation recipe](TRAINING.md#native-english-v5-experiment)
with the local ignored artifacts. To inspect this exact experimental checkpoint:

```sh
uv run silivra --model runs/silivra-native-english-v5-runtime/checkpoints/step-0000500
```

The general code default stays v2. Future quality work needs broader, reviewed
original-model training and independent held-out gains while retaining earlier
language/instruction competence. This experiment does not complete that milestone.
