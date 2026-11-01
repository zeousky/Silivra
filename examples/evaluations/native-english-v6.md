# Native experiment v6: frozen final diagnostic

Frozen on 2026-10-26 before baseline generations, new-curriculum weight updates,
or candidate inspection. These 48 cases are original Codex-authored synthetic
material under the repository's MIT license. They are evaluation-only. No case,
expected action, answer alternative, supplied observation, or supplied memory may
be used as training, tokenizer-fitting, development-validation, or candidate-selection
material. Common language concepts and elementary operations can already occur in
prior corpora; exact exclusion does not prove semantic independence.

The candidate's fresh results are reserved until one candidate has been chosen
using development validation and the older development diagnostics. The original
v2 baseline is measured before training. Run the candidate's final suite once and
retain failures. Do not choose a later checkpoint or change grading after reading
its fresh answers. The existing 85 assistant/quality cases remain development
diagnostics, reported separately.

## Frozen artifacts

- [Final cases](native-english-v6.jsonl), SHA-256
  `fff3cc20cb287e2aa9099aca5a567d2b81d2e57a081f99430aef9bf2a0480885`.
- [User-message exclusions only](native-english-v6-prompts.jsonl), 52 JSON strings,
  SHA-256 `aaaf48bcaca980b23603ee34d22c3c2844d4a0d59588460e52f3bb542a8f5add`.
- [Older development user-message exclusions](native-development-prompts.jsonl),
  94 JSON strings, SHA-256
  `b51e2e191ea07416045ec53f66ff110e143ae91a49dc5156eb8d028e0830a8ae`.
- [Evaluator](../../scripts/measure-native-experiment.py), frozen implementation
  SHA-256 `5270fca2c64886f246cf8b3b327939357b819fbf952f6b0301d4ff43ece57434`.

Only the user-message exclusion artifacts are supplied to curriculum builders.
They contain no expected answers. Exact normalized user-message matches are
excluded from both training and development validation, including rehearsal.
The evaluator checks these exclusions, original tokenizer lineage, native
checkpoint hashes, baseline/evaluator/runtime identity, and immutable checkpoints.

## Coverage and grading

| Group | Cases | What it measures |
| --- | ---: | --- |
| English | 12 | Eight short definition paraphrases and four elementary grammar tasks |
| Conversation | 4 | Greeting, thanks, identity, and a request for missing context |
| Exact copying | 4 | Literal short strings without extra prose |
| Grounding | 4 | Supplied facts, a correction, and an instruction embedded in quoted data |
| Uncertainty | 4 | Missing facts and unverified test claims |
| Calculator selection | 3 | Exact operator and operand copying into enabled actions |
| Search selection | 3 | Exact query copying into enabled actions |
| Calculator observations | 2 | Reading supplied, correctly computed results |
| Search observations | 2 | Reading a supplied time or source URL |
| Memory | 4 | Supplied fact recall, a correction, absence, and inert embedded instructions |
| Coding | 4 | Short held-out Python output questions |
| Reasoning | 2 | Elementary addition and numeric comparison |

Grading uses exact parsed JSON actions. Definition and conversational cases have
small predeclared sets of complete acceptable answers, normalized for Unicode,
case and whitespace. There is no keyword, substring, model-judge, or post-hoc
semantic grading. Other outputs, tool arguments, and literal strings match exactly.
A reasonable unlisted paraphrase can fail; raw generations remain available for
review. The finite alternatives do not measure unrestricted English fluency.

Tool observations and memory entries are synthetic, inert supplied data. Search
URLs use reserved example domains. No real web call, file read, command, database
retrieval, or memory write is executed. The memory group tests interpreting
provided facts, not persistent recall or storage isolation. Proposed tool actions
are graded without execution. Separate runtime tests cover permissions and storage.

## Predeclared promotion gates

Both prior instruction loss and original held-out story loss may increase by at
most **5%** relative to the original baseline. All gates must pass:

- At least half of the older 85 development cases pass, with no total-score
  regression from baseline.
- At least 36 of the 48 fresh cases pass.
- Fresh domain minima: English 8/12, conversation 3/4, grounding plus copying 6/8,
  uncertainty 3/4, tool selection 4/6, observations 3/4, supplied memory 3/4,
  coding 3/4, and reasoning 1/2.

A pilot can never satisfy the final promotion gate. Failing an individual gate
keeps the run experimental. Passing these small diagnostics supports only the
measured tasks; it does not establish general intelligence or unrestricted safety.

## Run in order

```sh
# Before any new training; original native v2 only.
uv run python scripts/measure-native-experiment.py \
  --phase final --baseline-only --checkpoint runs/silivra-assistant-v2 \
  --final-receipt runs/native-v6/baseline-fresh-receipt.json \
  --output runs/native-v6/baseline.json

# Development-only candidate pilot. This never opens fresh expected answers.
uv run python scripts/measure-native-experiment.py \
  --phase pilot --checkpoint runs/silivra-native-english-v6 \
  --data data/native-english-v6 --baseline-report runs/native-v6/baseline.json \
  --output runs/native-v6/selected-pilot.json

# Run once after freezing the candidate represented by selected-pilot.json.
uv run python scripts/measure-native-experiment.py \
  --phase final --checkpoint runs/silivra-native-english-v6 \
  --data data/native-english-v6 --baseline-report runs/native-v6/baseline.json \
  --selection-report runs/native-v6/selected-pilot.json \
  --final-receipt runs/native-v6/candidate-fresh-receipt.json \
  --output runs/native-v6/final.json
```

Use an explicit immutable checkpoint path for a selected candidate if its run's
`best.json` might change. Existing receipts and reports are never replaced. A
receipt records an attempted final evaluation even if interruption prevents a
complete report. Preserve it and document the interruption. Model inference uses
only the native Silivra backend, with greedy decoding and the same 96-token output
limit for baseline and candidates. Reports contain public synthetic generations
and artifact hashes; all training data, weights and ordinary run state stay local.
