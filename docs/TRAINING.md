# Training Silivra

Silivra keeps its own model, learned tokenizer and from-scratch MLX pretraining
path. Instruction tuning starts from a verified local checkpoint and writes a
separate run. Adding tools to the runtime gives the model actions it can request;
the model still needs suitable training and held-out evaluation to request those
actions reliably.

The current TinyStories checkpoint learned story continuation. Its pretraining
does not establish coding, broad factual knowledge, instruction following or
tool selection. The small examples in this repository are original, AI-assisted
format fixtures licensed under MIT. Their tool results are fictional. They are
useful for exercising the training pipeline, not a serious coding or knowledge
corpus. Falling loss on them is not evidence of general intelligence or a useful
coding assistant.

## Instruction records

Use UTF-8 JSONL with one conversation ending at an assistant decision per line. Every record must have
exactly `id`, `source` and `messages`. Source attribution requires `name`,
`license` and `url`; `author` and `revision` are optional. Record a real source
reference, license and revision for imported data. The preparation step validates
the presence of metadata; it does not verify a license's legal suitability.

```json
{"id":"original-add-example","source":{"name":"Original examples","license":"MIT","url":"local:my-data/train.jsonl","author":"Silivra contributors"},"messages":[{"role":"system","content":"Return one JSON action."},{"role":"user","content":"What is 2 + 3?"},{"role":"assistant","content":"{\"final\":\"5\"}"}]}
```

An optional system message comes first. A user message is followed by an
assistant action. An assistant can return either `{"final":"answer"}` or
`{"tool":"read_file","arguments":{"path":"main.py"}}`. No additional action
fields are accepted. If a tool action continues in the same record, it must be
followed by a `tool` message whose `name` matches the action, then another assistant
action. A record may end with an assistant final answer or a pending tool proposal;
the latter does not pretend the tool ran. A later user turn may follow an earlier final answer.
Tool results remain untrusted data. Include tool failures and honest uncertainty
in real instruction data, alongside successful actions.

The shared `silivra.instructions.format_prompt` format is
`silivra-role-json-v1`. It uses the existing tokenizer and ordinary role headers:

```text
### system
{"content":"Return one JSON action."}

### user
{"content":"What is 2 + 3?"}

### assistant
```

Non-assistant content is JSON-escaped on one line. Tool result lines contain
`name` and `content`. Assistant lines contain the action object itself, without
double serialization. Newlines, quotes, Unicode line separators and reserved
token spellings inside content cannot become additional role headers or literal
special tokens. The escaping round-trips the original text; it does not solve
semantic prompt injection. Use this same formatter during training and chat.

## Prepare data

Pin the concrete base checkpoint so later pretraining cannot change the base
behind a `latest.json` pointer. For example, after the existing 2,500-step run:

```sh
silivra prepare-instructions \
  --train examples/instructions/train.jsonl \
  --valid examples/instructions/valid.jsonl \
  --tokenizer runs/silivra-tiny/checkpoints/step-0002500/tokenizer.json \
  --output data/instructions \
  --max-seq-len 512
```

Preparation reads local files only. It never downloads a corpus or trains a new
tokenizer. The output must be new or empty and is published atomically. Limits
are 100,000 conversations and 64 MiB per split, 256 KiB per JSONL line, 128
messages per conversation, 64 KiB per message and 32 million prepared tokens
across the splits. Padded storage is additionally capped at 32 million cells per
split. The Python API permits stricter `max_records` and `max_bytes` limits.

Duplicate IDs and normalized duplicate conversations fail, including duplicates
whose source metadata differs. A user-turn sequence that appears in both train
and validation also fails, even if the assistant answers differ. This is exact
normalized leakage detection, not semantic or near-duplicate detection. Separate
whole projects, source documents and problem families when constructing a real
held-out set, and audit overlaps with pretraining and evaluation data too.

Each assistant turn produces one sample containing BOS, complete earlier
messages, the assistant prefix, the action and EOS. Only that turn's action and
EOS receive loss; earlier assistant turns, user/system/tool content and padding
receive zero loss. Prefix and answer are encoded separately so the inference
prefix token IDs match exactly. A sample that exceeds the requested context
fails with its record and turn number. Answers and tool results are never
silently truncated to fit. Do not increase the prepared context beyond the base
model's context limit.

The output contains a checksummed tokenizer, token/mask archives, normalized
source JSONL and a manifest carrying input hashes, output hashes and attribution.
Keep the source records private when their contents require it. Training does
not automatically ingest the live memory database or chat logs.

## Fine-tune and resume

For a small pipeline demonstration, put this configuration in a local JSON file:

```json
{"steps":50,"batch_size":2,"learning_rate":0.0001,"weight_decay":0.01,"eval_every":25,"eval_batches":1,"save_every":25,"log_every":10,"seed":42,"memory_gib":6.0}
```

```sh
silivra finetune \
  --data data/instructions \
  --base runs/silivra-tiny/checkpoints/step-0002500 \
  --run runs/silivra-instruct \
  --config /absolute/path/to/your/sft-config.json \
  --max-minutes 2
```

Choose an absent or empty output directory separate from the base checkpoint and
data. The trainer verifies the base checkpoint checksums, tokenizer identity,
prepared-data hashes, vocabulary and context compatibility before updating any
weights. It uses assistant-only cross-entropy, AdamW, gradient clipping and the
existing Apple silicon memory checks. Validation reports token-weighted
assistant-only loss on reproducibly sampled held-out batches. This loss is a
format/data diagnostic; evaluate actual tasks separately.

The default without `--config` is 1,000 steps with batches of four; start with an
explicit small configuration. A wall-time limit or interrupt saves a resumable
checkpoint after the latest completed update. The checkpoint contains model
weights, AdamW state, NumPy and MLX RNG state, data manifest, base lineage hashes,
configuration and training counters. Checkpoints are published atomically.
`latest.json` points to the latest checkpoint; `best.json` tracks the lowest
measured instruction validation loss.

```sh
silivra finetune \
  --data data/instructions \
  --base runs/silivra-tiny/checkpoints/step-0002500 \
  --run runs/silivra-instruct \
  --resume \
  --max-minutes 2
```

Resume automatically loads the saved configuration when `--config` is omitted.
The original data, base artifact and full training configuration, including total
steps, must match. The time budget may change. RNG state and sampling continue
from the checkpoint. Floating-point parallel reductions can introduce small
rounding differences between otherwise identical MLX runs; bitwise equality is
not promised. `tokens_seen` and `supervised_tokens_seen` count sampled assistant
action and EOS labels; `input_tokens_seen` counts all nonpadding next-token labels.
They are SFT exposures and do not include the base model's pretraining count.

## Build capability with evidence

Grow the project through versioned data and held-out evaluations. A useful next
curriculum needs properly licensed code and technical text, diverse instruction
tasks, tool selection and argument correctness, multi-step debugging, tool
failure recovery, citation use, memory correction and deletion, and handling
untrusted retrieved content. Keep authored tests and evaluation cases separate
from training records. Track dataset revisions, model checkpoints, runtime tool
versions and regressions together.

Continue pretraining with stronger technical data before expecting instruction
tuning to supply knowledge absent from the base. The current 512-token context
and small architecture limit how much code, tool schema, memory and history can
fit. Larger context or architecture changes require a deliberate training and
evaluation migration; the trainer rejects incompatible prepared data rather
than implying such a migration happened automatically.

Test protocol validity, correct tool names and arguments, grounded final
answers, actual code-test outcomes, recovery behavior and memory retrieval
precision. A format-valid checkpoint can still be incapable of solving tasks.
The focused tests cover masks, escaping, data bounds, leakage rejection,
corruption checks and a tiny synthetic interrupted/resumed MLX run. They do not
establish production coding or web-research quality.

## Procedural capability curriculum

The original AI-assisted MIT curriculum generator emits 400 training and 100
validation conversations across ten balanced families: copy, exact response,
addition, subtraction, multiplication, Python expressions, Python functions,
read-file proposals, file-search proposals and web-search proposals. Validation
uses separate words, filenames and arithmetic operands. Templates are shared:
this measures narrow generalization within the curriculum, not independent broad
intelligence. Tool records stop at an honest proposed action; no tool ran.

```sh
uv run python scripts/build-curriculum.py \
  --output data/curriculum-v1 \
  --tokenizer runs/silivra-tiny/checkpoints/step-0002500/tokenizer.json
uv run silivra prepare-instructions \
  --train data/curriculum-v1/train.jsonl --valid data/curriculum-v1/valid.jsonl \
  --tokenizer runs/silivra-tiny/checkpoints/step-0002500/tokenizer.json \
  --output data/instructions
uv run silivra finetune --data data/instructions \
  --base runs/silivra-tiny/checkpoints/step-0002500 --run runs/silivra-instruct \
  --config configs/finetune-curriculum.json --max-minutes 3
uv run silivra evaluate-assistant --run runs/silivra-instruct \
  --suite data/curriculum-v1/evaluation.jsonl --max-new-tokens 64
uv run silivra evaluate --run runs/silivra-instruct --data data/instructions --batches 25
```

The generator reads no network, private state or independent evaluation suite.
With the initial tokenizer the longest complete sample uses 440 tokens, and the
largest prompt uses 405. Each tool example exposes one exact runtime schema.
It records per-split source hashes and the derived validation evaluation hash.
Generation refuses an existing output and the CLI restricts output to ignored
`data/` inside the checkout. Training is a separate explicit command.

`evaluation.jsonl` contains all validation prompts and exact expected actions,
with zero or one advertised tool. Run the identical suite and generation settings
against the base and tuned checkpoint. It never executes proposals. Keep this
measure separate from the six fixed assistant diagnostics and actual multi-step
coding/research performance. The original 512-token context cannot sustain long
observations, memory, tool schemas and conversation history at once.
