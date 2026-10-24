# Calculator integration diagnostic

Four original Codex-authored synthetic MIT cases exercise the local model choosing
the real bounded calculator, interpreting its results, and combining a real file
observation with a calculation. These are tool integration checks, not unaided
model mathematics or a general intelligence benchmark. They do not train weights
or modify either tokenizer. Upstream pretraining overlap cannot be audited.

The first version was authored after the v1 diagnostic exposed wrong mental
arithmetic. It must be reported separately from the frozen unaided v1/v2 results.
The floor/remainder case requires two separate operations in its request; inspect
the recorded actions as well as its final-content grade.

```sh
uv run python scripts/measure-pretrained.py \
  --run runs/silivra-qwen3-8b \
  --suite examples/evaluations/calculator-v1.jsonl \
  --output docs/results/pretrained-8b-calculator.json
```

The shared evaluator keeps strict action parsing, actual tool observations,
temporary synthetic state, source hashes, output budgets and bounded runtime.
