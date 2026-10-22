# Contributing to Silivra

Read [the architecture](docs/ARCHITECTURE.md), [roadmap](docs/ROADMAP.md), and
[model card](docs/MODEL_CARD.md). Keep model capability claims tied to evaluations.

```sh
uv sync --locked
uv run python -m unittest discover -s tests -v
```

The complete suite includes small numerical MLX tests and needs Apple silicon.
Portable runtime checks run in Linux CI using the explicit module list in
`.github/workflows/tests.yml`. Network tests use fixtures and never need API keys.

Changes to data or learning objectives need provenance, licensing and held-out
validation. Keep base checkpoints immutable; write new runs separately. Tests should
exercise behavior and failure paths, including scope isolation and permission denial.
Training or assistant results need their config, checkpoint/data hashes, limits,
raw failures, and a clear separation from mocked runtime results.

Do not submit private state, credentials, raw datasets or large model artifacts.
Local Codex instructions are deliberately ignored; public architecture and workflow
documentation are the shared project contract.
