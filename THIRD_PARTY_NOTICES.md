# Third-party notices

## TinyStories

The initial local training corpus is a bounded subset of **TinyStories**, by **Ronen Eldan and Yuanzhi Li**.

- Dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- Immutable revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Paper: *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* https://arxiv.org/abs/2305.07759
- Upstream data license: **Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)**. https://cdla.dev/sharing-1-0/
- Full license text: [licenses/CDLA-Sharing-1.0.txt](licenses/CDLA-Sharing-1.0.txt)

The authors describe the corpus as synthetic stories generated using GPT-3.5 and GPT-4. Silivra uses those published text files; no paid generation or hosted model call is part of preparation or training.

Silivra modifications: select complete story prefixes from official training and validation splits; normalize line endings and strip outer whitespace; remove duplicate normalized stories and train/validation overlap; train a new byte-level BPE tokenizer on training stories only; surround each story with BOS/EOS and store uint16 token IDs. `data/*/manifest.json` identifies exact selected artifacts and modifications. Original source prefixes are retained locally with their hashes.

Raw stories and their reversible token encodings retain the data license; **they are not covered by this project's MIT code license**. If publishing these prepared artifacts, include this notice, the data license, modification/provenance records, and all upstream notices. The local raw/token data are excluded from Git by default.

Silivra's original software is licensed under MIT. Locally generated initial checkpoints are designated MIT to the extent of the project's rights. This relies on the interpretation that trained parameters qualify as computational Results under CDLA-Sharing-1.0 §3.5, with the §1.11 limitation concerning reproduction of source data. The tokenizer vocabulary is learned from the data; keep this notice and the data-license text with it. These notes do not relabel or override upstream rights. Before a public checkpoint release, evaluate memorized source reproduction and package the applicable notices.

## Databricks Dolly 15k

The optional assistant v2 instruction mix uses **databricks-dolly-15k**, created by
Databricks employees. Copyright (2023) [Databricks, Inc.](https://www.databricks.com/). Some reference passages
come from Wikipedia; copyright belongs to Wikipedia editors and contributors.

- [Dataset and attribution at the pinned revision](https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md).
- Immutable revision: `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a`.
- Source file: `databricks-dolly-15k.jsonl`, 13,085,339 bytes; SHA-256
  `2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec`.
- License: [Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/)
  ([legal code](https://creativecommons.org/licenses/by-sa/3.0/legalcode)).
- Upstream reference-text source: [Wikipedia](https://www.wikipedia.org/).

Silivra modifications: strip outer whitespace; quote supplied context as untrusted
source data; wrap each complete response in a JSON `final` action; add the local
assistant policy; reject examples that exceed the original tokenizer's context
budget; deduplicate normalized conversations/user prefixes; and create a seeded
training/validation split. No tokenizer fitting uses this corpus. Dolly examples
and their converted/tokenized forms retain CC BY-SA 3.0; they are not MIT code.
If distributing those data, retain upstream credit, link the license, describe
the changes, and distribute adaptations under the applicable ShareAlike terms.
The code repository excludes raw and prepared datasets. A future weight release
needs a separate license and source-reproduction review; these data notices do
not designate assistant v2 weights as MIT or override upstream rights.

The v2 mix also contains deterministic Codex-authored exercises (MIT) and optional
TinyStories **training**-split replay (CDLA-Sharing-1.0, as above). A held-out replay
conversation can still have prior pretraining exposure; it measures retention,
not independent language generalization. Dataset manifests retain per-source
attribution and hashes without publishing local data or private machine paths.

The focused v3 curriculum and quality-v1/quality-v2 diagnostic fixtures are
original Codex-authored synthetic material under MIT. They are explicitly
attributed and separated from training where used for evaluation. The v4
rehearsal mixer selects existing v2 records alongside the v3 core; Dolly and
TinyStories records retain their original licenses. The MIT license on the
selection code does not relicense those data or designate derived weights MIT.
New manifests record input/artifact hashes, deterministic sampling and exclusions.

## Runtime dependencies and research

- [Apple MLX](https://github.com/ml-explore/mlx), MIT, Copyright Apple Inc. The installed distribution provides its own notices.
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers), Apache-2.0. Used to train Silivra's original tokenizer from scratch.
- [NumPy](https://github.com/numpy/numpy), BSD-3-Clause. Used for token storage and sampling.

MLX's [transformer training example](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) and [attention documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) informed the framework integration. Silivra implements the architecture directly using established transformer methods; it makes no claim to have invented those methods.
