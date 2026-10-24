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

### Optional external local assistant

`setup-assistant` downloads **Qwen3-8B**, developed by the Qwen team at Alibaba,
in MLX Community's 4-bit conversion. The 8,190,735,360-parameter architecture,
pretrained weights and tokenizer are external artifacts; they are not Silivra's
original randomly initialized model or its learned tokenizer.

- Original model: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B).
- Original license revision: `b968826d9c46dd6066d109eabc6255188de91218`.
- Conversion: [mlx-community/Qwen3-8B-4bit](https://huggingface.co/mlx-community/Qwen3-8B-4bit).
- Immutable conversion revision: `545dc4251c05440727734bcd94334791f6ab0192`.
- License: Apache License 2.0; setup retains the upstream `LICENSE` and conversion
  model card beside the downloaded weights. Original license SHA-256:
  `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e`.
- Quantized weight SHA-256:
  `f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8`.
- Tokenizer SHA-256:
  `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.

Silivra performs inference only with these artifacts, without fine-tuning or
relabeling ownership. Quantization was performed upstream. The provisioner records
all file sizes/hashes; downloaded artifacts are ignored by Git and are not bundled
with the source repository. Distribution must retain applicable upstream notices.

The optional, unfinished text-only Qwen3.5-9B trial uses
[Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) at revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`, and
[MLX Community's conversion](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit)
at `938d8919941c6e7efd3c7150eff7fe9d12afa631`, under Apache-2.0.
The text backbone has 8,953,803,264 parameters; MLX-LM excludes vision/MTP weights.
Its upstream license is retained with SHA-256
`bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a`.
Download support does not establish native inference or capability results.

- [Apple MLX-LM](https://github.com/ml-explore/mlx-lm), MIT, Copyright Apple Inc.
- [Hugging Face Transformers](https://github.com/huggingface/transformers), Apache-2.0.
- [Outlines-core](https://github.com/dottxt-ai/outlines-core), Apache-2.0.
  Its Rust automaton constrains JSON action syntax; it does not train model weights.
- Jinja2/MarkupSafe (BSD-3-Clause), SentencePiece (Apache-2.0), Protobuf (BSD-3-Clause),
  Safetensors (Apache-2.0), and other locked transitive dependencies retain their
  installed distributions' license notices.

The pretrained-v1/pretrained-v2 and calculator-v1 diagnostics are original
Codex-authored synthetic MIT material.
It is evaluation-only and does not train either model; upstream pretraining
contamination cannot be ruled out for general knowledge and elementary exercises.

- [Apple MLX](https://github.com/ml-explore/mlx), MIT, Copyright Apple Inc. The installed distribution provides its own notices.
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers), Apache-2.0. Used to train the original tokenizer; the separate optional Qwen backend downloads its own pretrained tokenizer.
- [NumPy](https://github.com/numpy/numpy), BSD-3-Clause. Used for token storage and sampling.

MLX's [transformer training example](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) and [attention documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) informed the framework integration. Silivra implements the architecture directly using established transformer methods; it makes no claim to have invented those methods.
