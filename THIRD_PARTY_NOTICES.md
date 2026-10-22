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

## Runtime dependencies and research

- [Apple MLX](https://github.com/ml-explore/mlx), MIT, Copyright Apple Inc. The installed distribution provides its own notices.
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers), Apache-2.0. Used to train a new tokenizer; no pretrained tokenizer is downloaded.
- [NumPy](https://github.com/numpy/numpy), BSD-3-Clause. Used for token storage and sampling.

MLX's [transformer training example](https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm) and [attention documentation](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.scaled_dot_product_attention.html) informed the framework integration. Silivra implements the architecture directly using established transformer methods; it makes no claim to have invented those methods.
