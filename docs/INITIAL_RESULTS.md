# Initial training results — 2026-10-22

Silivra Tiny was initialized from random weights and trained locally on the tester MacBook. No pretrained weights or tokenizer were loaded. This is an early English story-completion experiment; language-model loss and generation speed do not establish assistant capability.

## Measured results

| Measurement | Result |
| --- | --- |
| Tester | Apple M4 MacBook Air, 16 GB unified memory |
| Environment | macOS 27.0.1; Python 3.12.14; MLX 0.32.4 |
| Power | Battery during training and benchmark; no macOS thermal warning observed during training |
| Parameters | 14,162,304 |
| Updates completed | 2,500 / 2,500 |
| Training token exposures | 10,240,000 (repeated sampling of 6,711,760 prepared training tokens) |
| Training-loop session | 898.68 seconds (14.98 minutes), including in-loop evaluation/checkpoint writes |
| Aggregate throughput over that session | 11,395 token exposures/second |
| Initial validation loss | 8.411208 nats/token |
| Final validation loss, same fixed 12-batch sample | 2.381120 nats/token |
| Final validation perplexity, same sample | 10.8170 |
| Separate 32-batch validation estimate | 2.361977 loss; 10.6119 perplexity |
| Peak training MLX allocation | 2.367 GiB |
| Training process peak RSS | 0.211 GiB (separate measurement; not total machine memory) |
| Warm generation median, includes prefill | 588.3 output tokens/second |
| Warm generation range, 3 trials | 489.4–688.8 output tokens/second |
| Cached decode median, excludes first token | 600.7 tokens/second |
| Warm median first-token latency | 3.66 ms |
| Benchmark peak MLX allocation after warmup/reset | 0.0643 GiB |
| Benchmark process peak RSS | 0.116 GiB |
| Weight SHA256 | `633efc5b513b7de90de6d1e55d37f45948852d1c19728078a54f7502a6a7cf40` |

The benchmark ran after the trainer exited. It uses one sequence, a fixed 23-token story prompt, greedy sampling, three 128-token trials, and suppresses EOS to time equal-length outputs. Kernels are warm and model loading is excluded. Each trial is short; the observed range is substantial and these are preliminary measurements on one machine. The separately reported cached decode rate counts the 127 tokens after the first token. Training and inference memory counters differ, and neither is total system RAM. No claim of zero system swap usage is made.

Validation samples come only from the separate official TinyStories validation split; they are not used to fit the tokenizer or optimizer. The fixed evaluation seed is 12345. The comparable initial/final sample uses 12 batches × 16 sequences × 256 predictions. The separate estimate uses 32 batches with the same sequence length and batch size. Windows are distinct within each batch and may repeat across batches. There is no independent assistant, factual, coding, or general benchmark.

## Data provenance

- Training: 30,000 unique complete stories; 6,711,760 tokens; 27,000,832 bytes transferred.
- Validation: 1,000 unique complete stories; 197,312 tokens; 851,968 bytes transferred.
- Tokenizer: original 4,096-token byte-level BPE, trained on training stories only; SHA256 `1640a5301686bd4970d6bd5c1b27926f71227bfd11b9e2d59375cbe33c035269`.
- Corpus revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
- Corpus license: CDLA-Sharing-1.0; [attribution and modifications](../THIRD_PARTY_NOTICES.md).

## Unedited generation samples

Temperature 0.7, top-k 40, seed 42, maximum 128 new tokens. These are the first three fixed prompts tested after the run, without retrying for quality. One output reaches the token cap. They show learned English patterns together with inconsistent objects, names, and events.

Prompt: `Once upon a time, a little fox`

```text
Once upon a time, a little fox named Paul was walking in the woods with his mommy. He was walking around and having lots of fun.
One day, Paul was playing in the garden when he saw a big, dark cave. He wanted to climb it, so he reached up and picked it up. He felt something special for him to rock in the cave. He was so happy that he knew he would stop and look at it.
So, his mum went to the cave and said, "Hi there, Pete. I have a surprise for you!”
Pretan was so excited. He ran to the cave and started to run out of the cave
```

Prompt: `This morning, Lily found a small red box.`

```text
This morning, Lily found a small red box. She was so happy! She picked it up and put it in her pocket. She was so happy and took it home with her new ball.
But then, Lily's mom came into the room. It was not a good treat for Lily. Lily was sad and asked her mom if she could have some juice. Her mom said yes, and they sat down.
Lily was happy she got to play and clean up her room. She loved taking her toys everywhere in the closet. But when she was not looking, it was too dirty and she couldn't play anymore.
```

Prompt: `The little robot wanted to help its friend.`

```text
The little robot wanted to help its friend. He hopped around the house and found an interesting spot. He put it in his pocket and started to play with it. He kept it up and down, making the walls beautiful.
One day, the sun shone in the sky and the sun shone brightly. The sun shone brightly, and the sun started to set. The family was so proud of their work. They decided to explore the world together.
The family was very happy. They thanked the little family for the sun, and the sun set. They all had a wonderful time together.
```

## Verification and artifacts

The 25 automated tests cover data preparation and split isolation, model causality, every-parameter gradients, cached/full-forward equivalence through context growth and rollover, weight reloads, sampling, and exact stop/resume equivalence of weights, optimizer, and RNG state. The CLI and export are also exercised directly. Checkpoint files and exported bundles carry SHA256 integrity records.

The local run records are in `runs/silivra-tiny/`: `run.json`, `summary.json`, `metrics.jsonl`, `benchmark.json`, `evaluation.json`, and `samples.json`. Checkpoint `step-0002500` is both best and latest in this run. These files and all raw/tokenized training data are excluded from Git. The export includes results and source, weights, tokenizer, settings and data provenance; it excludes raw/tokenized stories and optimizer state. The original run retains optimizer/RNG state for resume.

At this initial training milestone, no GitHub repository, external model host, or public release had been created. Subsequent assistant/runtime work and public-source setup are documented in [CAPABILITY_RESULTS.md](CAPABILITY_RESULTS.md) and the [roadmap](ROADMAP.md). These initial pretraining measurements remain unchanged.
