"""Bounded assistant-only SFT from a verified local Silivra checkpoint.

Instruction tuning adjusts an existing model; it does not turn a story-only
pretraining corpus into a knowledge or coding corpus. Output and resume state
remain separate from the original pretrained run.
"""

from dataclasses import asdict, dataclass
import fcntl
from functools import partial
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time

import numpy as np

from .checkpoint import resolve_checkpoint, sha256, verify_checkpoint, write_json
from .instructions import InstructionDataset, PROMPT_FORMAT
from .runtime import memory_stats, require_apple_gpu


@dataclass
class FinetuneConfig:
    steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    eval_every: int = 100
    eval_batches: int = 8
    save_every: int = 100
    log_every: int = 10
    seed: int = 42
    memory_gib: float = 6.0

    def validate(self):
        for name in ("steps", "batch_size", "eval_every", "eval_batches", "save_every", "log_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > 32:
            raise ValueError("batch_size must be at most 32")
        for name, low, high in (("learning_rate", 0, 0.001),
                                 ("memory_gib", 0, 8)):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not low < value <= high:
                raise ValueError(f"Invalid {name}")
        if (type(self.weight_decay) not in (float, int) or not math.isfinite(self.weight_decay)
                or not 0 <= self.weight_decay <= 1):
            raise ValueError("weight_decay must be finite and in [0, 1]")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64)")


def masked_loss(model, x, y, mask):
    """Token-weighted loss; prompt and pad labels have exactly zero weight."""
    import mlx.core as mx
    import mlx.nn as nn
    losses = nn.losses.cross_entropy(model(x), y, reduction="none")
    return mx.sum(losses * mask) / mx.maximum(mx.sum(mask), 1)


def evaluate_loss(model, dataset, batch_size, batches):
    import mlx.core as mx
    rng = np.random.default_rng(12345)
    model.eval()
    total, count = 0.0, 0.0
    for _ in range(batches):
        x, y, mask = (mx.array(value) for value in dataset.batch(rng, batch_size))
        weight = float(mx.sum(mask).item())
        total += float(masked_loss(model, x, y, mask).item()) * weight
        count += weight
    value = total / count
    if not math.isfinite(value):
        raise RuntimeError("Non-finite instruction validation loss")
    return value


def _verified_checkpoint(path):
    path = resolve_checkpoint(Path(path))
    checksums = json.loads((path / "checksums.json").read_text())
    if not {"model.safetensors", "tokenizer.json", "state.json"} <= set(checksums):
        raise ValueError("Checkpoint must checksum model weights, tokenizer and state")
    return path, verify_checkpoint(path)


def _manifest(data_dir):
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "silivra-instructions"
            or manifest.get("prompt_format") != PROMPT_FORMAT):
        raise ValueError("Prepare compatible instruction data before fine-tuning")
    for artifact in (manifest["tokenizer"], *manifest["splits"].values()):
        for path_key, hash_key in (("path", "sha256"), ("source_path", "source_sha256")):
            if path_key not in artifact:
                continue
            path = (data_dir / artifact[path_key]).resolve()
            if not path.is_relative_to(data_dir) or sha256(path) != artifact[hash_key]:
                raise ValueError(f"Instruction data checksum mismatch: {artifact[path_key]}")
    return manifest


def _save(run_dir, model, optimizer, architecture, config, step, rng, data_dir,
          base, best_loss, validation_loss, seconds, supervised_tokens_seen, input_tokens_seen):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    parent = run_dir / "checkpoints"
    parent.mkdir(exist_ok=True)
    destination = parent / f"step-{step:07d}"
    if destination.exists():
        verify_checkpoint(destination)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=parent))
    try:
        model.save_weights(str(staging / "model.safetensors"))
        mx.savez(str(staging / "optimizer.npz"), **dict(tree_flatten(optimizer.state)))
        mx.savez(str(staging / "mlx_rng.npz"), **{str(i): item for i, item in enumerate(mx.random.state)})
        shutil.copyfile(data_dir / "tokenizer.json", staging / "tokenizer.json")
        shutil.copyfile(data_dir / "manifest.json", staging / "data_manifest.json")
        write_json(staging / "state.json", {
            "format_version": 2, "training_mode": "instruction_sft", "prompt_format": PROMPT_FORMAT,
            "step": step, "model_config": architecture.to_dict(), "training_config": asdict(config),
            "tokens_seen": supervised_tokens_seen, "supervised_tokens_seen": supervised_tokens_seen,
            "input_tokens_seen": input_tokens_seen,
            "token_count_basis": "tokens_seen = sampled assistant-action and EOS labels; input_tokens_seen = all nonpadding next-token labels",
            "numpy_rng": rng.bit_generator.state, "base_checkpoint": base,
            "data_manifest_sha256": sha256(data_dir / "manifest.json"),
            "best_validation_loss": best_loss, "validation_loss": validation_loss,
            "training_seconds": seconds, "initialization": "verified local Silivra checkpoint",
            "loss": "assistant action and EOS tokens only; context and padding masked",
        })
        write_json(staging / "checksums.json", {
            item.name: sha256(item) for item in staging.iterdir() if item.is_file()
        })
        os.replace(staging, destination)
        write_json(run_dir / "latest.json", {"checkpoint": str(destination.relative_to(run_dir)), "step": step})
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def finetune(data_dir: Path, base_checkpoint: Path, run_dir: Path, config: FinetuneConfig | None = None,
             *, resume: bool = False, max_minutes: float = 15.0) -> dict:
    """Fine-tune or resume a bounded local instruction run.

    Total steps, base artifact hashes, tokenizer, data manifest and all optimizer
    settings must match on resume. Only the wall-time budget may change. MLX
    parallel floating-point reductions can differ by rounding between runs.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten
    from .config import ModelConfig
    from .model import SilivraLM

    data_dir, run_dir = Path(data_dir).resolve(), Path(run_dir).resolve()
    if config is None:
        if resume:
            saved = json.loads((resolve_checkpoint(run_dir) / "state.json").read_text())
            config = FinetuneConfig(**saved["training_config"])
        else:
            config = FinetuneConfig()
    config.validate()
    if type(max_minutes) not in (int, float) or not math.isfinite(max_minutes) or max_minutes <= 0:
        raise ValueError("max_minutes must be finite and positive")
    base_location = Path(base_checkpoint).resolve()
    if (run_dir.is_relative_to(base_location) or base_location.is_relative_to(run_dir)
            or run_dir.is_relative_to(data_dir) or data_dir.is_relative_to(run_dir)):
        raise ValueError("Fine-tuning output must be separate from base checkpoint and instruction data")
    base_path, base_state = _verified_checkpoint(base_location)
    base = {"state_sha256": sha256(base_path / "state.json"),
            "weights_sha256": sha256(base_path / "model.safetensors"),
            "tokenizer_sha256": sha256(base_path / "tokenizer.json")}
    if base_state.get("training_mode") == "instruction_sft":
        base.update(training_mode="instruction_sft", finetuning_step=base_state["step"],
                    parent=base_state["base_checkpoint"])
    else:
        base["pretraining_step"] = base_state["step"]
    manifest = _manifest(data_dir)
    architecture = ModelConfig.from_dict(base_state["model_config"])
    if (manifest["tokenizer"]["sha256"] != base["tokenizer_sha256"]
            or manifest["tokenizer"]["vocab_size"] != architecture.vocab_size
            or manifest["max_seq_len"] > architecture.max_seq_len):
        raise ValueError("Instruction tokenizer or context limit does not match the base checkpoint")
    datasets = {split: InstructionDataset(data_dir / manifest["splits"][split]["path"],
                 architecture.max_seq_len, architecture.vocab_size) for split in ("train", "valid")}
    hardware = require_apple_gpu(config.memory_gib)
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Fine-tuning run must be absent or empty; use --resume for an existing run")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another trainer is using this fine-tuning run") from None
    stopped, handlers = {"requested": False}, {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            handlers[sig] = signal.signal(sig, lambda *_: stopped.update(requested=True))
        previous, checkpoint = None, base_path
        if resume:
            checkpoint, previous = _verified_checkpoint(run_dir)
            checksums = json.loads((checkpoint / "checksums.json").read_text())
            if not {"optimizer.npz", "mlx_rng.npz", "data_manifest.json"} <= set(checksums):
                raise ValueError("Resume checkpoint is missing checksummed optimizer/RNG/provenance state")
            if previous.get("format_version") != 2:
                raise ValueError("Resume requires SFT checkpoint format_version=2 with exact token counters; start a separate new run for older experimental checkpoints")
            if (previous.get("training_mode") != "instruction_sft"
                    or previous.get("prompt_format") != PROMPT_FORMAT
                    or previous["training_config"] != asdict(config)
                    or previous["base_checkpoint"] != base
                    or previous["data_manifest_sha256"] != sha256(data_dir / "manifest.json")):
                raise ValueError("Resume requires the original SFT base, data and training configuration")
        mx.random.seed(config.seed)
        rng = np.random.default_rng(config.seed)
        model = SilivraLM(architecture)
        model.load_weights(str(checkpoint / "model.safetensors"))
        optimizer = optim.AdamW(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        step, best_loss, seconds_before = 0, float("inf"), 0.0
        supervised_tokens_seen, input_tokens_seen = 0, 0
        if previous:
            optimizer.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.npz")).items()))
            rng.bit_generator.state = previous["numpy_rng"]
            for key, array in mx.load(str(checkpoint / "mlx_rng.npz")).items():
                mx.random.state[int(key)][:] = array
            step, best_loss, seconds_before = previous["step"], previous["best_validation_loss"], previous["training_seconds"]
            supervised_tokens_seen, input_tokens_seen = previous["supervised_tokens_seen"], previous["input_tokens_seen"]
        mx.eval(model.parameters(), optimizer.state)
        baseline = evaluate_loss(model, datasets["valid"], config.batch_size, config.eval_batches)
        if not resume:
            write_json(run_dir / "run.json", {
                "format_version": 1, "training_mode": "instruction_sft", "prompt_format": PROMPT_FORMAT,
                "base_checkpoint": base, "model_config": architecture.to_dict(), "hardware": hardware,
                "training_config": asdict(config), "baseline_validation_loss": baseline,
                "data_manifest_sha256": sha256(data_dir / "manifest.json"),
            })
        print(json.dumps({"event": "sft_start", "step": step, "validation_loss": baseline}), flush=True)
        loss_and_grad = nn.value_and_grad(model, masked_loss)
        state = [model.state, optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step_fn(x, y, mask):
            loss, grads = loss_and_grad(model, x, y, mask)
            grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            return loss, norm

        start = time.monotonic()
        reason, validation = "completed", baseline

        def save():
            nonlocal best_loss
            improved = validation < best_loss
            best_loss = min(best_loss, validation)
            destination = _save(run_dir, model, optimizer, architecture, config, step, rng, data_dir,
                                base, best_loss, validation, seconds_before + time.monotonic() - start,
                                supervised_tokens_seen, input_tokens_seen)
            if improved:
                write_json(run_dir / "best.json", {"checkpoint": str(destination.relative_to(run_dir)), "step": step})
            return destination

        with (run_dir / "metrics.jsonl").open("a", buffering=1) as log:
            model.train()
            while step < config.steps:
                if stopped["requested"] or time.monotonic() - start >= max_minutes * 60:
                    reason = "interrupted" if stopped["requested"] else "time_budget"
                    break
                x, y, mask = (mx.array(value) for value in datasets["train"].batch(rng, config.batch_size))
                loss, norm = step_fn(x, y, mask)
                mx.eval(loss, norm, state)
                if not math.isfinite(float(loss.item())) or not math.isfinite(float(norm.item())):
                    raise RuntimeError("Non-finite SFT loss or gradients; resume the last good checkpoint")
                step += 1
                supervised_tokens_seen += int(mx.sum(mask).item())
                input_tokens_seen += int(mx.sum(y != manifest["tokenizer"]["pad_id"]).item())
                if step % config.log_every == 0 or step == config.steps:
                    row = {"event": "sft_train", "step": step, "loss": float(loss.item()),
                           "gradient_norm": float(norm.item()), **memory_stats()}
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + "\n")
                if mx.get_peak_memory() > config.memory_gib * 1024**3:
                    reason = "memory_budget"
                    break
                if step % config.eval_every == 0 or step % config.save_every == 0:
                    validation = evaluate_loss(model, datasets["valid"], config.batch_size, config.eval_batches)
                    log.write(json.dumps({"event": "sft_validation", "step": step, "loss": validation}) + "\n")
                    save()
                    model.train()
            validation = evaluate_loss(model, datasets["valid"], config.batch_size, config.eval_batches)
            destination = save()
        summary = {"status": reason, "step": step, "validation_loss": validation,
                   "tokens_seen": supervised_tokens_seen, "supervised_tokens_seen": supervised_tokens_seen,
                   "input_tokens_seen": input_tokens_seen,
                   "best_validation_loss": best_loss, "training_mode": "instruction_sft",
                   "checkpoint": str(destination.relative_to(run_dir)),
                   "session_seconds": time.monotonic() - start,
                   "total_training_seconds": seconds_before + time.monotonic() - start, **memory_stats()}
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"event": "sft_finished", **summary}), flush=True)
        return summary
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
