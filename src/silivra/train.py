"""Bounded, resumable from-scratch pretraining on the local Apple GPU."""

import fcntl
from functools import partial
import json
import math
import signal
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from silivra.checkpoint import resolve_checkpoint, save_checkpoint, sha256, verify_checkpoint, write_json
from silivra.runtime import memory_stats, require_apple_gpu


@dataclass
class TrainConfig:
    steps: int = 2500
    batch_size: int = 16
    seq_len: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.1
    warmup_steps: int = 100
    eval_every: int = 250
    eval_batches: int = 12
    save_every: int = 500
    log_every: int = 25
    seed: int = 42
    memory_gib: float = 6.0

    def validate(self):
        for name in ("steps", "batch_size", "seq_len", "eval_every", "eval_batches", "save_every", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.seq_len > 512 or self.batch_size > 32:
            raise ValueError("The 16 GB profile supports seq_len <= 512 and batch_size <= 32.")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate <= 0.01:
            raise ValueError("learning_rate must be finite and in (0, 0.01].")
        if not math.isfinite(self.weight_decay) or not 0 <= self.weight_decay <= 1:
            raise ValueError("weight_decay must be finite and in [0, 1].")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be nonnegative and less than steps.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64).")


def evaluate_loss(model, dataset, batch_size, batches, seed=12345):
    import mlx.core as mx
    import mlx.nn as nn
    rng = np.random.default_rng(seed)
    model.eval()
    values = []
    for _ in range(batches):
        x, y = dataset.batch(rng, batch_size)
        loss = nn.losses.cross_entropy(model(mx.array(x)), mx.array(y), reduction="mean")
        values.append(float(loss.item()))
    value = sum(values) / len(values)
    if not math.isfinite(value):
        raise RuntimeError("Non-finite validation loss; checkpoint is not publishable.")
    return value


def train(data_dir: Path, run_dir: Path, config: TrainConfig, *, resume=False,
          max_minutes: float = 15.0, model_config=None) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten
    from silivra.config import ModelConfig
    from silivra.data import TokenDataset
    from silivra.model import SilivraLM

    config.validate()
    if not math.isfinite(max_minutes) or max_minutes <= 0:
        raise ValueError("max_minutes must be a finite positive number.")
    hardware = require_apple_gpu(config.memory_gib)
    data_dir, run_dir = data_dir.resolve(), run_dir.resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    # Reject corrupted or accidentally mixed prepared artifacts before training.
    for record in [manifest["tokenizer"], *manifest["splits"].values()]:
        target = (data_dir / record["path"]).resolve()
        if not target.is_relative_to(data_dir) or sha256(target) != record["sha256"]:
            raise ValueError(f"Prepared data checksum mismatch: {record['path']}")
    train_data = TokenDataset(data_dir / "train.npy", config.seq_len)
    valid_data = TokenDataset(data_dir / "valid.npy", config.seq_len)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another trainer is already using this run directory.") from None
    stopped = {"requested": False}
    previous_handlers = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stopped.update(requested=True))
    try:
        checkpoint = None
        if resume:
            checkpoint = resolve_checkpoint(run_dir)
            previous = verify_checkpoint(checkpoint)
            if previous["training_config"] != asdict(config):
                raise ValueError("Resume requires the original training configuration (including total steps).")
            if previous["data_manifest_sha256"] != sha256(data_dir / "manifest.json"):
                raise ValueError("Resume dataset does not match the checkpoint.")
            model_config = ModelConfig.from_dict(previous["model_config"])
        elif (run_dir / "latest.json").exists() or (run_dir / "run.json").exists():
            raise FileExistsError("Run already exists. Use --resume or a different --run directory.")
        model_config = model_config or ModelConfig(vocab_size=manifest["tokenizer"]["vocab_size"])
        if model_config.vocab_size != manifest["tokenizer"]["vocab_size"] or model_config.max_seq_len < config.seq_len:
            raise ValueError("Model configuration does not match data vocabulary or sequence length.")
        mx.random.seed(config.seed)
        rng = np.random.default_rng(config.seed)
        model = SilivraLM(model_config)
        if config.warmup_steps:
            schedule = optim.join_schedules([
                optim.linear_schedule(0.0, config.learning_rate, config.warmup_steps),
                optim.cosine_decay(config.learning_rate, config.steps - config.warmup_steps,
                                   end=config.learning_rate * 0.1)
            ], [config.warmup_steps])
        else:
            schedule = optim.cosine_decay(config.learning_rate, config.steps, end=config.learning_rate * 0.1)
        optimizer = optim.AdamW(learning_rate=schedule, weight_decay=config.weight_decay)
        start_step, best_loss, elapsed_before = 0, float("inf"), 0.0
        if checkpoint:
            model.load_weights(str(checkpoint / "model.safetensors"))
            optimizer.state = tree_unflatten(list(mx.load(str(checkpoint / "optimizer.npz")).items()))
            rng.bit_generator.state = previous["numpy_rng"]
            for i, array in mx.load(str(checkpoint / "mlx_rng.npz")).items():
                mx.random.state[int(i)][:] = array
            start_step = previous["step"]
            best_loss = previous["best_validation_loss"]
            elapsed_before = previous["training_seconds"]
        mx.eval(model.parameters(), optimizer.state)
        baseline = evaluate_loss(model, valid_data, config.batch_size, config.eval_batches)
        if not resume:
            write_json(run_dir / "run.json", {
                "format_version": 1, "hardware": hardware, "model_config": model_config.to_dict(),
                "training_config": asdict(config), "parameters": model.num_parameters,
                "baseline_validation_loss": baseline, "data_manifest_sha256": sha256(data_dir / "manifest.json"),
                "initialization": "random; no pretrained weights or tokenizer",
            })
        print(json.dumps({"event": "start", "step": start_step, "parameters": model.num_parameters,
                          "validation_loss": baseline, "max_minutes": max_minutes}), flush=True)

        def loss_fn(model, x, y):
            return nn.losses.cross_entropy(model(x), y, reduction="mean")

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        state = [model.state, optimizer.state, mx.random.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step_fn(x, y):
            loss, grads = loss_and_grad(model, x, y)
            grads, norm = optim.clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(model, grads)
            return loss, norm

        model.train()
        start = last_log = time.monotonic()
        last_step, last_logged_step = start_step, start_step
        validation_loss = baseline
        reason = "completed"
        log_path = run_dir / "metrics.jsonl"

        def save(step, validation):
            nonlocal best_loss
            improved = validation < best_loss
            best_loss = min(best_loss, validation)
            dest = save_checkpoint(run_dir, model, optimizer, model_config, asdict(config), step,
                                   rng, data_dir, best_loss, validation, elapsed_before + time.monotonic() - start)
            if improved:
                write_json(run_dir / "best.json", {"checkpoint": str(dest.relative_to(run_dir)), "step": step})
            return dest

        with log_path.open("a", buffering=1) as log:
            for step in range(start_step + 1, config.steps + 1):
                if stopped["requested"] or time.monotonic() - start >= max_minutes * 60:
                    reason = "interrupted" if stopped["requested"] else "time_budget"
                    break
                x, y = train_data.batch(rng, config.batch_size)
                loss, norm = step_fn(mx.array(x), mx.array(y))
                mx.eval(loss, norm, state)
                loss_value, norm_value = loss.item(), norm.item()
                if not math.isfinite(loss_value) or not math.isfinite(norm_value):
                    raise RuntimeError("Non-finite training loss or gradient; resume the last good checkpoint.")
                last_step = step
                if mx.get_peak_memory() > config.memory_gib * 1024**3:
                    reason = "memory_budget"
                    break
                if step % config.log_every == 0 or step == config.steps:
                    now = time.monotonic()
                    row = {"event": "train", "step": step, "loss": loss_value,
                           "gradient_norm": norm_value, "learning_rate": float(optimizer.learning_rate.item()),
                           "tokens_per_second": (step - last_logged_step) * config.batch_size * config.seq_len / (now - last_log),
                           "tokens_seen": step * config.batch_size * config.seq_len, **memory_stats()}
                    line = json.dumps(row)
                    print(line, flush=True)
                    log.write(line + "\n")
                    last_log, last_logged_step = now, step
                if step % config.eval_every == 0 or step % config.save_every == 0:
                    validation_loss = evaluate_loss(model, valid_data, config.batch_size, config.eval_batches)
                    row = {"event": "validation", "step": step, "loss": validation_loss,
                           "perplexity": math.exp(min(validation_loss, 80))}
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + "\n")
                    # Every measured improvement is durable, along with optimizer + RNG.
                    save(step, validation_loss)
                    model.train()
                    last_log = time.monotonic()
                    last_logged_step = step
            validation_loss = evaluate_loss(model, valid_data, config.batch_size, config.eval_batches)
            dest = save(last_step, validation_loss)
        summary = {
            "status": reason, "step": last_step,
            "tokens_seen": last_step * config.batch_size * config.seq_len,
            "validation_loss": validation_loss, "best_validation_loss": best_loss,
            "perplexity": math.exp(min(validation_loss, 80)),
            "session_seconds": time.monotonic() - start,
            "total_training_seconds": elapsed_before + time.monotonic() - start,
            "checkpoint": str(dest.relative_to(run_dir)), **memory_stats(),
        }
        write_json(run_dir / "summary.json", summary)
        print(json.dumps({"event": "finished", **summary}), flush=True)
        return summary
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
