"""Non-pickle checkpoints, atomically published with their complete training state."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_checkpoint(path: Path, which: str = "latest") -> Path:
    path = path.resolve()
    if (path / "model.safetensors").is_file():
        return path
    reference = path / f"{which}.json"
    if not reference.is_file():
        raise FileNotFoundError(f"No {which} checkpoint at {path}. Run silivra train first.")
    candidate = (path / json.loads(reference.read_text())["checkpoint"]).resolve()
    if not candidate.is_relative_to(path):
        raise ValueError("Checkpoint pointer escapes the run directory.")
    if not (candidate / "model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete checkpoint: {candidate}")
    return candidate


def save_checkpoint(run_dir, model, optimizer, model_config, training_config,
                    step, rng, data_dir, best_loss, validation_loss, total_training_seconds):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    parent = run_dir / "checkpoints"
    parent.mkdir(exist_ok=True)
    destination = parent / f"step-{step:07d}"
    if destination.exists():
        return destination
    temporary = Path(tempfile.mkdtemp(prefix=".saving-", dir=parent))
    try:
        model.save_weights(str(temporary / "model.safetensors"))
        mx.savez(str(temporary / "optimizer.npz"), **dict(tree_flatten(optimizer.state)))
        mx.savez(str(temporary / "mlx_rng.npz"), **{str(i): x for i, x in enumerate(mx.random.state)})
        shutil.copy2(data_dir / "tokenizer.json", temporary / "tokenizer.json")
        shutil.copy2(data_dir / "manifest.json", temporary / "data_manifest.json")
        state = {
            "format_version": 1,
            "step": step,
            "tokens_seen": step * training_config["batch_size"] * training_config["seq_len"],
            "model_config": model_config.to_dict(),
            "training_config": training_config,
            "numpy_rng": rng.bit_generator.state,
            "data_manifest_sha256": sha256(data_dir / "manifest.json"),
            "best_validation_loss": best_loss,
            "validation_loss": validation_loss,
            "training_seconds": total_training_seconds,
            "initialization": "random; no pretrained weights or tokenizer",
        }
        write_json(temporary / "state.json", state)
        write_json(temporary / "checksums.json", {
            file.name: sha256(file) for file in temporary.iterdir() if file.is_file()
        })
        os.replace(temporary, destination)
        write_json(run_dir / "latest.json", {"checkpoint": str(destination.relative_to(run_dir)), "step": step})
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_checkpoint(path: Path) -> dict:
    for name, expected in json.loads((path / "checksums.json").read_text()).items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()) or sha256(target) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return json.loads((path / "state.json").read_text())


def load_model(path: Path, which: str = "latest"):
    import mlx.core as mx
    from silivra.config import ModelConfig
    from silivra.model import SilivraLM
    from silivra.data import load_tokenizer

    path = resolve_checkpoint(path, which)
    state = verify_checkpoint(path)
    model = SilivraLM(ModelConfig.from_dict(state["model_config"]))
    model.load_weights(str(path / "model.safetensors"))
    model.eval()
    mx.eval(model.parameters())
    return model, load_tokenizer(path / "tokenizer.json"), state, path
