"""Export trained weights together with source, provenance, and applicable licenses."""

import json
import os
from pathlib import Path
import shutil
import tempfile

from silivra.checkpoint import resolve_checkpoint, sha256, verify_checkpoint, write_json


def export_model(run_dir: Path, output: Path, *, which="best", archive=False) -> dict:
    checkpoint = resolve_checkpoint(run_dir, which)
    state = verify_checkpoint(checkpoint)
    output = output.resolve()
    archive_path = Path(str(output) + ".zip")
    if output.exists():
        raise FileExistsError(f"Export already exists: {output}")
    if archive and archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")
    repo = Path(__file__).resolve().parents[2]
    if not (repo / "LICENSE").is_file() or not (repo / "pyproject.toml").is_file():
        raise RuntimeError("Export requires a source checkout. Install it using uv sync in the project folder.")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".silivra-export-", dir=output.parent))
    try:
        # Inference bundle: no optimizer state or raw/tokenized training stories.
        for name in ("model.safetensors", "tokenizer.json", "state.json", "data_manifest.json"):
            shutil.copy2(checkpoint / name, staging / name)
        run_root = checkpoint.parent.parent
        for name in ("run.json", "summary.json", "metrics.jsonl", "benchmark.json", "evaluation.json", "samples.json", "assistant-baseline.json", "assistant-evaluation.json"):
            record = run_root / name
            if record.is_file():
                dest = staging / "results" / name
                dest.parent.mkdir(exist_ok=True)
                if name.endswith(".json"):
                    value = json.loads(record.read_text())
                    value = _public_records(value)
                    write_json(dest, value)
                else:
                    shutil.copy2(record, dest)
        for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(repo / name, staging / name)
        shutil.copytree(repo / "licenses", staging / "licenses")
        source = staging / "source"
        files = [repo / name for name in ("pyproject.toml", "uv.lock", ".python-version", ".gitignore", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md")]
        for folder, pattern in (("src", "*.py"), ("tests", "*.py"), ("scripts", "*.py"), ("configs", "*.json"), ("docs", "*.md"), ("docs", "*.json"), ("licenses", "*.txt"), ("examples", "*.jsonl"), (".github", "*.yml")):
            files.extend(sorted((repo / folder).rglob(pattern)))
        for file in files:
            destination = source / file.relative_to(repo)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, destination)
        cfg = state["model_config"]
        dim, layers, intermediate = cfg["dim"], cfg["n_layers"], cfg["hidden_dim"]
        kv_dim = cfg["n_kv_heads"] * (dim // cfg["n_heads"])
        parameters = cfg["vocab_size"] * dim + layers * (2 * dim**2 + 2 * dim * kv_dim + 3 * dim * intermediate + 2 * dim) + dim
        write_json(staging / "export.json", {
            "format_version": 1, "kind": "inference bundle with complete project source",
            "model": "Silivra Tiny", "training_step": state["step"],
            "tokens_seen": state.get("tokens_seen"), "validation_loss": state["validation_loss"],
            "training_mode": state.get("training_mode", "pretraining"),
            "parameters": parameters, "initialization": state["initialization"],
            "weight_license": "MIT to the extent of project rights; see THIRD_PARTY_NOTICES.md",
            "raw_training_data_included": False, "optimizer_included": False,
        })
        command = ('uv run silivra chat --run .. --no-tools --memory-chars 0 "Say hello."'
                   if state.get("training_mode") == "instruction_sft"
                   else 'uv run silivra generate --run .. "Once upon a time"')
        (staging / "README.md").write_text(
            "# Silivra Tiny — experimental local model\n\n"
            "Original from-scratch Silivra lineage on Apple silicon. Training mode: "
            + state.get("training_mode", "pretraining") + ". "
            "Assistant capability is experimental and must be evaluated. See source/docs/MODEL_CARD.md, "
            "source/docs/CAPABILITY_RESULTS.md and source/docs/INITIAL_RESULTS.md.\n\n"
            "This bundle includes the complete source checkout under source/, FP32 model.safetensors, "
            "the original tokenizer, training settings, data provenance, licenses, and file hashes. "
            "Raw/tokenized stories and optimizer state are excluded.\n\n"
            "From the source/ folder, run `uv sync --locked`, then "
            + "`" + command + "`. "
            "After setup it works offline with `uv run --offline`.\n\n"
            "Code and model parameters: MIT to the extent of project rights. "
            "TinyStories data: CDLA-Sharing-1.0. Retain THIRD_PARTY_NOTICES.md and licenses/.\n",
            encoding="utf-8",
        )
        write_json(staging / "checksums.json", {
            str(file.relative_to(staging)): sha256(file)
            for file in sorted(staging.rglob("*")) if file.is_file()
        })
        os.replace(staging, output)
        result = {"directory": str(output), "training_step": state["step"],
                  "model_sha256": sha256(output / "model.safetensors")}
        if archive:
            path = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
            result.update(archive=path, archive_sha256=sha256(Path(path)))
        return result
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _public_records(value):
    """Omit local machine paths from optional exported result records."""
    if isinstance(value, dict):
        return {key: _public_records(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_public_records(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value
