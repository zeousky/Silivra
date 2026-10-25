"""Explicit, bounded download of a pinned external model; never called by chat.

Weights and tokenizer belong to Qwen/MLX Community, not Silivra's original model.
Every file has a reviewed immutable revision, expected size and SHA-256 digest.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import time
import urllib.request


DEFAULT_PROFILE = "qwen3.5-9b"
DEFAULT_MODEL_RUN = Path("runs/silivra-qwen3.5-9b")
MODEL_ID = "mlx-community/Qwen3-8B-4bit"
REVISION = "545dc4251c05440727734bcd94334791f6ab0192"
ORIGINAL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MANIFEST_NAME = "silivra-pretrained.json"
# Values measured from the immutable Hub revision; no remote Python is included.
_SPECS = {
    "README.md": (857, "a0cd248b30da7a23710c9290fe42ca83a84dc06593978877b1e5b32f9bd33aaf"),
    "added_tokens.json": (707, "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
    "config.json": (939, "e5485285fd7e289e76e9cffa112f6dc2e3426519082f7db9b69041589f81a218"),
    "merges.txt": (1671853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model.safetensors": (4607835174, "f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8"),
    "model.safetensors.index.json": (64065, "3fb25463b4078b1fc27159daa605190029c2e965f533bf0b1b594f96cbfceb8a"),
    "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "tokenizer.json": (11422654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (9706, "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0"),
    "vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "LICENSE": (11343, "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"),
}

_QWEN35_SPECS = {
    "README.md": (1954, "7cbe4c3e07a5465d52d60c94c71f8d578113550eb767d331553f9ca19538ce56"),
    "chat_template.jinja": (7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    "config.json": (3331, "a96942cb6a8a1d3f1d17514d81a1925d04362a6a3233b389d13012211baaa9f8"),
    "model-00001-of-00002.safetensors": (5349771222, "a68b87558c6ef43f74c2bd63ce7e9092ceddc3101f3def0030774bae5f42aadd"),
    "model-00002-of-00002.safetensors": (600449850, "b0a770bf8469c7f3f18756a0e0283f1c1174344a83e059a4e483f6af4907352d"),
    "model.safetensors.index.json": (123592, "dd023913fb87cfdae27fb11dcf695117c925833796ccac3c64117d6652d8ff1e"),
    "preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    "processor_config.json": (1300, "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b"),
    "tokenizer.json": (19989343, "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"),
    "tokenizer_config.json": (1139, "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182"),
    "video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    "vocab.json": (6722759, "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"),
    "LICENSE": (11544, "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a"),
}

PROFILES = {
    "qwen3-8b": {"model_id": MODEL_ID, "revision": REVISION,
        "original_model_id": "Qwen/Qwen3-8B", "original_model_revision": ORIGINAL_REVISION,
        "parameter_count": 8190735360, "run": Path("runs/silivra-qwen3-8b"),
        "download_limit_bytes": 5 * 1024**3},
    "qwen3.5-9b": {"model_id": "mlx-community/Qwen3.5-9B-MLX-4bit",
        "revision": "938d8919941c6e7efd3c7150eff7fe9d12afa631",
        "original_model_id": "Qwen/Qwen3.5-9B",
        "original_model_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "parameter_count": 8953803264, "run": DEFAULT_MODEL_RUN,
        "download_limit_bytes": 6 * 1024**3},
}


def model_profile(name: str) -> dict:
    """Return an explicit pinned profile without performing network operations."""
    if not isinstance(name, str) or name not in PROFILES:
        raise ValueError("Unknown pretrained model profile.")
    return dict(PROFILES[name])


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_size(path: Path) -> int:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("Model downloads require regular files without links.")
    return info.st_size


def _download(url: str, target: Path, size: int, expected_hash: str, deadline: float) -> None:
    """Resume only bounded bytes; verify the complete file before promoting it."""
    if target.exists() or target.is_symlink():
        if _regular_size(target) != size or _digest(target) != expected_hash:
            raise ValueError(f"Existing {target.name} differs from the pinned model; preserved.")
        return
    partial = target.with_name(target.name + ".partial")
    offset = _regular_size(partial) if partial.exists() or partial.is_symlink() else 0
    if offset > size:
        raise ValueError(f"Partial {target.name} exceeds the pinned file size.")
    if offset < size:
        if time.monotonic() >= deadline:
            raise RuntimeError("Model setup reached its time budget; rerun to resume.")
        headers = {"User-Agent": "Silivra-model-setup/1", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            status = response.status
            if offset and status == 206:
                expected_range = f"bytes {offset}-{size - 1}/{size}"
                if response.headers.get("Content-Range") != expected_range:
                    raise ValueError("Download returned an unexpected resume range.")
            elif status == 200:
                offset = 0  # A server may ignore Range; restart this partial file.
            else:
                raise ValueError("Download did not return the requested model bytes.")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) != size - offset:
                raise ValueError("Download size differs from the pinned file size.")
            flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
            flags |= os.O_APPEND if offset else os.O_TRUNC
            descriptor = os.open(partial, flags, 0o600)
            with os.fdopen(descriptor, "wb") as output:
                info = os.fstat(output.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("Partial model file must be a regular file without links.")
                transferred, next_report = offset, offset + 512 * 1024**2
                while True:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Model setup reached its time budget; rerun to resume.")
                    chunk = response.read(min(4 * 1024**2, size - transferred + 1))
                    if not chunk:
                        break
                    if transferred + len(chunk) > size:
                        raise ValueError("Download exceeded the pinned file size.")
                    output.write(chunk)
                    transferred += len(chunk)
                    if transferred >= next_report:
                        print(f"{target.name}: {transferred / 10**9:.1f}/{size / 10**9:.1f} GB", file=sys.stderr, flush=True)
                        next_report += 512 * 1024**2
                output.flush()
                os.fsync(output.fileno())
    if _regular_size(partial) != size:
        raise RuntimeError(f"Incomplete {target.name}; rerun setup to resume.")
    if _digest(partial) != expected_hash:
        raise ValueError(f"SHA-256 mismatch for {target.name}; partial file was preserved for inspection.")
    os.replace(partial, target)


def setup_pretrained(output: Path | None = None, *, profile=DEFAULT_PROFILE) -> dict:
    """Download one pinned <=6-GiB profile in a bounded 15-minute session."""
    selected = model_profile(profile)
    specs = _SPECS if profile == "qwen3-8b" else _QWEN35_SPECS
    output = selected["run"] if output is None else output
    output = Path(output).absolute()
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ValueError("Model directory and parents must not be symlinks.")
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError("Model output must be a directory.")
    allowed = set(specs) | {name + ".partial" for name in specs} | {MANIFEST_NAME, ".DS_Store"}
    if any(item.name not in allowed for item in output.iterdir()):
        raise ValueError("Model output contains unrelated artifacts; choose a separate empty directory.")
    marker = output / MANIFEST_NAME
    if marker.exists() or marker.is_symlink():
        if _regular_size(marker) > 131072:
            raise ValueError("Invalid existing pretrained manifest; preserved.")
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise ValueError("Invalid existing pretrained manifest; preserved.") from None
        if (not isinstance(existing, dict) or existing.get("model_id") != selected["model_id"]
                or existing.get("revision") != selected["revision"]):
            raise ValueError("Model output belongs to another profile; choose a separate directory.")
    total = sum(size for size, _ in specs.values())
    if total > selected["download_limit_bytes"]:
        raise ValueError("Pinned model exceeds setup's download budget.")
    remaining = 0
    verified = set()
    for name, (size, digest) in specs.items():
        target, partial = output / name, output / (name + ".partial")
        if target.exists() or target.is_symlink():
            if _regular_size(target) != size or _digest(target) != digest:
                raise ValueError(f"Existing {name} differs from the pinned model; preserved.")
            verified.add(name)
            continue
        downloaded = _regular_size(partial) if partial.exists() or partial.is_symlink() else 0
        if downloaded > size:
            raise ValueError(f"Partial {name} exceeds the pinned file size.")
        remaining += max(0, size - downloaded)
    if shutil.disk_usage(output).free < remaining + 512 * 1024**2:
        raise RuntimeError("Model setup needs download space plus 512 MiB of free headroom.")
    print(f"External pretrained model: {selected['model_id']}, {selected['parameter_count']:,} text parameters, 4-bit; {total / 10**9:.2f} GB maximum.", file=sys.stderr, flush=True)
    deadline = time.monotonic() + 15 * 60

    def fetch(name):
        if name in verified:
            return
        size, digest = specs[name]
        repo, revision = ((selected["original_model_id"], selected["original_model_revision"])
                          if name == "LICENSE" else (selected["model_id"], selected["revision"]))
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        _download(url, output / name, size, digest, deadline)

    from concurrent.futures import ThreadPoolExecutor
    # Each worker writes a distinct fixed filename. Two streams bound transient
    # buffers while allowing multi-shard models to transfer concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, specs))
    manifest = {"format_version": 1, "backend": "mlx-lm", "profile": profile,
                **{key: selected[key] for key in ("model_id", "revision", "original_model_id",
                                                   "original_model_revision", "parameter_count")},
                "license": "Apache-2.0", "quantization_bits": 4, "context_window": 8192,
                "inference_mode": "text-only",
                "files": {name: {"size_bytes": size, "sha256": digest} for name, (size, digest) in specs.items()}}
    import tempfile
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pretrained-manifest-", dir=output)
    temporary = Path(temporary_name)
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, marker)
    return {"profile": profile, "model_id": selected["model_id"], "revision": selected["revision"], "parameter_count": manifest["parameter_count"],
            "quantization_bits": 4, "download_bytes": total, "status": "verified", "local_only_inference": True}
