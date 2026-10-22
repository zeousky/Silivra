"""Architecture configuration for models initialized and trained by Silivra."""

from dataclasses import asdict, dataclass, fields
import math
from collections.abc import Mapping
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Small decoder architecture sized conservatively for 16 GB Macs.

    The configuration defines the architecture, not learned weights. Constructing
    a model from this configuration always starts with random parameters.
    """

    vocab_size: int = 4096
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 6
    n_kv_heads: int = 2
    hidden_dim: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        for name in (
            "vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads",
            "hidden_dim", "max_seq_len",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("dim / n_heads must be even for rotary positions")
        if (
            type(self.rope_theta) not in (int, float)
            or not math.isfinite(self.rope_theta)
            or self.rope_theta <= 0
        ):
            raise ValueError("rope_theta must be a finite positive number")
        object.__setattr__(self, "rope_theta", float(self.rope_theta))

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible architecture fields."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Load fields without silently accepting misspelled configuration keys."""
        if not isinstance(values, Mapping):
            raise ValueError("Model configuration must be a JSON object")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown model configuration fields: {sorted(unknown, key=str)}")
        return cls(**dict(values))
