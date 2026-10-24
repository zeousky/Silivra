"""Constrain local decoding to the shape of one Silivra JSON action.

The grammar enforces JSON structure, allowed keys, required fields, argument
types and enabled tool names. It does not establish answer correctness or
authorize effects. ToolRegistry still validates semantic constraints (ranges,
lengths, patterns, array counts, paths and permissions) before execution.
``deferred_constraints`` records every schema constraint left to that validator.

Outlines-core supplies a small Rust token automaton. No model, remote tokenizer,
PyTorch or network is loaded here. Only ByteLevel tokenizers are supported;
their raw token bytes must be retained, including partial Unicode sequences.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import json
import re
import weakref


_CACHE = weakref.WeakKeyDictionary()
MAX_VOCABULARY_SIZE = 300_000
_SEMANTIC = frozenset({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                       "multipleOf", "minLength", "maxLength", "pattern", "format",
                       "minItems", "maxItems", "uniqueItems"})
_ANNOTATIONS = frozenset({"description", "title", "default", "examples", "$schema"})
_STRUCTURAL = frozenset({"type", "properties", "required", "additionalProperties", "items",
                         "enum", "const"})


def _shape(schema, path, deferred, *, depth=0, count=None):
    """Keep supported shape keywords and explicitly report semantic deferrals."""
    if count is None:
        count = [0]
    count[0] += 1
    if depth > 8 or count[0] > 256 or not isinstance(schema, dict):
        raise ValueError("Action schemas must be bounded JSON objects.")
    unknown = set(schema) - _STRUCTURAL - _SEMANTIC - _ANNOTATIONS
    if unknown:
        raise ValueError(f"Unsupported action schema keywords at {path}: {sorted(unknown)}")
    for key in sorted(set(schema) & _SEMANTIC):
        deferred.append({"path": path, "constraint": key, "value": schema[key]})
    kind = schema.get("type")
    if kind not in {"object", "array", "string", "integer", "number", "boolean", "null"}:
        raise ValueError(f"Action schema needs a supported explicit type at {path}.")
    result = {"type": kind}
    for key in ("const", "enum"):
        if key in schema:
            value = schema[key]
            if key == "enum" and (not isinstance(value, list) or not 1 <= len(value) <= 64):
                raise ValueError("Action schema enums must contain 1–64 values.")
            result[key] = value
    if kind == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if (not isinstance(properties, dict) or len(properties) > 32
                or any(not isinstance(name, str) or not name or len(name) > 100 for name in properties)
                or not isinstance(required, list) or any(not isinstance(name, str) for name in required)
                or len(required) != len(set(required)) or set(required) - properties.keys()
                or schema.get("additionalProperties") is not False):
            raise ValueError(f"Action objects need closed properties and valid required fields at {path}.")
        result.update(properties={
            name: _shape(value, f"{path}.{name}", deferred, depth=depth + 1, count=count)
            for name, value in properties.items()}, required=required, additionalProperties=False)
    elif kind == "array":
        result["items"] = _shape(schema.get("items"), f"{path}[]", deferred,
                                  depth=depth + 1, count=count)
    return result


def action_shape(tools: list[dict]) -> tuple[dict, tuple[dict, ...]]:
    """Build a final-or-enabled-tool grammar and report runtime-only constraints."""
    if not isinstance(tools, list) or len(tools) > 32:
        raise ValueError("Action grammar needs a list of at most 32 enabled tools.")
    actions = [{"type": "object", "properties": {"final": {"type": "string", "minLength": 1}},
                "required": ["final"], "additionalProperties": False}]
    deferred, names = [], set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name)
                or name in names):
            raise ValueError("Action tools need distinct valid names.")
        names.add(name)
        arguments = _shape(function.get("parameters"), f"{name}.arguments", deferred)
        if arguments["type"] != "object":
            raise ValueError("Tool arguments must be an object.")
        actions.append({"type": "object", "properties": {
            "tool": {"const": name}, "arguments": arguments},
            "required": ["tool", "arguments"], "additionalProperties": False})
    return {"anyOf": actions}, tuple(deferred)


def _byte_decoder() -> dict[str, int]:
    # GPT-2/ByteLevel's reversible 256-byte alphabet, also used by Qwen3.
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    result = {chr(byte): byte for byte in visible}
    next_character = 256
    for byte in range(256):
        if byte not in visible:
            result[chr(next_character)] = byte
            next_character += 1
    return result


@dataclass
class _TokenizerCache:
    vocabulary: object
    eos_token_id: int
    indexes: OrderedDict = field(default_factory=OrderedDict)


def _vocabulary(tokenizer) -> _TokenizerCache:
    from outlines_core import Vocabulary
    try:
        specification = json.loads(tokenizer.backend_tokenizer.to_str())
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("Structured decoding requires a local ByteLevel tokenizer.") from error
    if specification.get("decoder", {}).get("type") != "ByteLevel":
        raise ValueError("Structured decoding currently supports ByteLevel tokenizers only.")
    eos = getattr(tokenizer, "eos_token_id", None)
    if type(eos) is not int or eos < 0:
        raise ValueError("Structured decoding needs a single EOS token ID.")
    extra_eos = getattr(tokenizer, "eos_token_ids", ())
    if type(extra_eos) is int:
        extra_eos = (extra_eos,)
    excluded = set(tokenizer.all_special_ids) | set(extra_eos or ()) | {eos}
    decoder, tokens = _byte_decoder(), {}
    for text, token_id in tokenizer.get_vocab().items():
        if type(token_id) is not int or token_id < 0:
            raise ValueError("Tokenizer vocabulary contains an invalid token ID.")
        if token_id in excluded:
            continue
        try:
            raw = bytes(decoder[character] for character in text)
        except KeyError as error:
            raise ValueError("Tokenizer vocabulary does not use the ByteLevel alphabet.") from error
        if raw:
            tokens.setdefault(raw, []).append(token_id)
    if not tokens or len(tokens) > MAX_VOCABULARY_SIZE:
        raise ValueError("Tokenizer vocabulary is empty or exceeds the structured profile.")
    return _TokenizerCache(Vocabulary(eos, tokens), eos)


class ActionProcessor:
    """Fresh per-generation state, sharing an immutable compiled token index.

    MLX-LM calls this with the complete prompt on its first invocation and then
    with appended generated tokens. Only the appended suffix advances the JSON
    automaton. A processor must never be reused for a second generation.
    """

    def __init__(self, index, eos_token_id: int, deferred_constraints=()):
        self.index, self.eos_token_id = index, eos_token_id
        self.deferred_constraints = deferred_constraints
        self.state = index.get_initial_state()
        self._history_length = None
        self._masks = OrderedDict()
        self._allowed = OrderedDict()

    def allowed_tokens(self, history) -> list[int]:
        """Advance only new output tokens; this portable method does not import MLX."""
        size = len(history)
        if self._history_length is None:
            self._history_length = size
        elif size < self._history_length:
            raise ValueError("An action processor cannot be reused for a new generation.")
        else:
            suffix = history[self._history_length:]
            for token in suffix.tolist() if hasattr(suffix, "tolist") else suffix:
                if type(token) is not int:
                    raise ValueError("Invalid token in action history.")
                if token == self.eos_token_id:
                    if not self.index.is_final_state(self.state):
                        raise ValueError("Unexpected EOS before a complete JSON action.")
                    # MLX-LM pipelines one step ahead before stream_generate
                    # observes EOS. Keep an EOS-only sink for that lookahead;
                    # no text token can follow the completed action.
                    continue
                next_state = self.index.get_next_state(self.state, token)
                if next_state is None:
                    raise ValueError("Generated token violated the action grammar.")
                self.state = next_state
            self._history_length = size
        # A complete JSON object can be followed only by EOS. No whitespace,
        # prose, second object, or additional tool action is admitted.
        if self.index.is_final_state(self.state):
            return [self.eos_token_id]
        if self.state not in self._allowed:
            allowed = self.index.get_allowed_tokens(self.state)
            if not allowed:
                raise ValueError("Action grammar reached a state without valid tokens.")
            self._allowed[self.state] = allowed
            if len(self._allowed) > 32:
                self._allowed.popitem(last=False)
        self._allowed.move_to_end(self.state)
        return self._allowed[self.state]

    def __call__(self, history, logits):
        allowed = self.allowed_tokens(history)
        if len(logits.shape) != 2 or logits.shape[0] != 1:
            raise ValueError("Structured decoding supports one local generation at a time.")
        width = logits.shape[-1]
        key = (self.state, width)
        import mlx.core as mx
        if key not in self._masks:
            import numpy as np
            if any(token < 0 or token >= width for token in allowed):
                raise ValueError("Action grammar token IDs exceed model logits.")
            mask = np.zeros(width, dtype=np.bool_)
            mask[allowed] = True
            self._masks[key] = mx.array(mask)
            if len(self._masks) > 32:
                self._masks.popitem(last=False)
        self._masks.move_to_end(key)
        # Scores stay on the GPU. Only newly sampled token IDs cross to the CPU;
        # the full logits/vocabulary are never copied back for grammar checks.
        return mx.where(self._masks[key], logits, -float("inf"))


def make_action_processor(tokenizer, tools: list[dict]) -> ActionProcessor:
    """Compile/cache a shape grammar and return independent generation state."""
    from outlines_core import Index, json_schema
    schema, deferred = action_shape(tools)
    key = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    if len(key) > 64_000:
        raise ValueError("Action schema exceeds the structured profile.")
    cache = _CACHE.get(tokenizer)
    if cache is None:
        cache = _vocabulary(tokenizer)
        _CACHE[tokenizer] = cache
    if key not in cache.indexes:
        regex = json_schema.build_regex_from_schema(key, whitespace_pattern="")
        cache.indexes[key] = Index(regex, cache.vocabulary)
        if len(cache.indexes) > 8:
            cache.indexes.popitem(last=False)
    cache.indexes.move_to_end(key)
    return ActionProcessor(cache.indexes[key], cache.eos_token_id, deferred)
