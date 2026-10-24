"""Portable grammar behavior; constrained syntax is not learned task competence."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from silivra.agent import parse_action
from silivra.structured import (MAX_VOCABULARY_SIZE, action_shape, make_action_processor,
                               _byte_decoder, _vocabulary)
from silivra.tools import ToolRegistry


class ByteTokenizer:
    eos_token_id = 256
    all_special_ids = [256, 257]
    eos_token_ids = {256}
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{"decoder":{"type":"ByteLevel"}}')

    def get_vocab(self):
        vocabulary = {character: byte for character, byte in _byte_decoder().items()}
        vocabulary.update({"<eos>": 256, "<system>": 257})
        return vocabulary


class StructuredTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTokenizer()
        self.schemas = ToolRegistry(Path.cwd(), allow_network=True, allow_write=True,
                                    allow_exec=True).schemas()
        self.read = next(tool for tool in self.schemas if tool["function"]["name"] == "read_file")

    def accepts(self, raw, tools=()):
        processor = make_action_processor(self.tokenizer, list(tools))
        history = [10000, 10001]  # Prompt IDs are not interpreted as output JSON.
        for token in raw.encode("utf-8"):
            if token not in processor.allowed_tokens(history):
                return False
            history.append(token)
        return processor.allowed_tokens(history) == [self.tokenizer.eos_token_id]

    def test_final_string_including_unicode_and_code_newlines(self):
        for answer in ("299", "[-3,0,8,12]", 'def f():\n    return "hello"', "Café 👋", "\\quoted\\"):
            with self.subTest(answer=answer):
                raw = json.dumps({"final": answer}, ensure_ascii=False, separators=(",", ":"))
                self.assertTrue(self.accepts(raw))
                self.assertEqual(parse_action(raw)["final"], answer)

    def test_nonstring_finals_prose_duplicate_keys_and_multiple_actions_fail(self):
        for raw in ('{"final":299}', '{"final":[1,2]}', '{"final":true}',
                    '{"final":""}', '```json\n{"final":"x"}\n```',
                    '{"final":"x","final":"y"}', '{"final":"x"} trailing',
                    '{"final":"x"}{"tool":"read_file","arguments":{"path":"a"}}'):
            with self.subTest(raw=raw):
                self.assertFalse(self.accepts(raw))

    def test_tools_are_exposed_only_when_enabled_and_arguments_are_typed(self):
        valid = '{"tool":"read_file","arguments":{"path":"README.md","max_lines":10}}'
        self.assertFalse(self.accepts(valid))
        self.assertTrue(self.accepts(valid, [self.read]))
        for raw in ('{"tool":"run_command","arguments":{"argv":["ls"]}}',
                    '{"tool":"read_file","arguments":{}}',
                    '{"tool":"read_file","arguments":{"path":3}}',
                    '{"tool":"read_file","arguments":{"path":"a","max_lines":"10"}}',
                    '{"tool":"read_file","arguments":{"path":"a","surprise":true}}'):
            with self.subTest(raw=raw):
                self.assertFalse(self.accepts(raw, [self.read]))

    def test_semantic_constraints_are_explicitly_deferred_to_runtime(self):
        processor = make_action_processor(self.tokenizer, self.schemas)
        constraints = {(item["path"], item["constraint"]) for item in processor.deferred_constraints}
        self.assertIn(("read_file.arguments.max_lines", "maximum"), constraints)
        self.assertIn(("write_file.arguments.expected_sha256", "pattern"), constraints)
        self.assertIn(("web_search.arguments.query", "maxLength"), constraints)
        self.assertIn(("run_command.arguments.argv", "maxItems"), constraints)
        # This is intentionally a shape grammar: invalid ranges are rejected by
        # ToolRegistry rather than compiling massive string-length automata.
        self.assertTrue(self.accepts('{"tool":"read_file","arguments":{"path":"a","max_lines":-1}}', [self.read]))
        result = ToolRegistry(Path.cwd()).execute("read_file", {"path": "a", "max_lines": -1})
        self.assertFalse(result["ok"])

    def test_unknown_schema_features_fail_visibly(self):
        for parameters in ({"oneOf": [{"type": "object"}]},
                           {"type": "object", "properties": {}, "additionalProperties": True},
                           {"type": "object", "properties": {}, "required": ["missing"], "additionalProperties": False}):
            tool = {"type": "function", "function": {"name": "demo", "parameters": parameters}}
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                action_shape([tool])
        with self.assertRaises(ValueError):
            action_shape([self.read, self.read])

    def test_prompt_ignored_then_only_new_tokens_advance_and_eos_is_terminal(self):
        processor = make_action_processor(self.tokenizer, [])
        history = [10000] * 200
        self.assertNotIn(self.tokenizer.eos_token_id, processor.allowed_tokens(history))
        for token in b'{"final":"ok"}':
            self.assertIn(token, processor.allowed_tokens(history))
            history.append(token)
        self.assertEqual(processor.allowed_tokens(history), [256])
        self.assertEqual(processor.allowed_tokens(history), [256])
        self.assertEqual(processor.allowed_tokens(history + [256]), [256])
        with self.assertRaisesRegex(ValueError, "violated"):
            processor.allowed_tokens(history + [256, ord("{")])
        incomplete = make_action_processor(self.tokenizer, [])
        incomplete.allowed_tokens([10000])
        with self.assertRaisesRegex(ValueError, "EOS"):
            incomplete.allowed_tokens([10000, 256])

    def test_compiled_indexes_reused_but_generation_states_are_independent(self):
        one = make_action_processor(self.tokenizer, [])
        two = make_action_processor(self.tokenizer, [])
        self.assertIs(one.index, two.index)
        one.allowed_tokens([10000])
        one.allowed_tokens([10000, ord("{")])
        self.assertNotEqual(one.state, two.state)
        self.assertIn(ord("{"), two.allowed_tokens([10000]))
        with self.assertRaisesRegex(ValueError, "reused"):
            one.allowed_tokens([])
        with self.assertRaisesRegex(ValueError, "violated"):
            two.allowed_tokens([10000, ord("x")])

    def test_masks_scores_without_copying_full_logits_to_cpu(self):
        processor = make_action_processor(self.tokenizer, [])
        logits = np.arange(258, dtype=np.float32)[None, :]
        fake_mlx = SimpleNamespace(array=np.array, where=np.where)
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=fake_mlx), "mlx.core": fake_mlx}):
            masked = processor(np.array([10000]), logits)
            self.assertTrue(np.isfinite(masked[0, ord("{")]))
            self.assertTrue(np.isneginf(masked[0, ord("`")] ))
            self.assertTrue(np.isneginf(masked[0, 256]))
            self.assertTrue(np.isneginf(masked[0, 257]))
            cache_count = len(processor._masks)
            processor(np.array([10000]), logits)
            self.assertEqual(len(processor._masks), cache_count)
            with self.assertRaisesRegex(ValueError, "one local generation"):
                processor(np.array([10000]), logits[0])

    def test_unsupported_tokenizers_fail_without_downloading(self):
        self.tokenizer.backend_tokenizer = SimpleNamespace(to_str=lambda: '{"decoder":{"type":"Metaspace"}}')
        with self.assertRaisesRegex(ValueError, "ByteLevel"):
            make_action_processor(self.tokenizer, [])

    def test_qwen35_sized_byte_vocabulary_fits_explicit_bounded_profile(self):
        class LargerByteTokenizer(ByteTokenizer):
            def get_vocab(self):
                return {f"token{index}": index for index in range(248320)}

        with patch("outlines_core.Vocabulary", return_value=object()) as vocabulary:
            _vocabulary(LargerByteTokenizer())
            mapping = vocabulary.call_args.args[1]
            self.assertGreater(len(mapping), 200_000)
            self.assertLess(len(mapping), MAX_VOCABULARY_SIZE)
            self.assertNotIn(256, [index for indexes in mapping.values() for index in indexes])
        with patch("silivra.structured.MAX_VOCABULARY_SIZE", 255), self.assertRaisesRegex(ValueError, "exceeds"):
            _vocabulary(ByteTokenizer())


if __name__ == "__main__":
    unittest.main()
