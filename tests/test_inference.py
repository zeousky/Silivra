"""Completion behavior against uncached inference, including rolling windows."""

import unittest

import mlx.core as mx
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from silivra.config import ModelConfig
from silivra.inference import benchmark, complete
from silivra.model import SilivraLM


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        cls.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        cls.tokenizer.decoder = decoders.ByteLevel()
        cls.tokenizer.train_from_iterator(
            ["Once upon a time a little fox lived in a quiet forest.",
             "The fox found a bright red apple and shared it with a friend."],
            trainers.BpeTrainer(vocab_size=320, show_progress=False,
                                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"]),
        )

    def setUp(self):
        mx.random.seed(5)
        self.model = SilivraLM(ModelConfig(
            vocab_size=self.tokenizer.get_vocab_size(), dim=32, n_layers=2,
            n_heads=4, n_kv_heads=2, hidden_dim=64, max_seq_len=8,
        ))
        mx.eval(self.model.parameters())

    def uncached_greedy(self, prompt, count):
        tokenizer = self.tokenizer
        tokens = [tokenizer.token_to_id("<bos>")]
        tokens += tokenizer.encode(prompt, add_special_tokens=False).ids
        forbidden = [tokenizer.token_to_id(name)
                     for name in ("<bos>", "<pad>", "<unk>", "<eos>")]
        generated = []
        self.model.eval()
        for _ in range(count):
            scores = self.model(mx.array([tokens[-self.model.config.max_seq_len:]]))[0, -1]
            for token in forbidden:
                scores = scores.at[token].add(-float("inf"))
            selected = int(mx.argmax(scores).item())
            generated.append(selected)
            tokens.append(selected)
        return tokenizer.decode(generated, skip_special_tokens=True)

    def test_cached_completion_matches_full_forward_across_window_rollover(self):
        for prompt in ("", "a fox", "Once upon a time " * 6):
            with self.subTest(prompt=prompt):
                expected = self.uncached_greedy(prompt, 12)
                result = complete(self.model, self.tokenizer, prompt,
                                  max_new_tokens=12, temperature=0, top_k=0,
                                  ignore_eos=True)
                self.assertEqual(result["text"], expected)
                self.assertEqual(result["generated_tokens"], 12)
                self.assertLessEqual(result["prompt_tokens"], 8)
                initial_count = 1 + len(self.tokenizer.encode(prompt).ids)
                self.assertEqual(result["context_was_truncated"], initial_count > 8)

    def test_greedy_completion_is_independent_of_sampling_seed(self):
        first = complete(self.model, self.tokenizer, "a fox", max_new_tokens=6,
                         temperature=0, top_k=0, seed=1, ignore_eos=True)
        second = complete(self.model, self.tokenizer, "a fox", max_new_tokens=6,
                          temperature=0, top_k=0, seed=999, ignore_eos=True)
        self.assertEqual(first["text"], second["text"])
        self.assertEqual(first["generated_tokens"], second["generated_tokens"])

    def test_sampled_completion_repeats_with_same_seed(self):
        options = dict(max_new_tokens=10, temperature=0.8, top_k=8,
                       seed=17, ignore_eos=True)
        first = complete(self.model, self.tokenizer, "a fox", **options)
        second = complete(self.model, self.tokenizer, "a fox", **options)
        self.assertEqual(first["text"], second["text"])

    def test_single_token_timing_reports_overall_throughput(self):
        result = complete(self.model, self.tokenizer, "a fox", max_new_tokens=1,
                          temperature=0, top_k=0, ignore_eos=True)
        self.assertEqual(result["generated_tokens"], 1)
        self.assertGreater(result["total_seconds"], 0)
        self.assertAlmostEqual(result["tokens_per_second"], 1 / result["total_seconds"])
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertGreaterEqual(result["time_to_first_token_seconds"], result["prefill_seconds"])

    def test_invalid_completion_arguments_are_rejected(self):
        for options in (
            {"max_new_tokens": 0}, {"max_new_tokens": True}, {"max_new_tokens": 4097},
            {"temperature": -0.1}, {"temperature": float("nan")},
            {"temperature": float("inf")}, {"top_k": -1}, {"top_k": True},
            {"top_k": self.model.config.vocab_size + 1},
            {"seed": -1}, {"seed": 2 ** 64}, {"seed": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                complete(self.model, self.tokenizer, "a fox", **options)

    def test_benchmark_requires_enough_tokens_for_decode_measurement(self):
        with self.assertRaises(ValueError):
            benchmark(self.model, self.tokenizer, trials=1, max_new_tokens=1)


if __name__ == "__main__":
    unittest.main()
