"""Offline completion and reproducible Metal speed measurements."""

import math
import time

from silivra.runtime import memory_stats


def complete(model, tokenizer, prompt: str, *, max_new_tokens=128,
             temperature=0.7, top_k=40, seed=42, ignore_eos=False,
             on_text=None, cancelled=None):
    """Decode once, optionally reporting Unicode-safe text deltas.

    Cancellation is checked before prefill and between generated tokens. A
    running Metal operation is allowed to finish; the returned ``cancelled``
    flag identifies a partial completion rather than an accepted answer.
    """
    import mlx.core as mx
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 4096:
        raise ValueError("max_new_tokens must be in [1, 4096].")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative.")
    if type(top_k) is not int or not 0 <= top_k <= model.config.vocab_size:
        raise ValueError("top_k must be between zero and the vocabulary size.")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64).")
    if on_text is not None and not callable(on_text):
        raise ValueError("on_text must be callable or None.")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be callable or None.")
    mx.random.seed(seed)
    model.eval()
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("Checkpoint tokenizer is missing BOS/EOS tokens.")
    tokens = [bos] + tokenizer.encode(prompt, add_special_tokens=False).ids
    limit = model.config.max_seq_len
    context = tokens[-limit:]
    cache = model.make_cache()
    start = time.perf_counter()
    was_cancelled = bool(cancelled and cancelled())
    if not was_cancelled:
        logits = model(mx.array([context]), cache=cache)[:, -1, :]
        mx.eval(logits)
    prefill_seconds = time.perf_counter() - start
    generated = []
    decode_start = time.perf_counter()
    first_token_seconds = None
    first_token_at = None
    stream = None
    streamed_text = ""
    if on_text is not None:
        from tokenizers.decoders import DecodeStream
        stream = DecodeStream(skip_special_tokens=True)
    for i in range(max_new_tokens):
        if was_cancelled or (cancelled and cancelled()):
            was_cancelled = True
            break
        # BOS, PAD and UNK have no role in a completed story.
        scores = logits[0]
        for name in ("<bos>", "<pad>", "<unk>"):
            token_id = tokenizer.token_to_id(name)
            if token_id is not None:
                scores = scores.at[token_id].add(-float("inf"))
        if ignore_eos:
            scores = scores.at[eos].add(-float("inf"))
        if temperature == 0:
            token = int(mx.argmax(scores).item())
        else:
            scores = scores / temperature
            if top_k:
                threshold = mx.sort(scores)[-top_k]
                scores = mx.where(scores >= threshold, scores, -float("inf"))
            token = int(mx.random.categorical(scores).item())
        if first_token_seconds is None:
            first_token_at = time.perf_counter()
            first_token_seconds = first_token_at - start
        if token == eos and not ignore_eos:
            break
        generated.append(token)
        tokens.append(token)
        if stream is not None:
            # DecodeStream withholds incomplete UTF-8 byte sequences. Decoding
            # each token independently would corrupt emoji/non-ASCII output.
            delta = stream.step(tokenizer, token)
            if delta:
                streamed_text += delta
                on_text(delta)
        if cancelled and cancelled():
            was_cancelled = True
            break
        if i + 1 == max_new_tokens:
            break
        if cache[0].offset >= limit:
            # Recompute the trailing window, retaining valid absolute positions.
            cache = model.make_cache()
            logits = model(mx.array([tokens[-limit:]]), cache=cache)[:, -1, :]
        else:
            logits = model(mx.array([[token]]), cache=cache)[:, -1, :]
        mx.eval(logits)
    decode_seconds = time.perf_counter() - decode_start
    total_seconds = time.perf_counter() - start
    cached_decode_seconds = time.perf_counter() - first_token_at if first_token_at is not None else 0.0
    text = tokenizer.decode(generated, skip_special_tokens=True)
    if stream is not None:
        if not text.startswith(streamed_text):
            raise RuntimeError("Tokenizer streaming decoder did not preserve the decoded text prefix.")
        # An unfinished byte sequence at the generation limit is resolved by
        # the tokenizer's ordinary final decode, exactly matching returned text.
        if len(text) > len(streamed_text):
            on_text(text[len(streamed_text):])
    return {
        "text": text, "cancelled": was_cancelled,
        "prompt_tokens": len(context), "generated_tokens": len(generated),
        "prefill_seconds": prefill_seconds,
        "time_to_first_token_seconds": first_token_seconds,
        "generation_seconds": decode_seconds,
        "tokens_per_second": len(generated) / total_seconds,
        "decode_tokens_per_second": (len(generated) - 1) / cached_decode_seconds if len(generated) > 1 else None,
        "decode_seconds": cached_decode_seconds,
        "total_seconds": total_seconds,
        "context_window": limit, "context_was_truncated": len(tokens) - len(generated) > limit,
        **memory_stats(),
    }


def benchmark(model, tokenizer, *, trials=3, max_new_tokens=128):
    import mlx.core as mx
    import statistics
    if type(trials) is not int or not 1 <= trials <= 10:
        raise ValueError("trials must be in [1, 10].")
    if type(max_new_tokens) is not int or not 2 <= max_new_tokens <= 4096:
        raise ValueError("Benchmark max_new_tokens must be in [2, 4096].")
    prompt = "Once upon a time, a little fox lived beside a quiet forest. One morning, the fox found"
    complete(model, tokenizer, prompt, max_new_tokens=8, temperature=0)
    mx.synchronize()
    mx.reset_peak_memory()
    results = []
    for _ in range(trials):
        result = complete(model, tokenizer, prompt, max_new_tokens=max_new_tokens,
                          temperature=0, ignore_eos=True)
        del result["text"]
        results.append(result)
    return {
        "format_version": 1, "trials": results,
        "median_tokens_per_second": statistics.median(r["tokens_per_second"] for r in results),
        "median_decode_tokens_per_second": statistics.median(r["decode_tokens_per_second"] for r in results),
        "median_time_to_first_token_seconds": statistics.median(r["time_to_first_token_seconds"] for r in results),
        "sampling": "greedy; EOS suppressed for fixed-length timing; warm kernels; FP32 weights",
        "includes_model_load": False, **memory_stats(),
    }
