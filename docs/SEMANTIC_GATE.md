# Semantic gate

This gate must pass before any throughput, GSM8K, or publication claim.

## Model

`nvidia/Qwen3.6-35B-A3B-NVFP4`

Revision: `491c2f1ea524c639598bf8fa787a93fed5a6fbce`

## Required probes

- `/v1/models` exposes exactly `Qwen3.6-35B-A3B-NVFP4`;
- deterministic arithmetic: `19 × 23 = 437`;
- exact-string reproduction;
- short word problem;
- short code-generation prompt;
- repeated generation with temperature 0;
- no empty output, punctuation loop, operand substitution, or template leakage.

## FP8 baseline

The first correctness run uses:

```text
quantization: modelopt_mixed
KV cache: fp8
attention: FlashInfer
MoE: native FlashInfer/CUTLASS
```

The baseline must pass all semantic probes before the NVFP4-KV candidate is attempted.

## NVFP4-KV candidate gate

The candidate must additionally pass:

1. 20/20 `max_tokens=1` and `max_tokens=2` prefill/decode probes;
2. nonzero, finite per-block FP4 scale factors in the paged cache;
3. 100% semantic probe accuracy;
4. GSM8K@100 within 2 percentage points of FP8;
5. a measured throughput or KV-capacity benefit without more than a 5% latency regression.

A failed NVFP4-KV candidate does not invalidate the FP8 baseline. It is recorded as a blocked/experimental path and is not used for the release profile.

## Quality protocol

GSM8K uses 0-shot greedy generation with flexible numeric extraction through `local-chat-completions`, `--apply_chat_template`, the local tokenizer, a 2,048-token generation budget, and `chat_template_kwargs={"enable_thinking": false}`. Strict `#### N` string matching is retained in raw evidence but is not the release metric.
