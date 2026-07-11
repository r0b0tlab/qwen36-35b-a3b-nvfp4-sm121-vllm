# Benchmark evidence

Canonical release run: `nvidia-qwen36-native-20260711T100947Z`

Curated public claims are generated from machine-readable artifacts under [`runs/nvidia-qwen36-native-20260711T100947Z/`](runs/nvidia-qwen36-native-20260711T100947Z/). Raw sample payloads, telemetry streams, local virtual environments, PID files, and transient logs are intentionally excluded from Git.

## Quality

- Task: GSM8K test split, all 1,319 samples.
- Setting: 0-shot, local chat-completions API, chat template applied, thinking disabled, greedy temperature 0, 2,048-token generation budget.
- Primary metric: `exact_match,flexible-extract`.
- Result: 85.52% ± 0.97%.
- Artifact: `gsm8k/results.json`.

## Throughput and latency

- Tool: vLLM `bench serve`.
- Dataset: deterministic random prompts.
- Input/output: 2,048 prompt tokens and exactly 512 generated tokens (`--ignore-eos`).
- Concurrency: c1, c2, c4, c8, c16, c32.
- Repetitions: three per level; repetition 1 is warm-up and repetitions 2–3 are averaged.
- Result range: 90.86 output tok/s at c1 to 463.67 output tok/s at c32; zero failed requests.
- Artifacts: individual JSON files plus `concurrency/summary.json`.

## Context-depth sweep

- Tool: llama-benchy 0.4.0.
- Prompt/generation: 2,048 prompt tokens and exactly 128 generated tokens.
- Context depths: 0, 4,096, 8,192, 16,384 tokens.
- Repetitions: three per depth.
- Coherence: passed.
- Artifact: `llama-benchy/results.json`.

## Telemetry and energy

A two-second sampler recorded board power, GPU utilization, temperature, clocks, throttle state, and host memory with benchmark phase labels. The raw JSONL stream is excluded from Git. `telemetry-summary.json` preserves phase-level aggregates. `energy-efficiency.json` joins mean active power from stable repetitions 2–3 to throughput from those repetitions.

At c32, measured efficiency was 11.36 output tokens/J and 88.02 J per 1,000 output tokens.

## Runtime proof

`runtime-manifest.json` records the model revision, image ID, runtime versions, resolved serving profile, CUDA graph mode, MTP counters, KV capacity, and native-marker gates. `evidence/` contains selected startup lines and the 121-target scale audit.

The rejected `fp8-baseline/` directory is a labelled negative control showing why native B12X MoE alone did not clear the ordinary-linear no-Marlin gate. It never feeds release metrics.

## Integrity

`MANIFEST.sha256` hashes curated release artifacts. From the canonical run directory:

```bash
sha256sum -c MANIFEST.sha256
```

The HTML report, release consistency gate, manifest generation, and public-safety scan all passed.