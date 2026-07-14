# Benchmark evidence

Canonical release run: [`qwen36-v025-mtp-20260714`](runs/qwen36-v025-mtp-20260714/)

## Quality

- GSM8K test split: 1,319/1,319 samples.
- 0-shot local chat completions, chat template applied, thinking disabled, greedy decoding, 2,048-token budget, zero retries.
- Flexible-extract exact match: **86.50% ± 0.94%**.

## Selected throughput

- Random 2,048-token inputs and exact 512-token outputs.
- c1, c8, c32; three repetitions each; first dropped, final two averaged.
- Output throughput: c1 **93.05**, c8 **287.18**, c32 **397.98 tok/s**.
- MTP acceptance: **78.56%**.

## Reuse and equivalence

`evidence-reuse.json` hashes every imported source and curated destination artifact. `equivalence/summary.json` records clean-image runtime, semantic, long-generation, c1, and c32 regression gates. No duplicate full evaluation was run for the equivalent clean image.

## Integrity

```bash
cd benchmarks/runs/qwen36-v025-mtp-20260714
sha256sum -c MANIFEST.sha256
```
