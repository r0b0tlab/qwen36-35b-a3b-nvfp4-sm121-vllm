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

## Maximum context

- Separate c1 profile: 262,144 `max_model_len`, 6 GiB explicit FP8 KV, max sequences 1.
- Base AR KV capacity: **609,637 tokens**.
- MTP K=2 KV capacity: **528,230 tokens**.
- Five-position near-window retrieval, ordered dual-code retrieval, and 512 forced output tokens passed for both profiles.
- MTP K=2 accepted 534/706 drafted tokens (**75.64%**) during the long-context gate.
- Claim boundary: functional retrieval and bounded generation, not arbitrary 262K reasoning quality.

## Reuse and equivalence

`evidence-reuse.json` hashes every imported source and curated destination artifact. `equivalence/summary.json` records clean-image runtime, semantic, long-generation, c1, and c32 regression gates. No duplicate full evaluation was run for the equivalent clean image.

## Integrity

```bash
cd benchmarks/runs/qwen36-v025-mtp-20260714
sha256sum -c MANIFEST.sha256
```
