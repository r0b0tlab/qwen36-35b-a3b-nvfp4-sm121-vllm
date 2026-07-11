# Rejected runtime evidence

This run used native `FLASHINFER_B12X` for routed experts but vLLM hard-pinned ordinary `W4A16_NVFP4` linear layers to Marlin. It is retained only as diagnostic evidence and must not feed benchmark or release claims.

The accepted native baseline is:

```text
../fp8-native-w4a4-v1/
```
