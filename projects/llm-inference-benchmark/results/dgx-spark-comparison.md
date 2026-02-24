# DGX Spark — vLLM vs Ollama — GPT-OSS 120B

## Environment

- Hardware: DGX Spark (GB10, 128GB unified memory)
- Model: gpt-oss:120b (MXFP4, 65GB)
- OS: Ubuntu (DGX OS)
- Ollama version: latest (pulled 2026-02-24)

## Results

| Engine | State      | Load Duration | TTFT      | Throughput (tok/s) |
| ------ | ---------- | ------------- | --------- | ------------------ |
| Ollama | Warm       | ~160-210ms    | ~0.3-1.9s | 43.55 - 43.80      |
| Ollama | Cold start | ~20,712ms     | ~21s      | 43.52              |
| vLLM   | Warm       | TBD           | TBD       | TBD                |

## Ollama Observations

- TPS remains stable at ~43.5 tokens/s regardless of warm or cold state
- Cold start is dominated by model loading (~20s), making TTFT impractical without keepalive
- In production, `keepalive=-1` is recommended to keep the model resident in memory
- Warm-state TTFT varies between 0.3–1.9s depending on prompt eval time

## vLLM Observations

_(To be filled after vLLM testing)_
