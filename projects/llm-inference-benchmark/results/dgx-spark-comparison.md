# DGX Spark — vLLM vs Ollama — GPT-OSS 120B

## Environment

- Hardware: DGX Spark (GB10, 128GB unified memory)
- Model: gpt-oss:120b (MXFP4, 65GB)
- OS: Ubuntu (DGX OS)
- Ollama version: latest (pulled 2026-02-25)
- vLLM version: 0.10.1.1 (vllm-custom:25.09)

## Results

| Engine | State      | TTFT        | Throughput (tok/s) | Notes                        |
| ------ | ---------- | ----------- | ------------------ | ---------------------------- |
| Ollama | Warm       | ~300–1900ms | 43.55–43.80        | keepalive=-1 recommended     |
| Ollama | Cold start | ~21s        | 43.52              | model load dominates TTFT    |
| vLLM   | Warm       | ~2–22ms     | ~35.9              | stream=true, metrics-derived |

## Ollama Observations

- TPS remains stable at ~43.5 tokens/s regardless of warm or cold state
- Cold start is dominated by model loading (~20s), making TTFT impractical without keepalive
- In production, `keepalive=-1` is recommended to keep the model resident in memory
- Warm-state TTFT varies between 0.3–1.9s depending on prompt eval time

## vLLM Observations

- TTFT is significantly lower than Ollama (~2–22ms vs ~300–1900ms) due to streaming architecture
- TPS (~35.9 tok/s) is lower than Ollama in single-request scenario
- vLLM metrics endpoint (`/metrics`) is enabled by default — no additional configuration required
- TPS derived from: `request_generation_tokens_sum / e2e_request_latency_seconds_sum` via Prometheus metrics

## Summary

| Metric               | Ollama (Warm) | vLLM (Warm) | Winner |
| -------------------- | ------------- | ----------- | ------ |
| TTFT                 | ~300–1900ms   | ~2–22ms     | vLLM   |
| TPS (single request) | ~43.5 tok/s   | ~35.9 tok/s | Ollama |

> Note: Single-request benchmark only. Ollama outperforming vLLM in single-request TPS
> is consistent with known behavior — vLLM's advantage emerges under concurrent load
> via continuous batching. At 10+ concurrent users, vLLM throughput scales linearly
> while Ollama degrades significantly.
