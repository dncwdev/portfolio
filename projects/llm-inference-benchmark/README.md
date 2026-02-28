# LLM Inference Benchmark — GPT-OSS 120B

Performance comparison of GPT-OSS 120B across inference engines and hardware environments.

## Environments

| Hardware                               | Engine | Context                        |
| -------------------------------------- | ------ | ------------------------------ |
| H100 x2 (production node)              | vLLM   | Air-gapped on-prem, production |
| DGX Spark (GB10, 128GB unified memory) | vLLM   | Lab, controlled benchmark      |
| DGX Spark (GB10, 128GB unified memory) | Ollama | Lab, controlled benchmark      |

## Results Summary

### H100 x2 — Production Reference

| Metric           | Value                 |
| ---------------- | --------------------- |
| TTFT             | ~40ms                 |
| Throughput       | ~368 tok/s            |
| Concurrent users | N/A (production load) |

> Measured under live production conditions. Concurrent user count not isolated.

### DGX Spark — Controlled Multi-User Benchmark

| Engine | Concurrency | TTFT mean (s) | TTFT P99 (s) | TPS (tok/s) |
| ------ | :---------: | :-----------: | :----------: | :---------: |
| Ollama |      1      |     4.96      |     7.70     |    42.0     |
| Ollama |      5      |     98.86     |    104.94    |     8.5     |
| Ollama |     10      |    222.16     |    229.14    |     4.3     |
| vLLM   |      1      |     2.69      |     4.01     |    35.0     |
| vLLM   |      5      |     9.62      |    20.07     |    17.6     |
| vLLM   |     10      |     26.61     |    101.10    |    13.0     |

![Benchmark Chart](results/ollamavsvllm/ollama_vs_vllm_benchmark.png)

## Key Findings

- **H100 vs DGX Spark:** H100 x2 delivers ~10x lower TTFT (40ms vs 2.69s) and ~10x higher throughput (368 vs 35 tok/s) — discrete HBM3 vs unified memory architecture
- **Single user:** Ollama slightly faster than vLLM (42 vs 35 tok/s) — no scheduling overhead
- **5 concurrent users:** vLLM TTFT **10x lower** (9.6s vs 98.9s) — continuous batching effect
- **10 concurrent users:** vLLM TTFT **8x lower** (26.6s vs 222.2s) — Ollama effectively unusable
- **System throughput:** vLLM achieves **4.8x higher** system TPS under 10-user load
- **Capacity guideline:** Single DGX Spark GB10 can serve ~10 concurrent users on a 120B model with vLLM

## Methodology

- DGX Spark: concurrent users simulated via `asyncio` from local PC → DGX Spark over LAN
- Round-robin prompt selection across 10 enterprise-representative prompts (eliminates KV cache bias)
- 3 sequential requests per user, concurrency levels: 1 / 5 / 10
- `max_tokens=1024`, `REQUEST_TIMEOUT=600s`
- vLLM config: `max-num-seqs=8`, `max-num-batched-tokens=16384`, `--async-scheduling`
- H100: measured under production load, single data point reference only

## Files

| File                                                                                               | Description                         |
| -------------------------------------------------------------------------------------------------- | ----------------------------------- |
| [`dgx_spark_concurrent_inference_benchmark.ipynb`](dgx_spark_concurrent_inference_benchmark.ipynb) | Full benchmark notebook (DGX Spark) |
| [`results/ollamavsvllm/`](results/ollamavsvllm/)                                                   | Raw CSVs, summary, chart, metadata  |
