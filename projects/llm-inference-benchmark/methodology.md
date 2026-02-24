# Methodology

## Test Conditions

| Parameter    | Value                                                                          |
| ------------ | ------------------------------------------------------------------------------ |
| Model        | gpt-oss:120b (MXFP4, 65GB)                                                     |
| Test prompts | "Hello" / "Explain the difference between RAG and fine-tuning in 3 sentences." |
| Repetitions  | 2-3 runs per condition                                                         |
| Concurrency  | Single request (sequential)                                                    |

## Metrics

- **TTFT (Time to First Token)**: load duration + prompt eval duration
- **Throughput**: eval rate (tokens/s, output tokens only)

## Warm vs Cold State

- **Warm**: model already loaded in memory (keepalive=-1 or recent prior request)
- **Cold**: model unloaded via `keep_alive: 0` API call before measurement

## Tooling

- Ollama: `ollama run [model] --verbose` (built-in stats output)
- vLLM: TBD

## Environment Notes

- DGX Spark GB10 unified memory (128GB) — sufficient for 65GB model with headroom
- H100 x2: air-gapped production node, warm state (vLLM keeps model loaded by default)
- All tests single-node, single-request baseline (not multi-user load test)
