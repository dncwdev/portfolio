# Methodology

## Test Conditions

| Parameter    | Value                                                                          |
| ------------ | ------------------------------------------------------------------------------ |
| Model        | gpt-oss:120b (MXFP4, 65GB)                                                     |
| Test prompts | "Hello" / "Explain the difference between RAG and fine-tuning in 3 sentences." |
| Repetitions  | 2–3 runs per condition                                                         |
| Concurrency  | Single request (sequential)                                                    |

## Metrics

- **TTFT (Time to First Token)**: measured via `curl -w "%{time_starttransfer}"` with `stream=true`
- **Throughput (TPS)**: tokens generated per second
  - Ollama: `eval rate` from `ollama run --verbose`
  - vLLM: `request_generation_tokens_sum / e2e_request_latency_seconds_sum` from `/metrics`

## Warm vs Cold State

- **Warm**: model already loaded in memory (`keepalive=-1` or recent prior request)
- **Cold**: model unloaded via `keep_alive: 0` API call before measurement

## Model Notes

- gpt-oss-120b is natively released as MXFP4 quantization by OpenAI (no FP16/FP32 variant available)
- Ollama uses GGUF format; vLLM uses safetensors format — both based on the same MXFP4 weights
- Comparison is engine-only (Ollama vs vLLM), not a quantization comparison

## Environment Notes

- DGX Spark GB10 unified memory (128GB) — sufficient for 65GB model with headroom
- H100 x2: air-gapped production node, warm state (vLLM keeps model loaded by default)
- All tests: single-node, single-request baseline (not multi-user load test)

## Known Limitations

- Single-request TPS favoring Ollama is consistent with known behavior
- vLLM's advantage emerges at concurrent load via continuous batching
- Results reflect DGX Spark GB10 unified memory architecture — may differ from discrete GPU setups
