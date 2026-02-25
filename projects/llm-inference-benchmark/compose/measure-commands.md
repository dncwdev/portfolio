# Benchmark Measurement Commands

## Ollama — TTFT & TPS

```bash
# Warm state — verbose output (TPS included)
docker exec -it ollama-benchmark ollama run gpt-oss:120b --verbose \
  "Explain the difference between RAG and fine-tuning in 3 sentences."

# Cold start — unload model first
curl http://localhost:11434/api/generate \
  -d '{"model": "gpt-oss:120b", "keep_alive": 0}'

# Then re-run verbose command above
```

## vLLM — TTFT

```bash
# TTFT via streaming
curl -s http://localhost:58888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain the difference between RAG and fine-tuning in 3 sentences."}],
    "stream": true,
    "max_tokens": 1024
  }' \
  -o /dev/null \
  -w "TTFT: %{time_starttransfer}s | Total: %{time_total}s\n"
```

## vLLM — TPS via Prometheus Metrics

```bash
curl -s http://localhost:58888/metrics \
  | grep -E "generation_tokens_sum|e2e_request_latency_seconds_sum"

# TPS = request_generation_tokens_sum / e2e_request_latency_seconds_sum
```

```

```
