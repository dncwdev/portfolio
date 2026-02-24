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

- TPS는 warm/cold 관계없이 **43.5 tokens/s 수준으로 안정적**
- Cold start 시 모델 로딩(~20초)이 TTFT를 압도함
- 운영 환경에서는 `keepalive=-1` 설정으로 모델을 메모리에 상주시켜 warm 상태 유지 필요
- Warm 상태 기준 TTFT는 프롬프트 처리 시간에 따라 0.3~1.9s 범위

## vLLM Observations

_(To be filled after vLLM testing)_
