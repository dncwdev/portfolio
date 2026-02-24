# LLM Inference Benchmark — GPT-OSS 120B

Performance comparison of GPT-OSS 120B across different hardware environments and inference engines.

## Environments

| Environment  | Hardware                  | Engine |
| ------------ | ------------------------- | ------ |
| H100 Cluster | H100 x2 (production node) | vLLM   |
| DGX Spark    | GB10 128GB unified memory | vLLM   |
| DGX Spark    | GB10 128GB unified memory | Ollama |

## Results Summary

| Environment | Engine | TTFT (ms) | Throughput (tok/s) |
| ----------- | ------ | --------- | ------------------ |
| H100 x2     | vLLM   | TBD       | TBD                |
| DGX Spark   | vLLM   | TBD       | TBD                |
| DGX Spark   | Ollama | TBD       | TBD                |

> Detailed results and methodology: see [results/](results/) and [methodology.md](methodology.md)

## Key Findings

_(To be filled after DGX Spark testing)_

## Notes

- All tests use the same model: GPT-OSS 120B
- Prompt and generation length kept consistent across runs
- See [methodology.md](methodology.md) for full test conditions
