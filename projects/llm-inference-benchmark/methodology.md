# Methodology

## Test Conditions

| Parameter            | Value        |
| -------------------- | ------------ |
| Model                | GPT-OSS 120B |
| Input prompt length  | TBD tokens   |
| Output length        | TBD tokens   |
| Concurrent requests  | TBD          |
| Repetitions per test | TBD          |

## Metrics

- **TTFT (Time to First Token)**: measured from request submission to first token received
- **Throughput**: tokens generated per second (output tokens only)

## Tooling

- Measurement method: TBD (e.g., vLLM benchmark script, custom client)

## Constraints

- H100 environment: air-gapped, production node (shared load may affect results)
- DGX Spark: single-node, unified memory architecture
