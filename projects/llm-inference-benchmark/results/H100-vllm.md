# H100 x2 — vLLM — GPT-OSS 120B

## Environment

- Hardware: H100 x2 (single production node)
- Engine: vLLM (air-gapped, on-prem)
- Model: gpt-oss-120b (MXFP4)

## Results

| Metric                    | Value      |
| ------------------------- | ---------- |
| TTFT                      | ~40ms      |
| Throughput                | ~368 tok/s |
| Concurrent users (tested) | N/A        |

## Notes

- Measured under production load conditions (active users on the platform)
- Air-gapped environment, all artifacts served internally
- Concurrent user count not isolated during measurement
