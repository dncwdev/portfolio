docker exec -it vllm-ray-head bash

ray status

vllm serve /model \
  --served-model-name gpt-oss-120b \
  --max-model-len 131072 \
  --max-num-batched-tokens 16384 \
  --max-num-seqs 8 \
  --dtype auto \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --distributed-executor-backend ray \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --host 0.0.0.0 \
  --swap-space 32