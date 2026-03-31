#!/bin/bash
# BGE-Reranker-v2-m3 Reranker Server
# DGX Spark #2 - Port 58002
 
MODEL_DIR="/home/user/workspace/models/bge-reranker-v2-m3"
 
echo "Starting vLLM reranker server on port 58002..."
docker run -d \
  --name vllm-reranker \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 58002:8000 \
  -v ${MODEL_DIR}:/model \
  nvcr.io/nvidia/vllm:25.09-py3 \
  python -m vllm.entrypoints.openai.api_server \
  --model /model \
  --task score \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --dtype auto \
  --port 8000
 
echo "Done. Check logs: docker logs vllm-reranker"