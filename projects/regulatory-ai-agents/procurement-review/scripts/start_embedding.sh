#!/bin/bash
# BGE-M3 Embedding Server
# DGX Spark #2 - Port 8001

docker run -d \
  --name vllm-embedding \
  --runtime nvidia \
  --gpus all \
  -p 8001:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN=${HF_TOKEN} \
  vllm/vllm-openai:latest \
  --model BAAI/bge-m3 \
  --task embedding \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.3 \
  --dtype float16 \
  --port 8000