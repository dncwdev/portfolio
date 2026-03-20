#!/bin/bash
docker rm -f vllm-ray-head 2>/dev/null || true

docker run -d --gpus all \
  --name vllm-ray-head \
  --network=host \
  --ipc=host \
  -v /home/user/dev-test-workspace/models/gpt-oss-120b:/model \
  -v /home/user/dev-test-workspace/models/tiktoken_encodings:/opt/tiktoken_encodings \
  -e TIKTOKEN_ENCODINGS_BASE=/opt/tiktoken_encodings \
  -e VLLM_HOST_IP=192.168.100.10 \
  -e GLOO_SOCKET_IFNAME=enp1s0f1np1 \
  -e NCCL_SOCKET_IFNAME=enp1s0f1np1 \
  nvcr.io/nvidia/vllm:25.09-py3 \
  bash -lc "
    ray start --head \
      --node-ip-address=192.168.100.10 \
      --port=6379 \
      --dashboard-host=0.0.0.0 \
      --num-gpus=1 \
      --block
  "