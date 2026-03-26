#!/bin/bash
# BGE-Reranker-v2-m3 Model Download
 
MODEL_NAME="BAAI/bge-reranker-v2-m3"
MODEL_DIR="/home/user/workspace/models/bge-reranker-v2-m3"
 
mkdir -p ${MODEL_DIR}
 
echo "Downloading ${MODEL_NAME} to ${MODEL_DIR}..."
nohup hf download ${MODEL_NAME} \
  --local-dir ${MODEL_DIR} \
  > ~/download-bge-reranker.log 2>&1 &
 
echo "Download started in background. PID: $!"
echo "Check progress: tail -f ~/download-bge-reranker.log"