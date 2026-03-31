#!/bin/bash
# BGE-M3 Embedding Model Download
 
MODEL_NAME="BAAI/bge-m3"
MODEL_DIR="/home/user/workspace/models/bge-m3"
 
mkdir -p ${MODEL_DIR}
 
echo "Downloading ${MODEL_NAME} to ${MODEL_DIR}..."
nohup hf download ${MODEL_NAME} \
  --local-dir ${MODEL_DIR} \
  > ~/download-bge-m3.log 2>&1 &
 
echo "Download started in background. PID: $!"
echo "Check progress: tail -f ~/download-bge-m3.log"
 