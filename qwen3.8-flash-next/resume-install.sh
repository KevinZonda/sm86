#!/bin/bash
# 等待 ModelScope 下载完成后自动续装 Strata
DIR=/mnt/modelzoo/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ2_XS
while pgrep -f "modelscope download" >/dev/null 2>&1 || ls "$DIR"/*.incomplete >/dev/null 2>&1; do
  sleep 60
done
echo "=== model download finished, resuming Strata setup at $(date) ==="
cd /home/kevin/projects/nv-p2p/sm86/qwen3.8-flash-next/Strata
HF_ENDPOINT=https://hf-mirror.com ./setup.sh --yes --family qwen --model IQ2_XS --no-start \
  --prebuilt "https://gh-proxy.com/https://github.com/Niko1221/Strata/releases/download/v0.1.39/" \
  --gguf-dir "$DIR"
echo "=== setup exit code: $? at $(date) ==="
