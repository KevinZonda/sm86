MODEL := /mnt/modelzoo/MIRALABS/Qwen3.8-27B-W4A16-AutoRound
NAME  := qwen27b
EFFORT ?= medium

# 使用前需先激活 venv：
#   source ../.venv/bin/activate

.PHONY: serve chat check

check:
	@python -c "import vllm" 2>/dev/null || { \
		echo "错误: 当前 python 里没有 vllm，请先 source ../.venv/bin/activate"; \
		exit 1; }

serve: check
	HF_HUB_OFFLINE=1 vllm serve $(MODEL) \
	  --served-model-name $(NAME) \
	  --tensor-parallel-size 2 \
	  --max-model-len 262144 \
	  --gpu-memory-utilization 0.93 \
	  --max-num-seqs 1 \
	  --kv-cache-dtype fp8_e4m3 \
	  --enable-chunked-prefill \
	  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

chat: check
	vllm chat $(MODEL) \
	  --chat-template-kwargs '{"reasoning_effort":"$(EFFORT)"}'
