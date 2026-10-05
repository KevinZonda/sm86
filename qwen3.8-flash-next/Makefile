QUANT_TYPE  ?= iq3_xxs
MODEL       := $(shell echo $(QUANT_TYPE) | tr a-z A-Z)
HF_ENDPOINT := https://hf-mirror.com
PREBUILT    := https://gh-proxy.com/https://github.com/Niko1221/Strata/releases/download/v0.1.39/
CTX_SIZE    ?= 262144

.PHONY: serve setup setup-txt-only health

serve:
	cd Strata && ./run-$(QUANT_TYPE).sh 2>&1

# --yes 的默认模型/上下文会偏离本机现有配置，这里用 QUANT_TYPE 显式指定并锁定 256K 上下文
setup:
	cd Strata && HF_ENDPOINT=$(HF_ENDPOINT) ./setup.sh --setup --yes \
	  --family qwen --model $(MODEL) --context $(CTX_SIZE) --vision yes \
	  --prebuilt "$(PREBUILT)" 2>&1

setup-txt-only:
	cd Strata && HF_ENDPOINT=$(HF_ENDPOINT) ./setup.sh --setup --yes \
	  --family qwen --model $(MODEL) --context $(CTX_SIZE) \
	  --prebuilt "$(PREBUILT)" 2>&1

health:
	curl -s http://127.0.0.1:8080/health && echo
