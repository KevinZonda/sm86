# Qwen3.8-27B @ 262K 上下文 — 双卡 RTX 3080 20GB (SM86) 从零部署

纯 vLLM + NCCL，TP=2。本文假设一台刚装好的 Ubuntu（已验证内核 7.0 / Ubuntu archive，其余版本同理），只有 NVIDIA 原厂驱动。

目标：27B INT4 模型 + MTP 投机解码 + 262K 上下文，单用户 decode ~116 tok/s。

## 0. 硬件前提

- 双 RTX 3080 20GB（SM86），任意 PCIe 拓扑（无 NVLink 也行）
- 显存账本：权重 19.5GB + 激活/MTP ~4GB + fp8 KV 池 @262K ≈ 每卡 20GB 吃满 → `--max-num-seqs 1`（单并发是物理约束）

## 1. 系统依赖（一次性，sudo）

```bash
sudo apt update
sudo apt install -y python3.14 python3.14-venv gcc ninja-build
# flashinfer 首次启动要 JIT 编 CUDA 内核，需要 nvcc（CUDA 13 toolkit）：
sudo apt install -y cuda-toolkit-13
```

驱动：原厂 580.x 即可，不需要任何补丁/insmod。验证：`nvidia-smi` 能看到两张卡。

## 2. Python venv

```bash
python3.14 -m venv ~/venvs/sm86 && cd ~/venvs/sm86
NJMU="-i https://mirror.nju.edu.cn/pypi/web/simple"   # 或去掉用默认 PyPI
bin/pip install $NJMU --upgrade pip
bin/pip install $NJMU vllm==0.30.0
# vllm 0.30 钉死 torch 2.13，但我们验证过的组合是 torch 2.14（见 §6 注）：
bin/pip install $NJMU --no-deps torch==2.14.0 torchvision==0.29.0 triton==3.8.0
```

## 3. 下载模型 + 一处必须打的补丁

```bash
bin/pip install $NJMU -U huggingface_hub
bin/huggingface-cli download MIRALABS/Qwen3.8-27B-W4A16-AutoRound \
    --local-dir /mnt/modelzoo/MIRALABS/Qwen3.8-27B-W4A16-AutoRound
# 19.5GB，8 个分片
```

**补丁（必做，否则启动加载即炸）**：vLLM 会把 MTP draft 的 `fc` 建成量化模块，但 checkpoint 里它是 bf16 → `no module named fc.weight`。修法是在 ignore 列表加 `re:.*mtp.*`，且 **config.json 内嵌的 quantization_config 和 quantization_config.json 两份都要改**（vLLM 优先读前者）：

```bash
cd /mnt/modelzoo/MIRALABS/Qwen3.8-27B-W4A16-AutoRound
python3 - <<'EOF'
import json
for fn in ("config.json", "quantization_config.json"):
    cfg = json.load(open(fn))
    q = cfg.get("quantization_config", cfg)
    ign = q["ignore"]          # compressed-tensors 格式字段名是 ignore
    if not any("mtp" in x for x in ign):
        ign.append("re:.*mtp.*")
    json.dump(cfg, open(fn, "w"), indent=2)
    print(fn, "patched")
EOF
```

（原始文件留 `.orig` 备份的好习惯。）

## 4. 启动

```bash
HF_HUB_OFFLINE=1 ~/venvs/sm86/bin/vllm serve \
  /mnt/modelzoo/MIRALABS/Qwen3.8-27B-W4A16-AutoRound \
  --served-model-name qwen27b \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 1 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

首次启动 flashinfer JIT 编内核 2–5 分钟（之后同配置秒起）。测试：

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen27b",
  "messages": [{"role":"user","content":"用一句话解释 PCIe P2P DMA"}],
  "max_tokens": 128
}'
```

## 5. flag 为什么缺一不可（全是实测踩出来的）

| flag | 原因 |
|---|---|
| `--speculative-config '{"method":"mtp",...}'` | MTP 投机解码，decode TPS 41→116 的来源 |
| `--enable-chunked-prefill` | **MTP 必配**。不配则 16K 长上下文 prefill 51 分钟零输出 |
| `--max-num-seqs 1` | MTP 模式要求 `max_num_seqs ≤ Mamba cache blocks`（默认 16 > 15 直接拒启） |
| `--kv-cache-dtype fp8_e4m3` | KV 减半，262K 上下文放下的关键 |
| `--gpu-memory-utilization 0.93` | 显存吃满换 KV 容量 |

## 6. 实测数字与备注

- decode TPS：16K 输入/1K 输出 单用户 **116**（MTP on）/ **41**（off），NCCL 后端，2026-10-04
- 长上下文检索：needle @100K 深度 50% PASS（注意关思考模式，否则 32 token 截断必"失败"）
- torch 2.14 覆盖说明：vllm 0.30 上游钉死 torch 2.13，纯 NCCL 理论上用 2.13 也行，但本文所有数字都在 2.14 上测出，建议照抄。改 vllm 版本时此覆盖需重新评估
- 上下文上限 262K（KV 池余量约 38 万 token）；想更长只能再压权重或减上下文

## 7. 离线 bench（可选）

```bash
HF_HUB_OFFLINE=1 BENCH_MODEL=/mnt/modelzoo/MIRALABS/Qwen3.8-27B-W4A16-AutoRound \
  ~/venvs/sm86/bin/python bench_27b.py \
  --max-model-len 262144 --max-tokens 1024 --gpu-mem 0.93 --max-num-seqs 1 \
  --prompt-tokens 16384 --warm-repeat 1 \
  --mtp 3 --chunked-prefill --kv-dtype fp8_e4m3
```

（`bench_27b.py` 在 barlink 仓库 `demos/vllm_bl/` 下，纯 stdlib + vllm，可随手拷走。）
