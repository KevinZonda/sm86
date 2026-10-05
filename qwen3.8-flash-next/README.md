# Qwen3.8-Flash-Next 本地部署（Strata）

在本机（2× RTX 3080 20GB / 64GB RAM / Ryzen 7 5700X）上通过 [Strata](https://github.com/Niko1221/Strata) 推理引擎运行 **Qwen3.8-Flash-Next**（125B MoE，每 token 激活约 6B）。Strata 把专家分层存放：常用专家在显存、全部专家在内存、n-gram 查表在 SSD，配合 MTP 投机解码，使消费级显卡可跑该模型并提供 OpenAI / Anthropic 兼容 API。

## 目录结构

```
qwen3.8-flash-next/
├── README.md            ← 本文件
├── Strata/              ← Strata 源码（git.kigml.com 镜像克隆，v0.1.39）
│   ├── setup.sh         ← 安装/启动入口（首次跑完整安装，之后启动模型）
│   ├── setup.py
│   ├── third_party/llama.cpp/   ← 预置的 pinned 源码（gh-proxy 下载，setup 会跳过它的下载）
│   ├── .venv/           ← setup.sh 自动创建的私有 Python 环境（不要手工往里装东西）
│   ├── engine/          ← 编译出的推理引擎（sm_86，本地源码编译，无 Linux 预编译包）
│   └── strata-iq2_xs.json / run-iq2_xs.sh   ← 安装后生成：配置 + 启动脚本
└── Strata-data/         ← 模型数据（setup 默认放在 Strata 旁边； packs/ mtp/ models/）
```

模型 GGUF 分片在模型库目录（安装时通过 `--gguf-dir` 原地引用，不复制；与系统同盘，非机械盘/网盘）：

```
/mnt/modelzoo/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ2_XS/
├── Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf   (39.2 GB)
└── Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf   (28.8 GB)
```

## 网络环境说明（本机 GitHub / HuggingFace 直连不通）

| 用途 | 通路 |
|---|---|
| Git 克隆 Strata | `https://github.com/Niko1221/Strata.git`|
| GitHub 文件 / release / API | `https://gh-proxy.com/https://github.com/...` 前缀代理 |
| HuggingFace 下载 | `HF_ENDPOINT=https://hf-mirror.com`（setup.py 与 mtp_fetch.py 都认这个环境变量） |
| 模型本体（68GB） | ModelScope 镜像 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` |
| pip | 已配置南大镜像 `/etc/pip.conf` |

llama.cpp 第三方源码已预置到 `Strata/third_party/llama.cpp/`（pinned commit `3cf0325`），否则 setup 会从被墙的 github.com 下载。

## 为什么选择 IQ2_XS

| 规格 | 下载 | 内存需求 | 写速度（RTX 5070 12GB 参考） |
|---|---|---|---|
| Q2_0 | 66.4 GB | ~48 GB | 94 tok/s |
| **IQ2_XS** | **68.0 GB** | **~48 GB** | **79 tok/s** |
| IQ3_XXS | 75.8 GB | ~60 GB | 62 tok/s |
| IQ3_S | 83.6 GB | ~62 GB（本机 tight） | 53 tok/s |

IQ2_XS 是本机（61GB RAM）质量与速度的平衡点；显存 40GB 双卡可承载更多专家，实际速度应优于表单卡参考值。

## 安装与启动

### 1. 下载模型（ModelScope，68GB，可断点续传）

```bash
cd /home/kevin/projects/nv-p2p/sm86
.venv/bin/modelscope download \
  --model ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF \
  --include "IQ2_XS/*" \
  --local_dir /mnt/modelzoo/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
```

### 2. 安装（模型就位后执行）

```bash
cd /home/kevin/projects/nv-p2p/sm86/qwen3.8-flash-next/Strata
HF_ENDPOINT=https://hf-mirror.com ./setup.sh \
  --yes --family qwen --model IQ2_XS --no-start \
  --prebuilt "https://gh-proxy.com/https://github.com/Niko1221/Strata/releases/download/v0.1.39/" \
  --gguf-dir /mnt/modelzoo/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ2_XS
```

说明：
- `--prebuilt` 指向 gh-proxy 代理的 release 地址：GitHub 的 release 只有 Windows 资产，Linux 引擎会快速 404 后转为本地源码编译（10-20 分钟，需要 nvcc + g++，本机 CUDA 13.4 已就绪）。如果不加这个参数，setup 会对被墙的 github.com 做两次 60 秒超时的 HEAD 请求才走到编译。
- 首次安装会在模型校验处停下（若分片未下完）；重跑同一命令即可续装，已完成步骤自动跳过。
- MTP 草稿层（约 5GB，来自 Qwen 官方 BF16 checkpoint 的 Range 下载）走 `HF_ENDPOINT` 镜像，且有 SHA256 校验。
- 装好后启动脚本为 `run-iq2_xs.sh`，配置文件 `strata-iq2_xs.json`。

### 3. 启动 / 停止 / 验证

```bash
cd /home/kevin/projects/nv-p2p/sm86/qwen3.8-flash-next/Strata
nohup ./run-iq2_xs.sh > strata-server.out 2>&1 &   # 启动
curl http://127.0.0.1:8080/health                 # 验证（首次加载需几分钟，期间机器会卡 1-3 分钟，属正常）
```

- 浏览器界面：`http://127.0.0.1:8080`（Chat / Monitor / About）
- OpenAI 兼容：`http://127.0.0.1:8080/v1`（任意 API key、任意模型名）
- Anthropic 兼容：`http://127.0.0.1:8080/v1/messages`
- 停止：杀掉启动进程即可；引擎日志 `strata-iq2_xs.log`

### 4. 日常命令

```bash
./setup.sh                       # 再次运行 = 启动已装模型
./setup.sh --setup --vision yes  # 改配置（如开图像理解，会补下 0.9GB mmproj）
./update.sh                      # 更新 Strata（git pull 走 git.kigml.com 远端）
./setup.sh --check               # 环境自检
```

也可以用上层 Makefile：`make serve`（QUANT_TYPE 默认 iq3_xxs）、`make setup`（锁 256K 上下文 + vision）、`make health`。

### Vision（图像理解）

IQ2_XS 和 IQ3_XXS 两个配置均已开启（2026-10-05）：编码器 `Strata/engine/strata-vision`（GPU，预留 700 MiB 显存，每图最多 1024 image token），mmproj 在 `Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf`（两配置共用）。`/health` 返回 `"images": true` 即生效。API 用法：messages content 里加 `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`（也支持 http(s) 图片 URL）；网页 Chat 页可直接贴图，`chat.py` 里用 `/image <路径>`。注意：IQ3_XXS + vision 时 KV streaming 自动关（RAM 不够），KV 全留显存。

## 关于 .venv

**不需要手工做**：`setup.sh` 第一步就会在 `Strata/.venv/` 自动创建私有 Python 环境（系统 Python 3.14 + requirements.txt 固定版本），与仓库外层的 `.venv` 完全隔离。这是 Strata 的设计使然——更新 Strata 时 `.venv` 和引擎都会保留复用。

## 性能调优（后续可做）

- `./setup.sh --yes --calibrate`：实测选择最快的引擎参数（5-10 分钟）
- 双卡已默认启用（层切分，显存合计装更多专家，prefill 约快 20%）：见 `Strata/docs/MULTI_GPU.md`
- 首次启动后可观察 Monitor 页的专家命中率，必要时 `--setup --context <N>` 调整上下文
- 思考档位：API 请求里 `"reasoning_effort": "none"|"low"|"medium"|"high"`（默认 high，关思考最快）

## 故障排查

| 现象 | 处理 |
|---|---|
| 模型校验失败 / shard 不完整 | 重跑 ModelScope 下载命令续传，再重跑安装命令 |
| 首次启动机器卡 1-3 分钟 | 正常（往内存加载 35-55GB），勿关窗口；超 10 分钟再重启处理 |
| 磁盘忙、极慢、引擎意外退出 | 内存不足：关浏览器等程序，或换更小规格（`--setup --model Q2_0`） |
| MTP 校验失败 | 镜像不支持 Range 请求导致（#327）：setup 会自动重取；反复失败就重跑安装命令 |
| 端口 8080 被占 | 已在运行（查 `/health`），或 `--port 8081` |

完整文档在 `Strata/docs/`：`INSTALL.md`（安装）、`MODELS.md`（规格）、`MULTI_GPU.md`（多卡）、`TROUBLESHOOTING.md`（排错）、`DETAILS.md`（API 与全部参数）。
