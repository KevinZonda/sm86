# Strata 基准测试报告

- 日期：2026-10-05 13:13
- 机型：NVIDIA GeForce RTX 3080, 20480 MiB / NVIDIA GeForce RTX 3080, 20480 MiB；CPU AMD Ryzen 7 5700X 8-Core Processor；RAM 60GB
- 引擎：Strata 0.1.39（本地编译，archs [86]）；双卡层切分 + MTP 投机解码(spec 4)
- 模型：Qwen3.8-Flash-Next（ISTA-DASLab GSQ-RCO）IQ2_XS / IQ3_XXS，分片2硬链接共享

## 方法
- 每档上下文独立配置并重启服务；prompt = 90% 上下文长度的英文 filler + 固定输出任务（从1数到100）。
- prefill tok/s = prompt_tokens / 首token延迟（SSE 流式计时）；decode tok/s = (completion_tokens-1) / (首块→末块时间窗)。
- 短对话 = 仅计数任务（<50 tokens prompt）；显存 = 就绪后 nvidia-smi；内存 = 就绪前后 MemAvailable 差。
- KV 布局：IQ2_XS 全程 KV streaming（KV 驻内存）；IQ3_XXS 因 61GB 内存规则 KV 驻显存（setup 自动判定，长上下文档位专家缓存相应减少）。

## IQ2_XS
| 上下文 | prefill tok/s | 长ctx decode tok/s | 短对话 decode tok/s | 显存 卡0/卡1 (GiB) | 服务内存 (GiB) |
|---:|---:|---:|---:|---:|---:|
| 32K | 2983.9 | 155.5 | 173.0 | 19.1 / 19.2 | 36.7 |
| 64K | 3437.7 | 147.4 | 170.1 | 19.1 / 19.2 | 37.5 |
| 128K | 3676.8 | 144.2 | 165.8 | 19.1 / 19.2 | 38.5 |
| 192K | 3632.6 | 136.3 | 164.5 | 19.1 / 19.2 | 39.2 |
| 256K | 3521.5 | 135.4 | 167.2 | 19.1 / 19.2 | 40.1 |

## IQ3_XXS
| 上下文 | prefill tok/s | 长ctx decode tok/s | 短对话 decode tok/s | 显存 卡0/卡1 (GiB) | 服务内存 (GiB) |
|---:|---:|---:|---:|---:|---:|
| 32K | 2971.9 | 126.8 | 151.9 | 19.1 / 19.2 | 43.7 |
| 64K | 3378.6 | 121.3 | 150.8 | 19.1 / 19.2 | 43.8 |
| 128K | 3706.8 | 117.1 | 147.4 | 19.1 / 19.2 | 43.9 |
| 192K | 3707.1 | 111.2 | 143.2 | 19.1 / 19.2 | 43.8 |
| 256K | 3537.3 | 105.9 | 138.2 | 19.1 / 19.2 | 43.8 |

## 备注
- **decode 为乐观值**：计数任务输出高度可预测，MTP 投机解码接受率接近上限；自由文本实测约低 30-40%（参考：非流式自由文本短对话 iq2_xs ≈62-70 tok/s，iq3_xxs ≈45-53 tok/s，见 results_v1_invalid_decode.csv.bak 的 short_chat_tps）。
- decode 含 MTP 投机解码；长上下文 decode 因注意力范围增大低于短对话属正常。
- IQ3_XXS 专家更大（43 vs 36GB），显存命中率低于 IQ2_XS，decode 低约 20-30% 属预期。

原始数据：results.csv；全程日志：bench.log

## 2 并发测试（2026-10-05 增补）

- 方法：同模型同服务，先发 1 路基线再发 2 路完全并发（每路 prompt 带 nonce 防 prompt 缓存），流式计时；agg decode tok/s = 两路合计解码 token / 最大解码窗口。
- 长上下文场景 prompt ≈28.8K tokens；输出均为计数任务 200 tokens（MTP 友好，decode 为乐观值）。

| 模型 | 场景 | 并发 | 总耗时 wall(s) | 每路 decode tok/s | 聚合 decode tok/s | 每路 prefill tok/s | 显存 卡0/卡1 (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| IQ2_XS | short | 1 | 1.66 | 159.2–159.2 | 159.2 | 145.0 | 19.2 / 19.2 |
| IQ2_XS | short | 2 | 3.0 | 170.1–173.0 | 340.2 | 31.4 | 19.2 / 19.2 |
| IQ2_XS | long28k | 1 | 11.01 | 156.7–156.7 | 156.7 | 2963.9 | 19.2 / 19.3 |
| IQ2_XS | long28k | 2 | 13.9 | 157.9–157.9 | 315.9 | 2283.9 | 19.2 / 19.3 |
| IQ3_XXS | short | 1 | 2.43 | 102.6–102.6 | 102.6 | 118.4 | 19.2 / 19.3 |
| IQ3_XXS | short | 2 | 3.39 | 145.3–163.1 | 290.5 | 26.7 | 19.2 / 19.3 |
| IQ3_XXS | long28k | 1 | 10.91 | 146.3–146.3 | 146.3 | 3022.8 | 19.2 / 19.3 |
| IQ3_XXS | long28k | 2 | 13.93 | 151.9–151.9 | 303.8 | 2289.3 | 19.2 / 19.3 |

- **结论：2 并发下 decode 每路不掉速，聚合吞吐约翻倍**（IQ2_XS 340、IQ3_XXS 290 tok/s），
  说明引擎对 2 路序列做了 batch 合并，单路本来就吃不满双卡带宽。
- 长上下文并发 wall 13.9s vs 单发 11.0s：两路 prefill 排队执行，吞吐 ≈2x tokens / 1.26x 时间。
- 注意：长上下文并发的 "每路 prefill tok/s" 受 Strata prefix 缓存影响（nonce 在 prompt 末尾，
  与前一个单发请求共享几乎全部前缀），该列只作参考；短对话并发里第二路的低 prefill 数字是排队延迟，非真实 prefill 速度。

原始数据：concurrency_results.csv
