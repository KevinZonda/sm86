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