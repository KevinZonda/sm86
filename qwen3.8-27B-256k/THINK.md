# THINK.md — Qwen3.8-27B 思考模式管理

对应模型：[Qwen3.8-27B W4A16 @ 256k](./qwen3.8-27B-256k/README.md)。思考行为由 chat template 的
`chat_template_kwargs` 控制，vLLM 原样转发给 tokenizer，与服务端启动 flag 无关，每次请求可自由换档。

## 三个参数

| 参数 | 作用 | 默认值 |
|---|---|---|
| `enable_thinking` | 思考模式总开关 | `true`（开启） |
| `reasoning_effort` | 思考强度档位：`low` / `medium` / `xhigh` | `xhigh`（最高档，为跑分优化；本地跑明显拖首字响应） |
| `preserve_thinking` | 保留历史消息中的思考过程（利于多轮一致性与 KV Cache 复用） | `true` |

注意：低强度在多轮 Agent 任务中不一定更快——推理不足导致反复重试反而更慢。

## 换档方式

curl：

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen27b",
  "messages": [{"role":"user","content":"..."}],
  "max_tokens": 512,
  "chat_template_kwargs": {"reasoning_effort": "medium"}
}'
```

Python：

```python
client.chat.completions.create(
    model="qwen27b",
    messages=[...],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "medium"}},
)
```

交互式（连本机已启动的 serve）：

```bash
make chat    # 等价于 vllm chat --url http://localhost:8000/v1 --model-name qwen27b
```

注意：`vllm chat` 在 vLLM 0.30 里是 API client，**不支持**传 `chat_template_kwargs`，
要调思考强度请走上面的 curl / Python 方式。

## 场景建议

| 场景 | 建议 |
|---|---|
| 长上下文检索 / needle 类测试 | `enable_thinking: false`，否则短 `max_tokens` 截断必"失败"（见部署 README §6） |
| 日常问答 | `reasoning_effort: medium` |
| 翻译 / 摘要 / 格式整理 | `reasoning_effort: low` 或直接关思考 |
| 多步调试 / 复杂推理 | 保持 `xhigh` |

## 验证生效

开 `stream: true` 观察输出开头：`xhigh` 会先吐 `<think>` 且思考段很长，`low`/`medium` 很短或没有；
或对比 `usage.completion_tokens` 差异。
