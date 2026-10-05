#!/usr/bin/env python3
"""2 并发基准：单发基线 vs 2 路并发，短对话 + ~28K 长上下文，IQ2_XS / IQ3_XXS。

每个请求 prompt 带不同 nonce，避免 Strata prompt 缓存让并发里的第二个请求白嫖 prefill。
落盘: bench/concurrency_results.csv, 并追加 BENCHMARK.md 的并发章节。
"""
import csv
import sys
import time
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench as rb

ROOT = rb.ROOT
CSV = ROOT / "concurrency_results.csv"
LONG_REPS = 430                                   # ≈ 28.8K tokens
LONG_PROMPT = rb.BASE_PARA * LONG_REPS + rb.COUNT_TASK
MODELS = ["iq2_xs", "iq3_xxs"]
OUT_TOKENS = 200


def nonce_prompt(base, n):
    return base + f"\n\n(nonce: {n}-{time.time_ns()})"


def one(prompt, tag):
    t0 = time.time()
    d = rb.chat_stream(prompt, OUT_TOKENS)
    d["wall_s"] = round(time.time() - t0, 2)
    d["tag"] = tag
    d["dec_tps"] = round((d["completion_tokens"] - 1) / d["decode_s"], 1) if d["decode_s"] > 0 else None
    d["prefill_tps"] = round(d["prompt_tokens"] / d["ttft_s"], 1) if d["prompt_tokens"] else None
    return d


def scenario(model, label, base_prompt, parallel):
    n = 2 if parallel else 1
    prompts = [nonce_prompt(base_prompt, i) for i in range(n)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(lambda p: one(p, f"{label}-{'par' if concurrent else 'single'}-{id(p) % 997}"), prompts))
    wall = round(time.time() - t0, 2)
    agg_dec = sum(r["completion_tokens"] - 1 for r in rs)
    agg_dec_s = max(r["decode_s"] for r in rs) or 1
    row = {
        "model": model, "scenario": label, "concurrency": n, "wall_s": wall,
        "prefill_tps_min": min(r["prefill_tps"] for r in rs if r["prefill_tps"]),
        "ttft_max_s": max(r["ttft_s"] for r in rs),
        "dec_tps_min": min(r["dec_tps"] for r in rs if r["dec_tps"]),
        "dec_tps_max": max(r["dec_tps"] for r in rs if r["dec_tps"]),
        "agg_dec_tps": round(agg_dec / agg_dec_s, 1),
        "vram0_mib": rb.vram_mib()[0], "vram1_mib": rb.vram_mib()[1],
        "ts": time.strftime("%Y-%m-%d %H:%M"),
    }
    rb.log(f"{model} {label} x{n}: wall {wall}s, dec_tps {row['dec_tps_min']}-{row['dec_tps_max']}, agg {row['agg_dec_tps']}")
    for r in rs:
        rb.log(f"  ttft {r['ttft_s']}s prefill {r['prefill_tps']} dec {r['dec_tps']} tok/s ({r['completion_tokens']} tok / {r['decode_s']}s)")
    return row


def main():
    rows = []
    for model in MODELS:
        rb.stop_server()
        idle_mem = rb.mem_available_gib()
        rb.start_server(model)
        ram_used = round(idle_mem - rb.mem_available_gib(), 1)
        try:
            rows.append({**scenario(model, "short", rb.COUNT_TASK, False), "ram_used_gib": ram_used})
            rows.append({**scenario(model, "short", rb.COUNT_TASK, True), "ram_used_gib": ram_used})
            rows.append({**scenario(model, "long28k", LONG_PROMPT, False), "ram_used_gib": ram_used})
            rows.append({**scenario(model, "long28k", LONG_PROMPT, True), "ram_used_gib": ram_used})
        finally:
            rb.stop_server()
    write_csv(rows)
    append_report(rows)


def write_csv(rows):
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rb.log(f"已写入 {CSV}")


def append_report(rows):
    L = ["", "## 2 并发测试（2026-10-05 增补）", "",
         "- 方法：同模型同服务，先发 1 路基线再发 2 路完全并发（每路 prompt 带 nonce 防 prompt 缓存），"
         "流式计时；agg decode tok/s = 两路合计解码 token / 最大解码窗口。",
         "- 长上下文场景 prompt ≈28.8K tokens；输出均为计数任务 200 tokens（MTP 友好，decode 为乐观值）。", "",
         "| 模型 | 场景 | 并发 | 总耗时 wall(s) | 每路 decode tok/s | 聚合 decode tok/s | 每路 prefill tok/s | 显存 卡0/卡1 (GiB) |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['model'].upper()} | {r['scenario']} | {r['concurrency']} | {r['wall_s']} "
                 f"| {r['dec_tps_min']}–{r['dec_tps_max']} | {r['agg_dec_tps']} | {r['prefill_tps_min']} "
                 f"| {r['vram0_mib']/1024:.1f} / {r['vram1_mib']/1024:.1f} |")
    L += ["", "原始数据：concurrency_results.csv", ""]
    md = ROOT / "BENCHMARK.md"
    md.write_text(md.read_text().rstrip("\n") + "\n" + "\n".join(L))
    rb.log(f"已追加 {md}")


if __name__ == "__main__":
    main()
