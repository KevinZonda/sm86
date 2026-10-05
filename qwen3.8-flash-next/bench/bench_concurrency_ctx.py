#!/usr/bin/env python3
"""分档上下文 2 并发基准：64K/128K/192K/200K，重点采样显存/内存峰值。

每档: setup.sh 重配置 --context N -> 重启服务 -> 空闲采样 ->
并发 2 路 (prompt ≈90% 上下文, nonce 在开头防 prefix 缓存, 200 tokens 计数任务输出)
过程中 1s 间隔采样 MemAvailable 最小值 / 两卡显存最大值 -> 落盘。
"""
import csv
import sys
import time
import threading
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench as rb

ROOT = rb.ROOT
rb.LOG = ROOT / "results" / "conc2" / "bench.log"     # 与 bench_concurrency 共用
CSV = ROOT / "results" / "conc2" / "results_ctx.csv"
CONTEXTS = [65536, 131072, 196608, 204800]
MODELS = ["iq2_xs", "iq3_xxs"]
OUT_TOKENS = 200
PROMPT_RATIO = 0.9

stop_flag = threading.Event()
samples = []  # (ts, mem_avail_gib, vram0_mib, vram1_mib)


def sampler():
    while not stop_flag.is_set():
        samples.append((rb.mem_available_gib(), *rb.vram_mib()))
        time.sleep(1)


def make_prompt(ctx, nonce):
    reps = int(ctx * PROMPT_RATIO / rb.TPR)
    return f"(nonce: {nonce}-{time.time_ns()})\n" + rb.BASE_PARA * reps + rb.COUNT_TASK


def one(prompt):
    t0 = time.time()
    d = rb.chat_stream(prompt, OUT_TOKENS)
    d["wall_s"] = round(time.time() - t0, 2)
    d["dec_tps"] = round((d["completion_tokens"] - 1) / d["decode_s"], 1) if d["decode_s"] > 0 else None
    d["prefill_tps"] = round(d["prompt_tokens"] / d["ttft_s"], 1) if d["prompt_tokens"] else None
    return d


def setup_ctx(model, ctx):
    r = rb.run([str(rb.STRATA / "setup.sh"), "--yes", "--setup", "--family", "qwen",
                "--model", model.upper(), "--context", str(ctx), "--no-start",
                "--gguf-dir", str(rb.MODELS[model]), *rb.SETUP_EXTRA],
               env=rb.SETUP_ENV, cwd=rb.STRATA)
    if r.returncode != 0:
        rb.log("配置失败: " + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
        sys.exit(1)


def bench_ctx(model, ctx):
    setup_ctx(model, ctx)
    rb.stop_server()
    idle_mem = rb.mem_available_gib()
    rb.start_server(model)
    vram_idle = rb.vram_mib()
    try:
        global samples
        samples = []
        stop_flag.clear()
        th = threading.Thread(target=sampler, daemon=True)
        prompts = [make_prompt(ctx, i) for i in range(2)]
        t0 = time.time()
        th.start()
        with concurrent.futures.ThreadPoolExecutor(2) as ex:
            rs = list(ex.map(one, prompts))
        wall = round(time.time() - t0, 2)
        stop_flag.set()
        th.join(timeout=5)
        min_avail = min(s[0] for s in samples if s[0])
        vram_max = [max(s[1 + i] for s in samples) for i in range(2)]
        row = {
            "model": model, "context": ctx, "concurrency": 2, "wall_s": wall,
            "prompt_tokens_each": rs[0]["prompt_tokens"],
            "ttft_s": [r["ttft_s"] for r in rs],
            "prefill_tps_each": [r["prefill_tps"] for r in rs],
            "dec_tps_each": [r["dec_tps"] for r in rs],
            "agg_dec_tps": round(sum(r["completion_tokens"] - 1 for r in rs) / max(r["decode_s"] for r in rs), 1),
            "vram_idle_mib": vram_idle, "vram_peak_mib": vram_max,
            "ram_used_peak_gib": round(idle_mem - min_avail, 1),
            "ts": time.strftime("%Y-%m-%d %H:%M"),
        }
        rb.log(f"{model} @{ctx//1024}K x2: wall {wall}s, prompt {row['prompt_tokens_each']} tok, "
               f"ttft {row['ttft_s']}s, dec {row['dec_tps_each']} tok/s, agg {row['agg_dec_tps']}, "
               f"vram_peak {vram_max}, ram_peak_used {row['ram_used_peak_gib']}GiB")
        return row
    finally:
        rb.stop_server()


def main():
    rows = []
    for model in MODELS:
        for ctx in CONTEXTS:
            rows.append(bench_ctx(model, ctx))
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rb.log(f"已写入 {CSV}")
    append_report(rows)


def append_report(rows):
    L = ["", "## 分档上下文 2 并发（2026-10-05 增补，重点：内存/显存）", "",
         "- 方法：每档独立配置并重启服务；2 路完全并发，prompt = 90% 上下文长度（nonce 在开头，"
         "无前缀缓存共享，两路均全量 prefill）；输出 200 tokens 计数任务。",
         "- 显存/内存为过程中 1s 间隔采样的峰值；ram_used_peak = 服务就绪后 MemAvailable 基线 − 过程中最小值。", "",
         "| 模型 | 上下文 | 每路 prompt tokens | 每路 TTFT (s) | 每路 decode tok/s | 聚合 decode tok/s | 显存峰值 卡0/卡1 (GiB) | 内存峰值占用 (GiB) |",
         "|---|---:|---:|---|---|---:|---:|---:|"]
    for r in rows:
        ttft = " / ".join(str(x) for x in r["ttft_s"])
        dec = " / ".join(str(x) for x in r["dec_tps_each"])
        L.append(f"| {r['model'].upper()} | {r['context']//1024}K | {r['prompt_tokens_each']} | {ttft} | {dec} "
                 f"| {r['agg_dec_tps']} | {r['vram_peak_mib'][0]/1024:.1f} / {r['vram_peak_mib'][1]/1024:.1f} "
                 f"| {r['ram_used_peak_gib']} |")
    L += ["", "原始数据：results/conc2/results_ctx.csv", ""]
    md = ROOT / "BENCHMARK.md"
    md.write_text(md.read_text().rstrip("\n") + "\n" + "\n".join(L))
    rb.log(f"已追加 {md}")


if __name__ == "__main__":
    main()
