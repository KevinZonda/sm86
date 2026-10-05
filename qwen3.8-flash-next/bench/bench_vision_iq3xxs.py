#!/usr/bin/env python3
"""IQ3_XXS + vision（KV streaming 关，KV 驻显存）基准：5 档上下文单路流式计时。

与 BENCHMARK.md 首轮 IQ3_XXS 数据（无 vision）对比，量化 vision 开启后的代价。
每档: setup.sh 重配置(--vision yes --context N) -> 重启 -> 长ctx prefill/decode +
短对话计数 decode + 自由文本 decode -> 落盘。
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench as rb

ROOT = rb.ROOT
rb.LOG = ROOT / "results" / "vision_iq3" / "bench.log"
CSV = ROOT / "results" / "vision_iq3" / "results.csv"
MODEL = "iq3_xxs"
CONTEXTS = [32768, 65536, 131072, 196608, 262144]
FREE_PROMPT = "请用中文写一段约200字的说明文，介绍光合作用的过程，不要分点，直接成段。"


def setup_ctx(ctx):
    r = rb.run([str(rb.STRATA / "setup.sh"), "--yes", "--setup", "--family", "qwen",
                "--model", MODEL.upper(), "--context", str(ctx), "--vision", "yes", "--no-start",
                "--gguf-dir", str(rb.MODELS[MODEL]), *rb.SETUP_EXTRA],
               env=rb.SETUP_ENV, cwd=rb.STRATA)
    if r.returncode != 0:
        rb.log("配置失败: " + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
        sys.exit(1)


def dec_tps(d):
    return round((d["completion_tokens"] - 1) / d["decode_s"], 1) if d["decode_s"] > 0 else None


def bench_ctx(ctx):
    setup_ctx(ctx)
    rb.stop_server()
    idle_mem = rb.mem_available_gib()
    rb.start_server(MODEL)
    h = rb.http_json("/health")
    assert h.get("images"), "vision 未生效!"
    try:
        vram = rb.vram_mib()
        ram_used = round(idle_mem - rb.mem_available_gib(), 1) if idle_mem else None

        reps = int(ctx * rb.PROMPT_TARGET_RATIO / rb.TPR)
        long_prompt = rb.BASE_PARA * reps + rb.COUNT_TASK
        d1 = rb.chat_stream(long_prompt, 200)
        prefill_tps = round(d1["prompt_tokens"] / d1["ttft_s"], 1) if d1["prompt_tokens"] else None

        d2 = rb.chat_stream(rb.COUNT_TASK.strip(), 200)
        d3 = rb.chat_stream(FREE_PROMPT, 300)

        row = {"model": MODEL, "vision": "on", "kv": "vram", "context": ctx,
               "prompt_tokens": d1["prompt_tokens"], "prefill_s": d1["ttft_s"],
               "prefill_tps": prefill_tps,
               "decode_tps_count": dec_tps(d1), "short_count_tps": dec_tps(d2),
               "short_free_tps": dec_tps(d3),
               "vram0_mib": vram[0], "vram1_mib": vram[1], "ram_used_gib": ram_used,
               "ts": time.strftime("%Y-%m-%d %H:%M")}
        rb.log(str(row))
        return row
    finally:
        rb.stop_server()


# 无 vision 的同期数据（BENCHMARK.md 首轮，2026-10-05 白天，同机同配置无 vision）
BASELINE = {  # ctx -> (prefill_tps, longctx_count_tps, short_count_tps)
    32768: (2971.9, 126.8, 151.9), 65536: (3378.6, 121.3, 150.8),
    131072: (3706.8, 117.1, 147.4), 196608: (3707.1, 111.2, 143.2),
    262144: (3537.3, 105.9, 138.2),
}


def append_report(rows):
    L = ["", "## IQ3_XXS + Vision（2026-10-05 增补；KV streaming 关，KV 驻显存，编码器 GPU 预留 700 MiB）", "",
         "- 方法：与首轮相同（单路流式，prompt = 90% 上下文英文 filler + 计数任务 200 tokens），"
         "配置为 `--vision yes`（mmproj BF16，每图 ≤1024 image token）。",
         "- 自由文本列：说明文写作 300 tokens，MTP 接受率接近真实负载，比计数任务更接近日常速度。", "",
         "| 上下文 | prefill tok/s | 长ctx decode (计数) | 短对话 decode (计数) | 短对话 decode (自由文本) | 显存 卡0/卡1 (GiB) | 服务内存 (GiB) |",
         "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['context']//1024}K | {r['prefill_tps']} | {r['decode_tps_count']} | {r['short_count_tps']} "
                 f"| {r['short_free_tps']} | {r['vram0_mib']/1024:.1f} / {r['vram1_mib']/1024:.1f} | {r['ram_used_gib']} |")
    L += ["", "与首轮（无 vision，同为 KV 驻显存）对比：", "",
          "| 上下文 | prefill 变化 | 长ctx decode 变化 | 短对话 decode 变化 |",
          "|---:|---:|---:|---:|"]
    for r in rows:
        b = BASELINE[r["context"]]
        pct = lambda n, o: f"{100*(n-o)/o:+.1f}%"
        L.append(f"| {r['context']//1024}K | {pct(r['prefill_tps'], b[0])} | {pct(r['decode_tps_count'], b[1])} "
                 f"| {pct(r['short_count_tps'], b[2])} |")
    L += ["", "原始数据：results/vision_iq3/results.csv", ""]
    md = ROOT / "BENCHMARK.md"
    md.write_text(md.read_text().rstrip("\n") + "\n" + "\n".join(L))
    rb.log(f"已追加 {md}")


def main():
    rows = []
    for ctx in CONTEXTS:
        rows.append(bench_ctx(ctx))
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    append_report(rows)


if __name__ == "__main__":
    main()
