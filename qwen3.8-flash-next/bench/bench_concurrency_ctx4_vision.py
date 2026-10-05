#!/usr/bin/env python3
"""分档上下文 4 并发基准（Vision On）：与 conc4 相同方法，配置加 --vision yes。

量化常驻图像编码器（GPU，预留 700 MiB）对高并发文本吞吐的影响。
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench as rb
import bench_concurrency_ctx as bc
import bench_concurrency_ctx4 as b4

OUT = bc.ROOT / "results" / "conc4v"
b4.CSV = OUT / "results.csv"
bc.rb.LOG = OUT / "bench.log"

# 包装 setup：注入 --vision yes（否则重配置会把 vision 关掉）
_orig_setup = bc.setup_ctx


def setup_ctx_vision(model, ctx):
    r = rb.run([str(rb.STRATA / "setup.sh"), "--yes", "--setup", "--family", "qwen",
                "--model", model.upper(), "--context", str(ctx), "--vision", "yes", "--no-start",
                "--gguf-dir", str(rb.MODELS[model]), *rb.SETUP_EXTRA],
               env=rb.SETUP_ENV, cwd=rb.STRATA)
    if r.returncode != 0:
        rb.log("配置失败: " + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
        sys.exit(1)


bc.setup_ctx = setup_ctx_vision

# 无 vision 的同期基线（BENCHMARK.md conc4 表）
BASELINE = {"iq2_xs": {65536: 594.0, 131072: 564.5, 196608: 545.2, 204800: 552.8},
            "iq3_xxs": {65536: 491.4, 131072: 460.1, 196608: 442.2, 204800: 435.0}}


def append_report(rows):
    L = ["", f"## 分档上下文 4 并发 + Vision On（2026-10-05 增补；编码器 GPU 常驻，预留 700 MiB）", "",
         "- 方法：与 conc4 相同（每路 prompt = 90% 上下文，nonce 开头）；配置 `--vision yes`，请求为纯文本，"
         "量化的是常驻编码器对文本并发的代价。", "",
         "| 模型 | 上下文 | 成功/总数 | 聚合 decode tok/s | 显存峰值 卡0/卡1 (GiB) | 内存峰值占用 (GiB) | 服务存活 |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        vp = r["vram_peak_mib"]
        vs = f"{vp[0]/1024:.1f} / {vp[1]/1024:.1f}" if vp and vp[0] else "-"
        L.append(f"| {r['model'].upper()} | {r['context']//1024}K | {r['ok_requests']}/{b4.CONC} | {r['agg_dec_tps']} "
                 f"| {vs} | {r['ram_used_peak_gib']} | {'是' if r['server_alive_after'] else '否'} |")
    L += ["", "与无 vision 的 conc4 聚合吞吐对比：", "",
          "| 模型 | 上下文 | 无 vision | 有 vision | 变化 |",
          "|---|---:|---:|---:|---:|"]
    for r in rows:
        o = BASELINE[r["model"]][r["context"]]
        n = r["agg_dec_tps"]
        L.append(f"| {r['model'].upper()} | {r['context']//1024}K | {o} | {n} "
                 f"| {100*(n-o)/o:+.1f}%" if n else f"| {r['model'].upper()} | {r['context']//1024}K | {o} | 失败 | - |")
    if any(r.get("errors") for r in rows):
        L += ["", "失败/异常明细："]
        for r in rows:
            if r.get("errors"):
                L.append(f"- {r['model'].upper()} @{r['context']//1024}K: {r['errors']}")
    L += ["", "原始数据：results/conc4v/results.csv", ""]
    md = bc.ROOT / "BENCHMARK.md"
    md.write_text(md.read_text().rstrip("\n") + "\n" + "\n".join(L))
    rb.log(f"已追加 {md}")


def main():
    rows = []
    for model in bc.MODELS:
        for ctx in bc.CONTEXTS:
            try:
                rows.append(b4.bench_ctx4(model, ctx))
            except Exception as e:
                rb.log(f"{model} @{ctx//1024}K 整点失败: {type(e).__name__}: {e}")
                rows.append({"model": model, "context": ctx, "concurrency": b4.CONC,
                             "wall_s": None, "prompt_tokens_each": None, "ok_requests": 0,
                             "errors": f"point failed: {e}"[:200], "ttft_s": [], "prefill_tps_each": [],
                             "dec_tps_each": [], "agg_dec_tps": None, "vram_idle_mib": None,
                             "vram_peak_mib": [None, None], "ram_used_peak_gib": None,
                             "server_alive_after": False, "ts": time.strftime("%Y-%m-%d %H:%M")})
                rb.stop_server()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(b4.CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    append_report(rows)


if __name__ == "__main__":
    main()
