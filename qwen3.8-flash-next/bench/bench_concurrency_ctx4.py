#!/usr/bin/env python3
"""分档上下文 4 并发基准：64K/128K/192K/200K，重点采样显存/内存峰值。

复用 bench_concurrency_ctx 的构件；与 2 并发版同方法，仅并发数=4。
单路请求异常（超时/断连/服务 OOM）不拖垮整点；每点结束后检查服务存活。
"""
import csv
import sys
import time
import json
import threading
import urllib.request
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_bench as rb
import bench_concurrency_ctx as bc

CSV = bc.ROOT / "concurrency_ctx4_results.csv"
CONC = 4
API = rb.API


def server_alive():
    try:
        with urllib.request.urlopen(API + "/health", timeout=3) as r:
            return json.load(r).get("loaded", False)
    except Exception:
        return False


def guarded_one(prompt):
    try:
        return bc.one(prompt)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def bench_ctx4(model, ctx):
    bc.setup_ctx(model, ctx)
    rb.stop_server()
    idle_mem = rb.mem_available_gib()
    rb.start_server(model)
    vram_idle = rb.vram_mib()
    try:
        bc.samples = []
        bc.stop_flag.clear()
        th = threading.Thread(target=bc.sampler, daemon=True)
        prompts = [bc.make_prompt(ctx, i) for i in range(CONC)]
        t0 = time.time()
        th.start()
        with concurrent.futures.ThreadPoolExecutor(CONC) as ex:
            rs = [f.result() for f in [ex.submit(guarded_one, p) for p in prompts]]
        wall = round(time.time() - t0, 2)
        bc.stop_flag.set()
        th.join(timeout=5)
        ok = [r for r in rs if "error" not in r]
        errs = [r["error"] for r in rs if "error" in r]
        s = [x for x in bc.samples if x[0]]
        min_avail = min(x[0] for x in s) if s else None
        vram_max = [max(x[1 + i] for x in bc.samples) for i in range(2)] if bc.samples else [None, None]
        row = {
            "model": model, "context": ctx, "concurrency": CONC, "wall_s": wall,
            "prompt_tokens_each": ok[0]["prompt_tokens"] if ok else None,
            "ok_requests": len(ok), "errors": "; ".join(errs)[:200],
            "ttft_s": [r.get("ttft_s") for r in rs],
            "prefill_tps_each": [r.get("prefill_tps") for r in rs],
            "dec_tps_each": [r.get("dec_tps") for r in rs],
            "agg_dec_tps": (round(sum(r["completion_tokens"] - 1 for r in ok) / max(r["decode_s"] for r in ok), 1)
                            if len(ok) >= 2 and max(r["decode_s"] for r in ok) > 0 else None),
            "vram_idle_mib": vram_idle, "vram_peak_mib": vram_max,
            "ram_used_peak_gib": round(idle_mem - min_avail, 1) if min_avail else None,
            "server_alive_after": server_alive(),
            "ts": time.strftime("%Y-%m-%d %H:%M"),
        }
        rb.log(f"{model} @{ctx//1024}K x{CONC}: wall {wall}s, ok {len(ok)}/{CONC}, "
               f"dec {[r.get('dec_tps') for r in rs]}, agg {row['agg_dec_tps']}, "
               f"vram_peak {vram_max}, ram_peak_used {row['ram_used_peak_gib']}GiB, "
               f"alive_after={row['server_alive_after']}" + (f" ERR: {errs[0][:80]}" if errs else ""))
        return row
    finally:
        rb.stop_server()


def main():
    rows = []
    for model in bc.MODELS:
        for ctx in bc.CONTEXTS:
            try:
                rows.append(bench_ctx4(model, ctx))
            except Exception as e:
                rb.log(f"{model} @{ctx//1024}K 整点失败: {type(e).__name__}: {e}")
                rows.append({"model": model, "context": ctx, "concurrency": CONC,
                             "wall_s": None, "prompt_tokens_each": None, "ok_requests": 0,
                             "errors": f"point failed: {e}"[:200], "ttft_s": [], "prefill_tps_each": [],
                             "dec_tps_each": [], "agg_dec_tps": None, "vram_idle_mib": None,
                             "vram_peak_mib": [None, None], "ram_used_peak_gib": None,
                             "server_alive_after": False, "ts": time.strftime("%Y-%m-%d %H:%M")})
                rb.stop_server()
    with open(CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rb.log(f"已写入 {CSV}")
    append_report(rows)


def append_report(rows):
    L = ["", f"## 分档上下文 {CONC} 并发（2026-10-05 增补，重点：内存/显存）", "",
         f"- 方法：与 2 并发版相同，仅并发数={CONC}；每路 prompt = 90% 上下文（nonce 开头，无前缀共享）。",
         "- 单路异常不拖垮整点；server_alive_after = 全部请求结束后 /health 是否仍正常。", "",
         "| 模型 | 上下文 | 成功/总数 | 每路 TTFT (s) | 每路 decode tok/s | 聚合 decode tok/s | 显存峰值 卡0/卡1 (GiB) | 内存峰值占用 (GiB) |",
         "|---|---:|---:|---|---|---:|---:|---:|"]
    for r in rows:
        ttft = " / ".join(str(x) for x in r["ttft_s"]) if r["ttft_s"] else "-"
        dec = " / ".join(str(x) for x in r["dec_tps_each"]) if r["dec_tps_each"] else "-"
        vp = r["vram_peak_mib"]
        vs = f"{vp[0]/1024:.1f} / {vp[1]/1024:.1f}" if vp and vp[0] else "-"
        L.append(f"| {r['model'].upper()} | {r['context']//1024}K | {r['ok_requests']}/{CONC} | {ttft} | {dec} "
                 f"| {r['agg_dec_tps']} | {vs} | {r['ram_used_peak_gib']} |")
    if any(r.get("errors") for r in rows):
        L += ["", "失败/异常明细："]
        for r in rows:
            if r.get("errors"):
                L.append(f"- {r['model'].upper()} @{r['context']//1024}K: {r['errors']}")
    L += ["", "原始数据：concurrency_ctx4_results.csv", ""]
    md = bc.ROOT / "BENCHMARK.md"
    md.write_text(md.read_text().rstrip("\n") + "\n" + "\n".join(L))
    rb.log(f"已追加 {md}")


if __name__ == "__main__":
    main()
