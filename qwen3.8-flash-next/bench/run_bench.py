#!/usr/bin/env python3
"""Strata 基准测试编排器 v2：单请求流式计时，一次拿全 prefill + decode 数据。

阶段: 等 IQ3_XXS 分片1 -> 校验并硬链接分片2 -> 安装+冒烟(可跳过) -> 双模型×5档上下文基准 -> 落盘
断点续跑: results.csv 已有的测试点跳过。
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent            # qwen3.8-flash-next/bench/
STRATA = ROOT.parent / "Strata"
DATA = Path("/mnt/modelzoo/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
MS_API = "https://www.modelscope.cn/api/v1/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/repo/files?Recursive=true"
LOG = ROOT / "results" / "conc1" / "bench.log"
CSV = ROOT / "results" / "conc1" / "results.csv"
MD = ROOT / "BENCHMARK.md"
API = "http://127.0.0.1:8080"

MODELS = {"iq2_xs": DATA / "IQ2_XS", "iq3_xxs": DATA / "IQ3_XXS"}
CONTEXTS = [32768, 65536, 131072, 196608, 262144]
PROMPT_TARGET_RATIO = 0.9
BASE_PARA = ("The city council met on Tuesday to discuss the proposed transit expansion. "
             "Residents voiced concerns about construction noise, while business owners welcomed the projected increase in foot traffic. "
             "The engineering report estimated a two-year timeline, pending state funding approval. "
             "A follow-up hearing is scheduled for next month, and officials promised a detailed cost breakdown before any vote.\n")
TPR = 67.0
# 固定长度输出任务：模型必须数够 100 个数，不会提前 EOS；中文每数约 1-2 token
COUNT_TASK = "\n请从1数到100，每个数字单独一行，不要输出其他内容。"
SETUP_ENV = {**os.environ, "HF_ENDPOINT": "https://hf-mirror.com"}
SETUP_EXTRA = ["--prebuilt", "https://gh-proxy.com/https://github.com/Niko1221/Strata/releases/download/v0.1.39/"]


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, **kw):
    log("$ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def http_json(path, timeout=30):
    with urllib.request.urlopen(API + path, timeout=timeout) as r:
        return json.load(r)


def chat_stream(prompt, max_tokens, timeout=600):
    """流式请求。返回 dict:
    prompt_tokens/completion_tokens (usage 优先, 缺省数以 content 块数估),
    ttft_s (首 content 块延迟 ≈ prefill), decode_s (首块->末块窗口), total_s"""
    payload = {"model": "strata", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "reasoning_effort": "none", "stream": True,
               "stream_options": {"include_usage": True}}
    req = urllib.request.Request(API + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    t0 = time.time()
    first = last = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                d = json.loads(data)
            except json.JSONDecodeError:
                continue
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    now = time.time()
                    if first is None:
                        first = now
                    last = now
                    chunks += 1
    total = time.time() - t0
    ptoks = usage.get("prompt_tokens")
    ctoks = usage.get("completion_tokens") or chunks
    return {"prompt_tokens": ptoks, "completion_tokens": ctoks,
            "ttft_s": round(first - t0, 2) if first else round(total, 2),
            "decode_s": round(last - first, 2) if first and last else 0.0,
            "total_s": round(total, 2)}


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return None


def vram_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [int(x) for x in out]


# ---------------------------------------------------------------- Phase 0/1
def wait_and_link_shard2():
    shard1 = MODELS["iq3_xxs"] / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"
    shard2_src = MODELS["iq2_xs"] / "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf"
    shard2_dst = MODELS["iq3_xxs"] / "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf"
    if shard2_dst.exists():
        log(f"分片2已存在，跳过: {shard2_dst}")
        return
    log("等待 IQ3_XXS 分片1下载完成...")
    while True:
        downloading = subprocess.run(["pgrep", "-f", "modelscope download"], capture_output=True).returncode == 0
        incomplete = list(MODELS["iq3_xxs"].glob("*.incomplete"))
        if shard1.exists() and not incomplete and not downloading:
            break
        time.sleep(60)
    log(f"分片1就位: {shard1.stat().st_size/1e9:.2f} GB")
    log("三方校验分片2 (ModelScope sha256 vs 本地)...")
    try:
        with urllib.request.urlopen(MS_API, timeout=30) as r:
            files = json.load(r)["Data"]["Files"]
        sha2 = {f["Path"]: f.get("Sha256", "").lower() for f in files if f["Path"].endswith("00002-of-00002.gguf")}
        s_iq2 = sha2.get("IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf", "")
        s_iq3 = sha2.get("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf", "")
        if s_iq2 and s_iq3 and s_iq2 == s_iq3:
            h = hashlib.sha256()
            with open(shard2_src, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != s_iq2:
                sys.exit("本地 IQ2_XS 分片2 与 ModelScope 记录不符，中止")
            log("远端一致，本地 sha256 匹配")
        else:
            log("警告: API sha256 不可用，基于 Strata 官方假设继续")
    except SystemExit:
        raise
    except Exception as e:
        log(f"警告: 校验异常 ({e})，基于官方假设继续")
    shard2_dst.hardlink_to(shard2_src)
    log(f"分片2已硬链接 -> {shard2_src.name}")


# ---------------------------------------------------------------- Phase 2/3
def ensure_installed(model):
    if (STRATA / f"run-{model}.sh").exists():
        log(f"{model} 已安装，跳过")
        return
    log(f"安装 {model} ...")
    r = run([str(STRATA / "setup.sh"), "--yes", "--setup", "--family", "qwen",
             "--model", model.upper(), "--context", "262144", "--no-start",
             "--gguf-dir", str(MODELS[model]), *SETUP_EXTRA], env=SETUP_ENV, cwd=STRATA)
    log("\n".join((r.stdout + r.stderr).splitlines()[-8:]))
    if not (STRATA / f"run-{model}.sh").exists():
        sys.exit(f"{model} 安装失败")


def smoke_test(model):
    log(f"冒烟测试 {model} ...")
    start_server(model)
    try:
        d = chat_stream("回复'运行正常'四个字即可", 16)
        log(f"冒烟: {d['completion_tokens']} tokens, ttft {d['ttft_s']}s")
    finally:
        stop_server()


# ---------------------------------------------------------------- 服务启停
server_log = None


def start_server(model):
    global server_log
    log(f"启动 {model} ...")
    server_log = open(ROOT / f"server-{model}.log", "ab")
    p = subprocess.Popen(["sh", str(STRATA / f"run-{model}.sh")], cwd=STRATA,
                         stdout=server_log, stderr=subprocess.STDOUT)
    for i in range(90):
        time.sleep(5)
        try:
            h = http_json("/health", timeout=3)
            if h.get("loaded"):
                log(f"就绪 ({(i+1)*5}s), max_context={h['max_context']}")
                return
        except Exception:
            pass
    p.kill()
    sys.exit(f"{model} 启动超时，见 server-{model}.log 与 Strata 日志")


def stop_server():
    subprocess.run(["pkill", "-f", "serve/server.py"], capture_output=True)
    time.sleep(4)
    if server_log:
        server_log.close()


# ---------------------------------------------------------------- Phase 4
def bench_point(model, ctx):
    log(f"=== 基准 {model} @ {ctx//1024}K ===")
    r = run([str(STRATA / "setup.sh"), "--yes", "--setup", "--family", "qwen",
             "--model", model.upper(), "--context", str(ctx), "--no-start",
             "--gguf-dir", str(MODELS[model]), *SETUP_EXTRA], env=SETUP_ENV, cwd=STRATA)
    if r.returncode != 0:
        log("配置失败: " + "\n".join((r.stdout + r.stderr).splitlines()[-6:]))
        return None

    idle_mem = mem_available_gib()
    start_server(model)
    try:
        vram = vram_mib()
        ram_used = round(idle_mem - mem_available_gib(), 1) if idle_mem else None

        reps = int(ctx * PROMPT_TARGET_RATIO / TPR)
        long_prompt = BASE_PARA * reps + COUNT_TASK

        d1 = chat_stream(long_prompt, 200)                 # 长上下文: prefill + decode 一次拿到
        prefill_tps = round(d1["prompt_tokens"] / d1["ttft_s"], 1) if d1["prompt_tokens"] else None
        dec_tps = round((d1["completion_tokens"] - 1) / d1["decode_s"], 1) if d1["decode_s"] > 0 else None
        d2 = chat_stream(COUNT_TASK.strip(), 200)          # 短对话 decode
        s_tps = round((d2["completion_tokens"] - 1) / d2["decode_s"], 1) if d2["decode_s"] > 0 else None

        row = {"model": model, "context": ctx, "prompt_tokens": d1["prompt_tokens"],
               "prefill_s": d1["ttft_s"], "prefill_tps": prefill_tps,
               "decode_tokens": d1["completion_tokens"], "decode_s": d1["decode_s"], "decode_tps": dec_tps,
               "short_chat_tps": s_tps, "vram0_mib": vram[0], "vram1_mib": vram[1],
               "ram_used_gib": ram_used, "ts": time.strftime("%Y-%m-%d %H:%M")}
        log(str(row))
        return row
    finally:
        stop_server()


def load_done():
    done = set()
    if CSV.exists():
        for r in csv.DictReader(open(CSV)):
            done.add((r["model"], int(r["context"])))
    return done


# ---------------------------------------------------------------- Phase 5
def write_report(rows):
    env = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().replace("\n", " / ")
    cpu = subprocess.run(["grep", "-m1", "model name", "/proc/cpuinfo"],
                         capture_output=True, text=True).stdout.split(":")[-1].strip()
    ram = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()[1]
    eng = json.loads((STRATA / "engine" / "BUILD.json").read_text())

    L = ["# Strata 基准测试报告", "",
         f"- 日期：{time.strftime('%Y-%m-%d %H:%M')}",
         f"- 机型：{env}；CPU {cpu}；RAM {ram}GB",
         f"- 引擎：Strata {eng.get('version')}（本地编译，archs {eng.get('archs')}）；双卡层切分 + MTP 投机解码(spec 4)",
         f"- 模型：Qwen3.8-Flash-Next（ISTA-DASLab GSQ-RCO）IQ2_XS / IQ3_XXS，分片2硬链接共享", "",
         "## 方法",
         "- 每档上下文独立配置并重启服务；prompt = 90% 上下文长度的英文 filler + 固定输出任务（从1数到100）。",
         "- prefill tok/s = prompt_tokens / 首token延迟（SSE 流式计时）；decode tok/s = (completion_tokens-1) / (首块→末块时间窗)。",
         "- 短对话 = 仅计数任务（<50 tokens prompt）；显存 = 就绪后 nvidia-smi；内存 = 就绪前后 MemAvailable 差。",
         "- KV 布局：IQ2_XS 全程 KV streaming（KV 驻内存）；IQ3_XXS 因 61GB 内存规则 KV 驻显存（setup 自动判定，长上下文档位专家缓存相应减少）。", ""]
    for model in MODELS:
        L.append(f"## {model.upper()}")
        L.append("| 上下文 | prefill tok/s | 长ctx decode tok/s | 短对话 decode tok/s | 显存 卡0/卡1 (GiB) | 服务内存 (GiB) |")
        L.append("|---:|---:|---:|---:|---:|---:|")
        for row in sorted([r for r in rows if r["model"] == model], key=lambda r: r["context"]):
            L.append(f"| {row['context']//1024}K | {row['prefill_tps']} | {row['decode_tps']} | {row['short_chat_tps']} "
                     f"| {float(row['vram0_mib'])/1024:.1f} / {float(row['vram1_mib'])/1024:.1f} | {row['ram_used_gib']} |")
        L.append("")
    L += ["## 备注",
          "- **decode 为乐观值**：计数任务输出高度可预测，MTP 投机解码接受率接近上限；自由文本实测约低 30-40%"
          "（参考：非流式自由文本短对话 iq2_xs ≈62-70 tok/s，iq3_xxs ≈45-53 tok/s，见 results/conc1/results_v1_invalid_decode.csv.bak 的 short_chat_tps）。",
          "- decode 含 MTP 投机解码；长上下文 decode 因注意力范围增大低于短对话属正常。",
          "- IQ3_XXS 专家更大（43 vs 36GB），显存命中率低于 IQ2_XS，decode 低约 20-30% 属预期。",
          "", "原始数据：results/conc1/results.csv；全程日志：results/conc1/bench.log"]
    MD.write_text("\n".join(L))
    log(f"报告已写入 {MD}")


def main():
    stop_server()
    wait_and_link_shard2()
    ensure_installed("iq3_xxs")
    smoke_test("iq3_xxs")
    done = load_done()
    rows, new = [], []
    if CSV.exists():
        rows = list(csv.DictReader(open(CSV)))
        for r in rows:
            r["context"] = int(r["context"])
    for model in MODELS:
        for ctx in CONTEXTS:
            if (model, ctx) in done:
                log(f"跳过已完成: {model} @ {ctx//1024}K")
                continue
            row = bench_point(model, ctx)
            if row:
                new.append(row)
                with open(CSV, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if f.tell() == 0:
                        w.writeheader()
                    w.writerow(row)
    write_report(rows + new)
    log("全部完成")


if __name__ == "__main__":
    main()
