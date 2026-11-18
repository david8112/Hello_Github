#!/usr/bin/env python
"""历史样本外预测：在过去一段时间里随机抽起点，用和实盘完全相同的方式各做一次预测，攒出更大的"预测 vs 实际"样本。

  python hist_eval.py --scales 1h,2h --start "2026-10-16" --end "2026-11-15 12:00"     # 时间是 UTC

- 起点取 UTC 偶数整点（15m/1h/2h 三个周期在这些时刻都正好有一根 K 线收盘），按固定随机种子打乱顺序依次跑，
  所以任何时候停下来，已完成的部分都是整段时间的一个随机样本。可随时中断，重跑会跳过已有结果。
- 每个起点只用起点及之前的 K 线（和实盘一样），结果存成 outputs/hist/<代码>_<周期>_<北京时间>.npz，格式同 forecast.py。
- 和看板的定时刷新错开：每次预测前，如果看板刷新正占着锁就等它跑完；自己预测时写 /tmp/kit-hist.busy，刷新脚本会等这一个预测结束。
之后用 `python eval_dataset.py --src outputs/hist --dst outputs/eval/hist --lag-min 5` 生成评估数据集。
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import forecast as F

HERE = Path(__file__).resolve().parent
LOCK = "/tmp/kit-dashboard.lock"
BUSY = Path("/tmp/kit-hist.busy")
CODES = ("SOLUSDT", "BTCUSDT")


def wait_for_dashboard() -> None:
    """看板刷新（dashboard_refresh.sh）持有锁时等待；拿得到锁就立刻放掉继续。"""
    while True:
        with open(LOCK, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(fh, fcntl.LOCK_UN)
                return
            except OSError:
                pass
        time.sleep(15)


def need_bars(cfg, scale: str, days: float) -> int:
    Lc, _ = cfg.data.window_for(scale)
    per_day = F.bars_per_day("crypto", scale)
    warm = int(np.ceil(12.0 * cfg.data.vol_ema_half_life_days * per_day))
    return int(Lc + warm + 8 + (days + 2) * per_day + 100)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scales", default="1h,2h")
    ap.add_argument("--start", required=True, help="最早起点（UTC）")
    ap.add_argument("--end", required=True, help="最晚起点（UTC），应早于实盘存档开始的时间")
    ap.add_argument("--max-anchors", type=int, default=10**6)
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="采样噪声种子（和实盘一致用 0）")
    ap.add_argument("--order-seed", type=int, default=20261118, help="起点打乱顺序的随机种子")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "hist")
    ap.add_argument("--threads", type=int, default=0, help="torch 线程数，0=默认")
    args = ap.parse_args()
    os.nice(5)
    if args.threads:
        torch.set_num_threads(args.threads)
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    args.out.mkdir(parents=True, exist_ok=True)

    model, cfg, instruments, stats = F.load_bundle(F.DEFAULT_CKPT, F.KIT_ROOT / "data", args.seed)
    model.to("cpu").eval()

    # 行情：需要"最早起点之前的上下文 + 预热"一直到现在，单独存一份长历史，不动实盘用的文件
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    data = {}
    for code in CODES:
        for scale in scales:
            path = HERE / "data_crypto" / f"hist_{code}_{scale}_okx_swap.csv"
            n = need_bars(cfg, scale, (now - start) / pd.Timedelta(days=1))
            fresh = path.exists() and (time.time() - path.stat().st_mtime) < 6 * 3600 and sum(1 for _ in open(path)) > n * 0.98
            if not fresh:
                print(f"下载 {code} {scale} {n} 根 …", flush=True)
                subprocess.run([sys.executable, str(HERE / "fetch_crypto.py"), "--exchange", "okx", "--swap", "--symbol", code,
                                "--interval", scale, "--bars", str(n), "--out", str(path)], check=True, stdout=subprocess.DEVNULL)
            df = F.load_bars(path, scale, "crypto")
            ts = pd.DatetimeIndex(df["datetime"])
            data[(code, scale)] = (df, F.build_calendar("crypto", F.KIT_ROOT / "data", None, ts))
            print(f"  {code} {scale}: {len(df)} 根 {ts[0]} .. {ts[-1]}", flush=True)

    anchors = pd.date_range(start.ceil("2h"), end.floor("2h"), freq="2h")
    anchors = anchors[np.random.default_rng(args.order_seed).permutation(len(anchors))][:args.max_anchors]
    todo = [(a, code, scale) for a in anchors for scale in scales for code in CODES]
    print(f"{len(anchors)} 个起点 × {len(scales)} 个周期 × {len(CODES)} 个品种 = {len(todo)} 个预测", flush=True)
    done = 0
    for a, code, scale in todo:
        tag = f"{code}_{scale}_{F.to_display(a, 'crypto'):%Y%m%d_%H%M}"
        out = args.out / f"{tag}.npz"
        if out.exists():
            done += 1
            continue
        df, tdays = data[(code, scale)]
        wait_for_dashboard()
        BUSY.write_text(str(os.getpid()))
        try:
            r = F.forecast_window(model, cfg, df, a, code=code, scale=scale, market="crypto", instruments=instruments,
                                  stats=stats, tdays=tdays, paths=args.paths, steps=args.steps, seed=args.seed,
                                  device=torch.device("cpu"), precision="bf16", quiet=True)
        finally:
            BUSY.unlink(missing_ok=True)
        if pd.Timestamp(r["anchor"]) != a:
            print(f"[skip] {tag}: 数据里没有 {a} 这根收盘（实际锚点 {r['anchor']}）", flush=True)
            continue
        tmp = args.out / f".{tag}.tmp.npz"
        np.savez_compressed(tmp, **{f"pred_{k}": v.astype(np.float32) for k, v in r["pred"].items()},
                            fut_ts=r["fut_ts"].to_numpy(dtype="datetime64[ns]"), prev_close=np.float64(r["prev_close"]),
                            k_rep=r["k_rep"], anchor=np.datetime64(r["anchor"], "ns"), code=code, scale=scale,
                            market="crypto", paths=args.paths, steps=args.steps, precision="bf16", ckpt=F.DEFAULT_CKPT.name,
                            made_at=np.datetime64(pd.Timestamp.now(tz="UTC").tz_localize(None), "ns"))
        tmp.rename(out)
        done += 1
        print(f"[{done}/{len(todo)}] {tag}  {r['seconds']:.0f}s  终点中位 {np.median(r['term']):+.2%}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
