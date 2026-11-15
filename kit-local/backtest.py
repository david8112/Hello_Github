#!/usr/bin/env python
"""在历史数据上做「样本外」滚动检验：每隔 Lh 根取一个锚点（窗口互不重叠），用锚点之前的数据预测，
再和锚点之后的真实走势对比。step_market_284M 的训练截止 2025-12-31，所以 --start 2026-01-01 之后都是样本外。

  python backtest.py --market crypto --code BTCUSDT --scale 1d --file data_crypto/BTCUSDT_1d.csv --start 2026-01-01

统计内容：
- 方向：预测中位数的涨跌方向与实际是否一致（和「全猜涨」「延续前一窗口趋势」两个简单基准比较）
- 区间：真实收盘落在 5%-95% 预测区间内的比例（逐根，以及窗口终点）
- 校准：实际终值在 K 条路径中的分位（理想情况下应均匀分布）
- 波动率：预测波动率与实际波动率的秩相关（和「用前一窗口的实际波动率来猜」的基准比较）

仅供研究学习，不构成投资建议。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import forecast as F


def spearman(a, b) -> float:
    a, b = pd.Series(a), pd.Series(b)
    return float(a.rank().corr(b.rank())) if len(a) > 2 else float("nan")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="KiT 样本外滚动检验")
    ap.add_argument("--code", required=True)
    ap.add_argument("--scale", required=True, choices=F.SCALES)
    ap.add_argument("--market", default="ashare", choices=F.MARKETS)
    ap.add_argument("--file", type=Path, default=None)
    ap.add_argument("--open-time", action="store_true")
    ap.add_argument("--start", default="2026-01-01", help="第一个锚点不早于这个时间（默认训练截止之后）")
    ap.add_argument("--end", default=None, help="最后一个窗口的终点不晚于这个时间（默认数据末尾）")
    ap.add_argument("--paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=Path, default=F.DEFAULT_CKPT)
    ap.add_argument("--data-root", type=Path, default=F.KIT_ROOT / "data")
    ap.add_argument("--device", default="cuda" if F.torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("fp32", "bf16"), default=None,
                    help="默认 CUDA 上 bf16、CPU 上 fp32；CPU 支持 AMX/AVX512-BF16 时可用 bf16 提速")
    ap.add_argument("--out", type=Path, default=F.HERE / "outputs")
    args = ap.parse_args()

    market, scale, code = args.market, args.scale, args.code.strip().upper()
    path = args.file or (args.data_root / "ashare" / scale / f"{code}.parquet")
    df = F.load_bars(path, scale, market, open_time=args.open_time)
    ts = pd.DatetimeIndex(df["datetime"])
    tdays = F.build_calendar(market, args.data_root, None, ts)
    model, cfg, instruments, stats = F.load_bundle(args.ckpt, args.data_root, args.seed)
    device = F.torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    model.to(device).eval()
    Lc, Lh = cfg.data.window_for(scale)

    start = pd.Timestamp(args.start) - (F.DISPLAY_OFFSET if market == "crypto" else pd.Timedelta(0))   # 北京时间 -> 内部时间
    i0 = max(int(ts.searchsorted(start, side="left")), Lc)   # 第一个锚点 = 收盘时间 >= start 的第一根
    end_ts = None if args.end is None else pd.Timestamp(args.end) - (F.DISPLAY_OFFSET if market == "crypto" else pd.Timedelta(0))
    last = len(ts) if end_ts is None else int(ts.searchsorted(end_ts, side="right"))
    idx = list(range(i0, last - Lh, Lh))           # 锚点 = 第 i 根（含），之后要有 Lh 根真实数据
    if not idx:
        raise SystemExit("区间内没有完整的检验窗口")
    print(f"{code} {scale} ({market}): {len(idx)} 个不重叠窗口，锚点 {F.to_display(ts[idx[0]], market)} .. "
          f"{F.to_display(ts[idx[-1]], market)}{F.tz_note(market)}，"
          f"每个窗口预测 {Lh} 根，K={args.paths}, steps={args.steps}, device={device}, precision={precision}", flush=True)

    rows = []
    for j, i in enumerate(idx):
        r = F.forecast_window(model, cfg, df, ts[i], code=code, scale=scale, market=market, instruments=instruments,
                              stats=stats, tdays=tdays, paths=args.paths, steps=args.steps, seed=args.seed + j,
                              device=device, precision=precision, quiet=True)
        pc, term, q = r["prev_close"], r["term"], r["q"]
        act_close = r["fut_real"]["close"].to_numpy()
        act = float(np.log(act_close[-1] / pc))
        act_lr = np.diff(np.log(np.concatenate([[pc], act_close])))
        prev = df["close"].to_numpy()[i - Lh:i + 1]                      # 锚点之前 Lh 根（含锚点）
        prev_lr = np.diff(np.log(prev))
        rows.append(dict(
            anchor=r["anchor"], prev_close=pc, actual=act, pred_median=float(np.median(term)),
            pred_mean=float(term.mean()), q05=float(np.quantile(term, .05)), q95=float(np.quantile(term, .95)),
            p_up=float((term > 0).mean()), actual_pctile=float((term <= act).mean()),
            bar_coverage=float(np.mean((act_close >= q[0]) & (act_close <= q[4]))),
            end_in_band=bool(np.quantile(term, .05) <= act <= np.quantile(term, .95)),
            pred_vol=r["pred_vol"], real_vol=float(np.std(act_lr, ddof=1)),
            prev_vol=float(np.std(prev_lr, ddof=1)), prev_ret=float(np.log(prev[-1] / prev[0])),
            seconds=round(r["seconds"], 1)))
        x = rows[-1]
        print(f"[{j + 1}/{len(idx)}] 锚点 {F.to_display(x['anchor'], market)}  预测中位 {F.pct(x['pred_median'])} "
              f"[{F.pct(x['q05'])}, {F.pct(x['q95'])}]  P(涨)={x['p_up']:.0%}  实际 {F.pct(x['actual'])}  "
              f"逐根覆盖 {x['bar_coverage']:.0%}  {x['seconds']:.0f}s", flush=True)

    t = pd.DataFrame(rows)
    if market == "crypto":                                           # 第一列显示北京时间，另存 UTC
        t.insert(0, "anchor_utc8", F.to_display(t["anchor"], market))
        t = t.rename(columns={"anchor": "anchor_utc"})
        t["anchor"] = t["anchor_utc"]
    hit = np.sign(t["pred_median"]) == np.sign(t["actual"])
    up_rate = float((t["actual"] > 0).mean())
    mom_hit = np.sign(t["prev_ret"]) == np.sign(t["actual"])
    summary = {
        "windows": len(t),
        "direction_hit_KiT": float(hit.mean()),
        "direction_hit_always_up": up_rate,
        "direction_hit_momentum": float(mom_hit.mean()),
        "spearman_pred_vs_actual_return": spearman(t["pred_median"], t["actual"]),
        "bar_coverage_5_95_mean": float(t["bar_coverage"].mean()),
        "end_in_5_95_rate": float(t["end_in_band"].mean()),
        "actual_pctile_mean": float(t["actual_pctile"].mean()),
        "spearman_vol_KiT": spearman(t["pred_vol"], t["real_vol"]),
        "spearman_vol_prev_window": spearman(t["prev_vol"], t["real_vol"]),
        "mean_pred_vol": float(t["pred_vol"].mean()), "mean_real_vol": float(t["real_vol"].mean()),
    }
    a0, a1 = F.to_display(t["anchor"].iloc[0], market), F.to_display(t["anchor"].iloc[-1], market)
    tag = f"backtest_{code}_{scale}_{a0:%Y%m%d}_{a1:%Y%m%d}_{precision}"
    args.out.mkdir(parents=True, exist_ok=True)
    t.drop(columns=["anchor"] if "anchor_utc" in t else []).to_csv(args.out / f"{tag}.csv", index=False)
    pd.Series(summary).to_json(args.out / f"{tag}_summary.json", indent=2, force_ascii=False)

    print("\n===== 汇总 =====")
    print(f"窗口数 {summary['windows']}（每个窗口 {Lh} 根 {scale}）")
    print(f"方向命中率  KiT {summary['direction_hit_KiT']:.0%}  |  全猜涨 {up_rate:.0%}  |  延续前一窗口趋势 {summary['direction_hit_momentum']:.0%}")
    print(f"预测收益与实际收益的秩相关 {summary['spearman_pred_vs_actual_return']:+.2f}")
    print(f"5%-95% 区间：逐根覆盖率 {summary['bar_coverage_5_95_mean']:.0%}，窗口终点落在区间内 {summary['end_in_5_95_rate']:.0%}（理想约 90%）")
    print(f"实际终值在预测路径中的平均分位 {summary['actual_pctile_mean']:.2f}（理想 0.5；偏低=模型偏乐观，偏高=偏悲观）")
    print(f"波动率秩相关  KiT {summary['spearman_vol_KiT']:+.2f}  |  用前一窗口波动率 {summary['spearman_vol_prev_window']:+.2f}"
          f"   （平均预测波动 {summary['mean_pred_vol']:.2%} / 实际 {summary['mean_real_vol']:.2%} 每根）")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(t))
    e = np.expm1
    ax.vlines(x, e(t["q05"]) * 100, e(t["q95"]) * 100, color="#e8743b", lw=6, alpha=0.35, label="pred 5-95%")
    ax.scatter(x, e(t["pred_median"]) * 100, color="#e8743b", marker="_", s=300, lw=2, label="pred median")
    ax.scatter(x, e(t["actual"]) * 100, color="#1f4e79", zorder=3, s=28, label="actual")
    ax.axhline(0, color="k", lw=0.8)
    lab = [F.to_display(a, market).strftime("%m-%d %H:%M" if market == "crypto" else "%m-%d") for a in t["anchor"]]
    ax.set_xticks(x, lab, fontsize=7)
    ax.set_xlabel("forecast start (last bar close" + (f", {F.DISPLAY_TZ})" if market == "crypto" else ")"), fontsize=8)
    ax.set_ylabel(f"return over next {Lh} bars (%)")
    ax.set_title(f"KiT out-of-sample {code} {scale}: direction hit {summary['direction_hit_KiT']:.0%} "
                 f"(always-up {up_rate:.0%}), end-in-band {summary['end_in_5_95_rate']:.0%}, "
                 f"vol rank-corr {summary['spearman_vol_KiT']:+.2f}", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(args.out / f"{tag}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"输出: {args.out / tag}.{{csv,png}}, {tag}_summary.json")


if __name__ == "__main__":
    main()
