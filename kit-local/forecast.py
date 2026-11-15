#!/usr/bin/env python
"""用 KiT 预测「真正的未来」K 线（不需要锚点之后的真实数据）。

官方 eval/infer_for_pred.py 是回测工具：它要求锚点之后已经有 Lh 根真实 K 线，
所以只能在历史数据里「事后核对」。本脚本复用 KiT 的推理 API，但只用历史：
读入一个 OHLCV 文件，从最后一根（或 --anchor 指定的那根）K 线往后预测 Lh 根，
输出 16 条采样路径的分位数区间、上涨概率、图和 CSV。
如果文件里锚点之后恰好还有真实 K 线，会顺便叠加到图上做对照。

用法:
  # 1) 用官方示例数据（复现视频里的案例，锚点之后的真实走势会叠加在图上）
  python forecast.py --code 300260.SZ --scale 30m --anchor 2026-01-26

  # 2) 用你自己的数据（CSV/parquet，列: datetime, open, high, low, close, volume；
  #    也认 akshare 的中文列名 日期/时间, 开盘, 最高, 最低, 收盘, 成交量）
  python forecast.py --file my_600519_1d.csv --code 600519.SH --scale 1d

只支持 A 股（market=ashare）。每个周期需要的历史长度由模型固定：
  1m 1200 根 / 5m 960 / 15m 800 / 30m 640 / 1h 400 / 2h 360 / 1d 250
另外建议再多给约 60 个交易日用于成交量 EMA 预热。

仅供研究学习，不构成投资建议。
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KIT_ROOT = Path(os.environ.get("KIT_ROOT", HERE / "KiT")).resolve()
if not (KIT_ROOT / "kit_v1").is_dir():
    raise SystemExit(f"找不到 KiT 代码目录: {KIT_ROOT}（先运行 setup.sh，或设置环境变量 KIT_ROOT）")
sys.path.insert(0, str(KIT_ROOT))

import torch  # noqa: E402

from kit_v1.data.calendar import (SESSION_TEMPLATES, _daily_grid_minutes,  # noqa: E402
                                  bars_per_day, load_trading_days)
from kit_v1.data.normalize import load_stats  # noqa: E402
from kit_v1.infer import (denormalize_paths, flow_sample,  # noqa: E402
                          load_model_and_config, paths_to_ohlcv,
                          prepare_inference_window)

MARKET = "ashare"
SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
COLUMN_ALIASES = {
    "datetime": "datetime", "date": "datetime", "time": "datetime", "timestamp": "datetime",
    "日期": "datetime", "时间": "datetime", "日期时间": "datetime",
    "open": "open", "开盘": "open", "开盘价": "open",
    "high": "high", "最高": "high", "最高价": "high",
    "low": "low", "最低": "low", "最低价": "low",
    "close": "close", "收盘": "close", "收盘价": "close",
    "volume": "volume", "vol": "volume", "成交量": "volume",
}


def load_bars(path: Path, scale: str) -> pd.DataFrame:
    """读 CSV/parquet，统一列名，按时间排序，并检查 K 线时间戳落在 A 股交易时段网格上。"""
    if not path.is_file():
        raise SystemExit(f"数据文件不存在: {path}")
    df = pd.read_parquet(path) if path.suffix.lower() in (".parquet", ".pq") else pd.read_csv(path)
    df = df.rename(columns={c: COLUMN_ALIASES.get(str(c).strip().lower(), COLUMN_ALIASES.get(str(c).strip(), c))
                            for c in df.columns})
    missing = [c for c in ("datetime", "open", "high", "low", "close", "volume") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} 缺少列 {missing}；现有列: {list(df.columns)}")
    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    df = df.dropna().sort_values("datetime").drop_duplicates("datetime", keep="last")
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        warnings.warn(f"丢弃 {int(bad.sum())} 根价格<=0 的 K 线（停牌/脏数据）", stacklevel=2)
        df = df.loc[~bad]
    df = df.reset_index(drop=True)

    ts = pd.DatetimeIndex(df["datetime"])
    if scale == "1d":
        # KiT 约定日线时间戳是当天收盘 15:00；只给了日期的数据补成 15:00
        close_min = SESSION_TEMPLATES[MARKET][-1][1]
        df["datetime"] = ts.normalize() + pd.Timedelta(minutes=close_min)
    else:
        # 分钟线时间戳必须是 K 线「结束时间」，例如 5m 第一根是 09:35，1h 是 10:30/11:30/14:00/15:00
        grid = set(_daily_grid_minutes(MARKET, scale))
        mins = ts.hour * 60 + ts.minute
        off = ~np.isin(mins, list(grid))
        if off.any():
            examples = ", ".join(str(t) for t in ts[off][:3])
            raise SystemExit(
                f"{int(off.sum())} 根 {scale} K 线的时间不在 A 股交易时段网格上（例如 {examples}）。"
                f"KiT 要求时间戳为 K 线结束时间，合法的时分: "
                f"{', '.join(f'{m // 60:02d}:{m % 60:02d}' for m in sorted(grid)[:6])} ...")
    return df


def build_calendar(data_root: Path, extra_path: Path | None, hist_ts: pd.DatetimeIndex) -> np.ndarray | None:
    """交易日历 = 官方 trade_dates.csv ∪ 你自己的日历 ∪ 历史数据里出现过的日期。"""
    parts = []
    for p in (data_root / "trade_dates.csv", extra_path):
        if p is not None:
            td = load_trading_days(p)
            if td is not None:
                parts.append(td)
    parts.append(np.unique(hist_ts.normalize().to_numpy().astype("datetime64[D]")))
    return np.unique(np.concatenate(parts)) if parts else None


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="KiT：从历史 K 线预测未来走势（A 股）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--code", required=True, help="股票代码，如 600519.SH / 300750.SZ（决定使用哪个个股 embedding）")
    ap.add_argument("--scale", required=True, choices=SCALES)
    ap.add_argument("--file", type=Path, default=None,
                    help="你的 OHLCV 文件（CSV/parquet）；不填则读 KiT/data/ashare/<scale>/<code>.parquet 示例数据")
    ap.add_argument("--anchor", default=None,
                    help="从哪根 K 线之后开始预测；默认用文件最后一根。只写日期 = 当天收盘")
    ap.add_argument("--ckpt", type=Path, default=KIT_ROOT / "ckpt" / "step_market_284M" / "step_market_284M.pt")
    ap.add_argument("--data-root", type=Path, default=KIT_ROOT / "data", help="放 trade_dates.csv / instruments.json / stats.json 的目录")
    ap.add_argument("--calendar", type=Path, default=None,
                    help="额外的交易日历 CSV（第一列 YYYYMMDD）。官方日历只到 2026-04-10，预测更晚的日期时建议提供")
    ap.add_argument("--paths", type=int, default=16, help="采样路径数 K（官方默认 16）")
    ap.add_argument("--steps", type=int, default=32, help="流匹配积分步数（官方默认 32）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None,
                    help="默认 CUDA 上 bf16、CPU 上 fp32（与官方一致）")
    ap.add_argument("--ctx-show", type=int, default=80, help="图里显示多少根历史 K 线")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    code, scale = args.code.strip().upper(), args.scale
    path = args.file or (args.data_root / MARKET / scale / f"{code}.parquet")
    df = load_bars(path, scale)

    # 个股 embedding / 归一化统计：优先用与 checkpoint 同捆绑的元数据（覆盖 3914 只 A 股），否则用 data-root 的
    bundle = args.ckpt.parent
    inst_path = bundle / "instruments.json" if (bundle / "instruments.json").is_file() else args.data_root / "instruments.json"
    stats_path = bundle / "stats.json" if (bundle / "stats.json").is_file() else args.data_root / "stats.json"
    instruments = json.loads(inst_path.read_text(encoding="utf-8"))
    stats = load_stats(stats_path)
    inst = instruments.get(code)
    if inst is None:
        print(f"[warn] {code} 不在 {inst_path.name} 的 {len(instruments)} 个品种里，使用通用(UNK)个股 embedding，效果可能变差")
        inst = {"sector": 0, "vocab_index": 0}

    model, cfg = load_model_and_config(str(args.ckpt), "tiny", args.seed)
    Lc, Lh = cfg.data.window_for(scale)

    ts = pd.DatetimeIndex(df["datetime"])
    if args.anchor:
        a = pd.Timestamp(args.anchor)
        date_only = a == a.normalize() and len(args.anchor.strip()) <= 10
        end = a + pd.Timedelta(hours=23, minutes=59, seconds=59) if date_only else a
    else:
        end = ts[-1]
    n_hist = int(ts.searchsorted(end, side="right"))
    if n_hist < Lc:
        raise SystemExit(f"{scale} 需要至少 {Lc} 根历史 K 线，锚点 {end} 之前只有 {n_hist} 根（数据范围 {ts[0]} .. {ts[-1]}）")
    warm = int(math.ceil(12.0 * cfg.data.vol_ema_half_life_days * bars_per_day(MARKET, scale)))
    if n_hist < Lc + warm:
        print(f"[warn] 锚点前只有 {n_hist} 根，建议 >= {Lc + warm} 根（多出的用于成交量 EMA 预热）")
    hist = df.iloc[max(0, n_hist - (Lc + warm + 8)):n_hist].reset_index(drop=True)
    fut_real = df.iloc[n_hist:n_hist + Lh].reset_index(drop=True)   # 可能为空：真正的未来

    tdays = build_calendar(args.data_root, args.calendar, ts)
    device = torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    model.to(device).eval()

    win = prepare_inference_window(
        hist, MARKET, scale, Lc=Lc, Lh=Lh, stats=stats, soft_clip_c=cfg.data.soft_clip_c,
        half_life_days=cfg.data.vol_ema_half_life_days, end_ts=end,
        sector_id=int(inst.get("sector", 0)), inst_id=int(inst.get("vocab_index", 0)), trading_days=tdays)
    fut_ts = pd.DatetimeIndex(win["future_timestamps"])
    cal_end = pd.Timestamp(tdays[-1]) if tdays is not None else None
    if cal_end is not None and fut_ts[-1].normalize() > cal_end:
        print(f"[warn] 预测区间延伸到 {fut_ts[-1].date()}，超出交易日历（截至 {cal_end.date()}）："
              f"这部分只跳过周末、不跳节假日。可用 --calendar 提供最新交易日历")

    print(f"{code} {scale}: 历史 {win['ctx_timestamps'][0]} .. {win['ctx_timestamps'][-1]} ({Lc} 根) -> "
          f"预测 {fut_ts[0]} .. {fut_ts[-1]} ({Lh} 根)  device={device} precision={precision} K={args.paths}",
          flush=True)
    t0 = pd.Timestamp.now()
    chunk = 16 if device.type != "cuda" else 48
    paths = flow_sample(model, win["x_ctx"], win["cond"], Lh=Lh, steps=args.steps, solver="euler",
                        n_paths=args.paths, generator=torch.Generator().manual_seed(args.seed),
                        device=device, chunk_size=chunk, precision=precision)
    feats = denormalize_paths(paths, MARKET, scale, stats, c=cfg.data.soft_clip_c)[0]
    pred = {k: np.asarray(v) for k, v in
            paths_to_ohlcv(feats, win["prev_close"], win["prev_ema"], win["half_life_bars"]).items()}
    secs = (pd.Timestamp.now() - t0).total_seconds()

    prev_close = win["prev_close"]
    term = np.log(pred["close"][:, -1] / prev_close)
    k_rep = int(np.argsort(term)[len(term) // 2])
    q = np.quantile(pred["close"], [.05, .25, .5, .75, .95], axis=0)
    pct = lambda x: f"{np.expm1(x):+.2%}"  # noqa: E731  对数收益 -> 简单收益
    print(f"  最后收盘 {prev_close:.3f}；{Lh} 根后收盘的预测（{args.paths} 条路径）:")
    print(f"    中位数 {pct(np.median(term))}  均值 {pct(term.mean())}  "
          f"5%-95% 区间 [{pct(np.quantile(term, .05))}, {pct(np.quantile(term, .95))}]  "
          f"上涨路径占比 {(term > 0).mean():.0%}   用时 {secs:.0f}s")

    tag = f"{code}_{scale}_{pd.Timestamp(win['ctx_timestamps'][-1]).strftime('%Y%m%d_%H%M')}"
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"datetime": fut_ts, "close_q05": q[0], "close_q25": q[1], "close_q50": q[2],
                          "close_q75": q[3], "close_q95": q[4],
                          **{f"rep_{k}": pred[k][k_rep] for k in ("open", "high", "low", "close", "volume")}})
    if len(fut_real):
        n = len(fut_real)
        table.loc[:n - 1, "actual_close"] = fut_real["close"].to_numpy()
        act = float(np.log(fut_real["close"].iloc[-1] / prev_close))
        inside = np.mean((fut_real["close"].to_numpy() >= q[0][:n]) & (fut_real["close"].to_numpy() <= q[4][:n]))
        print(f"  [对照] 文件里锚点之后有 {n} 根真实 K 线：实际 {pct(act)}，"
              f"{inside:.0%} 的真实收盘落在 5%-95% 预测区间内")
    table.to_csv(args.out / f"{tag}.csv", index=False)
    np.savez_compressed(args.out / f"{tag}.npz", **{f"pred_{k}": v.astype(np.float32) for k, v in pred.items()},
                        fut_ts=fut_ts.to_numpy(dtype="datetime64[ns]"), prev_close=np.float64(prev_close), k_rep=k_rep)

    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        def candles(ax, xs, o, h, l, c, up, dn, w=0.6, alpha=1.0):
            for x, oi, hi, li, ci in zip(xs, o, h, l, c):
                col = up if ci >= oi else dn
                ax.plot([x, x], [li, hi], color=col, lw=0.8, alpha=alpha)
                ax.add_patch(Rectangle((x - w / 2, min(oi, ci)), w, max(abs(ci - oi), (hi - li) * 1e-3 + 1e-12),
                                       color=col, alpha=alpha))

        ctx = hist.tail(args.ctx_show)
        nc = len(ctx)
        xf = np.arange(nc, nc + Lh)
        fig, ax = plt.subplots(figsize=(12, 5.5))
        candles(ax, np.arange(nc), ctx["open"], ctx["high"], ctx["low"], ctx["close"], "#8a8a8a", "#bdbdbd")
        ax.fill_between(xf, q[0], q[4], color="#e8743b", alpha=0.15, label="pred close 5-95%")
        ax.fill_between(xf, q[1], q[3], color="#e8743b", alpha=0.25, label="pred close 25-75%")
        ax.plot(xf, q[2], color="#e8743b", ls="--", lw=1.2, label="pred close median")
        if len(fut_real):
            n = len(fut_real)
            candles(ax, xf[:n], fut_real["open"], fut_real["high"], fut_real["low"], fut_real["close"],
                    "#d43a3a", "#2f9e5e", w=0.5)
            ax.plot([], [], color="#d43a3a", lw=4, label="actual (red up / green down)")
        ax.axvline(nc - 0.5, color="k", ls=":", lw=1)
        all_ts = list(ctx["datetime"]) + list(fut_ts)
        ticks = np.linspace(0, len(all_ts) - 1, 7).astype(int)
        fmt = "%Y-%m-%d" if scale == "1d" else "%m-%d %H:%M"
        ax.set_xticks(ticks, [pd.Timestamp(all_ts[i]).strftime(fmt) for i in ticks], fontsize=8)
        ax.set_xlim(-1, nc + Lh)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"KiT {code} {scale}  anchor {win['ctx_timestamps'][-1]}  next {Lh} bars: "
                     f"median {pct(np.median(term))}  [5% {pct(np.quantile(term, .05))}, 95% {pct(np.quantile(term, .95))}]  "
                     f"P(up)={(term > 0).mean():.0%}", fontsize=10)
        fig.savefig(args.out / f"{tag}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    print(f"  输出: {args.out / tag}.{{png,csv,npz}}")


if __name__ == "__main__":
    main()
