#!/usr/bin/env python
"""用 KiT 预测「真正的未来」K 线（不需要锚点之后的真实数据）。支持 A 股和加密货币。

官方 eval/infer_for_pred.py 是回测工具：它要求锚点之后已经有 Lh 根真实 K 线，
所以只能在历史数据里「事后核对」。本脚本复用 KiT 的推理 API，但只用历史：
读入一个 OHLCV 文件，从最后一根（或 --anchor 指定的那根）K 线往后预测 Lh 根，
输出 K 条采样路径的分位数区间、上涨概率、图和 CSV。
如果文件里锚点之后恰好还有真实 K 线，会顺便叠加到图上做对照。

用法:
  # A 股：官方示例数据（复现视频里的案例，锚点之后的真实走势会叠加在图上）
  python forecast.py --code 300260.SZ --scale 30m --anchor 2026-01-26

  # A 股：自己的数据（CSV/parquet，列: datetime, open, high, low, close, volume；
  #       也认 akshare 的中文列名 日期/时间, 开盘, 最高, 最低, 收盘, 成交量）
  python forecast.py --file my_600519_1d.csv --code 600519.SH --scale 1d

  # 加密货币：先用 fetch_crypto.py 下载（币安或 OKX）。数据文件里是 UTC 收盘时间（模型训练约定），
  #          打印、图表和结果 CSV 一律显示北京时间（UTC+8）；--anchor 也按北京时间填写
  python fetch_crypto.py --symbol BTCUSDT --interval 1d          # 或 --exchange okx
  python forecast.py --market crypto --code BTCUSDT --scale 1d --file data_crypto/BTCUSDT_1d.csv

每个周期需要的历史长度由模型固定：
  1m 1200 根 / 5m 960 / 15m 800 / 30m 640 / 1h 400 / 2h 360 / 1d 250
另外建议再多给约 5 天 x 12 的数据用于成交量 EMA 预热。

仅供研究学习，不构成投资建议。
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import contextlib
import json
import math
import sys
import time
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

from kit_v1.data.calendar import (SESSION_TEMPLATES, TIMESCALE_MINUTES,  # noqa: E402
                                  _daily_grid_minutes, bars_per_day,
                                  load_trading_days)
from kit_v1.data.normalize import load_stats  # noqa: E402
from kit_v1.infer import (denormalize_paths, flow_sample,  # noqa: E402
                          load_model_and_config, paths_to_ohlcv,
                          prepare_inference_window)

MARKETS = ("ashare", "crypto")
SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
DEFAULT_CKPT = KIT_ROOT / "ckpt" / "step_market_284M" / "step_market_284M.pt"
# 显示用的时区：数据和模型输入一律保持 UTC（训练约定），只有打印、图表、结果 CSV 换成北京时间
DISPLAY_OFFSET = pd.Timedelta(hours=8)
DISPLAY_TZ = "UTC+8"
COLUMN_ALIASES = {
    "datetime": "datetime", "date": "datetime", "time": "datetime", "timestamp": "datetime",
    "日期": "datetime", "时间": "datetime", "日期时间": "datetime",
    "open": "open", "开盘": "open", "开盘价": "open",
    "high": "high", "最高": "high", "最高价": "high",
    "low": "low", "最低": "low", "最低价": "low",
    "close": "close", "收盘": "close", "收盘价": "close",
    "volume": "volume", "vol": "volume", "成交量": "volume",
}


def to_display(ts, market: str):
    """内部时间 -> 显示时间。加密货币内部是 UTC，显示为 UTC+8；A 股数据本来就是北京时间，不变。"""
    return ts + DISPLAY_OFFSET if market == "crypto" else ts


def tz_note(market: str) -> str:
    return f"（北京时间 {DISPLAY_TZ}）" if market == "crypto" else ""


def pct(log_ret) -> str:
    """对数收益 -> 简单涨跌幅字符串。"""
    return f"{np.expm1(log_ret):+.2%}"


def load_bars(path: Path, scale: str, market: str = "ashare", open_time: bool = False) -> pd.DataFrame:
    """读 CSV/parquet，统一列名，按时间排序，并检查 K 线时间戳是否符合 KiT 约定。

    KiT 约定时间戳是 K 线「收盘时间」：A 股日线 = 当天 15:00；分钟线 = 该 K 线结束时刻；
    加密货币（UTC）日线 = 次日 00:00。open_time=True 表示文件里是开盘时间，自动加一个周期。
    """
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
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC" if market == "crypto" else "Asia/Shanghai").dt.tz_localize(None)
    if open_time:
        df["datetime"] += pd.Timedelta(minutes=TIMESCALE_MINUTES[scale])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float64)
    df = df.dropna().sort_values("datetime").drop_duplicates("datetime", keep="last")
    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        warnings.warn(f"丢弃 {int(bad.sum())} 根价格<=0 的 K 线（停牌/脏数据）", stacklevel=2)
        df = df.loc[~bad]
    df = df.reset_index(drop=True)

    ts = pd.DatetimeIndex(df["datetime"])
    if market == "ashare" and scale == "1d":
        # A 股日线：只给了日期的数据补成收盘 15:00
        df["datetime"] = ts.normalize() + pd.Timedelta(minutes=SESSION_TEMPLATES[market][-1][1])
        return df
    grid = set(_daily_grid_minutes(market, scale))
    mins = np.asarray(ts.hour * 60 + ts.minute)
    if market == "crypto":
        mins = np.where(mins == 0, 1440, mins)          # 00:00 收盘 = 前一天的最后一根
    off = ~np.isin(mins, list(grid))
    if off.any():
        examples = ", ".join(str(t) for t in ts[off][:3])
        hint = ("加密货币要求 UTC 收盘时间，日线是次日 00:00；如果文件里是开盘时间，加 --open-time"
                if market == "crypto" else
                "A 股分钟线要求 K 线结束时间，例如 5m 第一根是 09:35，1h 是 10:30/11:30/14:00/15:00")
        raise SystemExit(f"{int(off.sum())} 根 {scale} K 线的时间不符合 KiT 约定（例如 {examples}）。{hint}")
    return df


def build_calendar(market: str, data_root: Path, extra_path: Path | None, hist_ts: pd.DatetimeIndex) -> np.ndarray | None:
    """交易日历。A 股 = 官方 trade_dates.csv ∪ 自己的日历 ∪ 历史数据里出现过的日期；加密货币 = 每一天。"""
    if market == "crypto":
        start = hist_ts[0].normalize() - pd.Timedelta(days=1)
        return pd.date_range(start, hist_ts[-1].normalize() + pd.Timedelta(days=800), freq="D").to_numpy().astype("datetime64[D]")
    parts = []
    for p in (data_root / "trade_dates.csv", extra_path):
        if p is not None:
            td = load_trading_days(p)
            if td is not None:
                parts.append(td)
    parts.append(np.unique(hist_ts.normalize().to_numpy().astype("datetime64[D]")))
    return np.unique(np.concatenate(parts))


def anchor_end(text: str | None, market: str, ts: pd.DatetimeIndex) -> pd.Timestamp:
    """锚点（北京时间）-> 内部截止时间（含）。

    A 股：只写日期 = 当天收盘（当天 23:59:59 之前的最后一根）。
    加密货币：按北京时间填写，内部换算成 UTC；只写日期 = 北京时间当天 24:00 之前已收盘的最后一根
    （日线的话就是当天 08:00 收盘的那根，即 UTC 前一天的日线）。
    """
    if not text:
        return ts[-1]
    a = pd.Timestamp(text)
    date_only = a == a.normalize() and len(text.strip()) <= 10
    if market == "crypto":
        a = a - DISPLAY_OFFSET
        return a + pd.Timedelta(days=1) if date_only else a
    return a + pd.Timedelta(hours=23, minutes=59, seconds=59) if date_only else a


def load_bundle(ckpt: Path, data_root: Path, seed: int = 0):
    """模型 + 与 checkpoint 同捆绑的 instruments/stats（没有时退回 data_root 的）。"""
    b = ckpt.parent
    inst_path = b / "instruments.json" if (b / "instruments.json").is_file() else data_root / "instruments.json"
    stats_path = b / "stats.json" if (b / "stats.json").is_file() else data_root / "stats.json"
    instruments = json.loads(inst_path.read_text(encoding="utf-8"))
    stats = load_stats(stats_path)
    model, cfg = load_model_and_config(str(ckpt), "tiny", seed)
    return model, cfg, instruments, stats


def forecast_window(model, cfg, df: pd.DataFrame, end: pd.Timestamp, *, code: str, scale: str, market: str,
                    instruments: dict, stats: dict, tdays, paths: int = 16, steps: int = 32, seed: int = 0,
                    device: torch.device = torch.device("cpu"), precision: str = "fp32",
                    quiet: bool = False) -> dict:
    """对一个锚点做一次预测，返回预测路径、分位数，以及（如果有）锚点之后的真实 K 线。"""
    Lc, Lh = cfg.data.window_for(scale)
    inst = instruments.get(code)
    if inst is None:
        if not quiet:
            print(f"[warn] {code} 不在 instruments.json 的 {len(instruments)} 个品种里，使用通用(UNK)个股 embedding，效果可能变差")
        inst = {"sector": 0, "vocab_index": 0}
    elif inst.get("market") != market:
        raise SystemExit(f"{code} 在模型里属于 {inst.get('market')} 市场，但 --market 是 {market}")

    ts = pd.DatetimeIndex(df["datetime"])
    n_hist = int(ts.searchsorted(end, side="right"))
    if n_hist < Lc:
        raise SystemExit(f"{scale} 需要至少 {Lc} 根历史 K 线，锚点 {end} 之前只有 {n_hist} 根（数据范围 {ts[0]} .. {ts[-1]}）")
    warm = int(math.ceil(12.0 * cfg.data.vol_ema_half_life_days * bars_per_day(market, scale)))
    if n_hist < Lc + warm and not quiet:
        print(f"[warn] 锚点前只有 {n_hist} 根，建议 >= {Lc + warm} 根（多出的用于成交量 EMA 预热）")
    hist = df.iloc[max(0, n_hist - (Lc + warm + 8)):n_hist].reset_index(drop=True)
    fut_real = df.iloc[n_hist:n_hist + Lh].reset_index(drop=True)

    win = prepare_inference_window(
        hist, market, scale, Lc=Lc, Lh=Lh, stats=stats, soft_clip_c=cfg.data.soft_clip_c,
        half_life_days=cfg.data.vol_ema_half_life_days, end_ts=end,
        sector_id=int(inst.get("sector", 0)), inst_id=int(inst.get("vocab_index", 0)), trading_days=tdays)
    fut_ts = pd.DatetimeIndex(win["future_timestamps"])
    if len(fut_real) and not pd.DatetimeIndex(fut_real["datetime"]).equals(fut_ts[:len(fut_real)]):
        if not quiet:
            print("[warn] 锚点之后的真实 K 线时间与日历推算不一致（停牌/缺数据？），对照结果仅供参考")

    t0 = time.time()
    chunk = 48 if device.type == "cuda" else 16
    # KiT 只在 CUDA 上启用低精度；CPU 上指定 bf16 时在外层套 CPU autocast（有 AMX/AVX512-BF16 的 CPU 会快很多）
    cpu_amp = (torch.autocast("cpu", dtype=torch.bfloat16) if device.type == "cpu" and precision == "bf16"
               else contextlib.nullcontext())
    with cpu_amp:
        z = flow_sample(model, win["x_ctx"], win["cond"], Lh=Lh, steps=steps, solver="euler", n_paths=paths,
                        generator=torch.Generator().manual_seed(seed), device=device, chunk_size=chunk,
                        precision=precision)
    feats = denormalize_paths(z, market, scale, stats, c=cfg.data.soft_clip_c)[0]
    pred = {k: np.asarray(v) for k, v in
            paths_to_ohlcv(feats, win["prev_close"], win["prev_ema"], win["half_life_bars"]).items()}
    prev_close = float(win["prev_close"])
    term = np.log(pred["close"][:, -1] / prev_close)
    logret = np.diff(np.log(np.concatenate([np.full((pred["close"].shape[0], 1), prev_close), pred["close"]], 1)), axis=1)
    return dict(code=code, scale=scale, market=market, Lc=Lc, Lh=Lh, hist=hist, fut_real=fut_real,
                fut_ts=fut_ts, anchor=pd.Timestamp(win["ctx_timestamps"][-1]), prev_close=prev_close,
                pred=pred, term=term, k_rep=int(np.argsort(term)[len(term) // 2]),
                q=np.quantile(pred["close"], [.05, .25, .5, .75, .95], axis=0),
                pred_vol=float(np.std(logret, axis=1, ddof=1).mean()),
                seconds=time.time() - t0)


def bar_label(ts: pd.Timestamp, scale: str, market: str) -> str:
    """图上的时间标签（K 线收盘时间）。加密货币显示北京时间：UTC 日线在北京时间 08:00 收盘。"""
    if market == "crypto":
        return to_display(ts, market).strftime("%Y-%m-%d %H:%M" if scale == "1d" else "%m-%d %H:%M")
    return ts.strftime("%Y-%m-%d" if scale == "1d" else "%m-%d %H:%M")


def plot_result(res: dict, png: Path, ctx_show: int = 80) -> None:
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

    ctx, fut_real, q, term, Lh = res["hist"].tail(ctx_show), res["fut_real"], res["q"], res["term"], res["Lh"]
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
    all_ts = list(ctx["datetime"]) + list(res["fut_ts"])
    ticks = np.linspace(0, len(all_ts) - 1, 7).astype(int)
    ax.set_xticks(ticks, [bar_label(pd.Timestamp(all_ts[i]), res["scale"], res["market"]) for i in ticks], fontsize=8)
    ax.set_xlim(-1, nc + Lh)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    tz = f" ({DISPLAY_TZ})" if res["market"] == "crypto" else ""
    ax.set_title(f"KiT {res['code']} {res['scale']}  last bar closes {to_display(res['anchor'], res['market']):%Y-%m-%d %H:%M}{tz}  next {Lh} bars: "
                 f"median {pct(np.median(term))}  [5% {pct(np.quantile(term, .05))}, 95% {pct(np.quantile(term, .95))}]  "
                 f"P(up)={(term > 0).mean():.0%}  K={len(term)}", fontsize=10)
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="KiT：从历史 K 线预测未来走势（A 股 / 加密货币）",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--code", required=True, help="品种代码：A 股如 600519.SH；加密货币如 BTCUSDT（决定使用哪个个股 embedding）")
    ap.add_argument("--scale", required=True, choices=SCALES)
    ap.add_argument("--market", default="ashare", choices=MARKETS)
    ap.add_argument("--file", type=Path, default=None,
                    help="OHLCV 文件（CSV/parquet）；A 股不填则读 KiT/data/ashare/<scale>/<code>.parquet 示例数据")
    ap.add_argument("--open-time", action="store_true", help="文件里的时间戳是 K 线开盘时间（自动换成收盘时间）")
    ap.add_argument("--anchor", default=None,
                    help="从哪根 K 线之后开始预测（北京时间）；默认用文件最后一根。只写日期 = 当天收盘")
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data-root", type=Path, default=KIT_ROOT / "data", help="A 股的 trade_dates.csv 所在目录")
    ap.add_argument("--calendar", type=Path, default=None,
                    help="A 股额外的交易日历 CSV（第一列 YYYYMMDD）。官方日历只到 2026-04-10，预测更晚的日期时建议提供")
    ap.add_argument("--paths", type=int, default=16, help="采样路径数 K（官方默认 16，越多越稳、越慢）")
    ap.add_argument("--steps", type=int, default=32, help="流匹配积分步数（官方默认 32）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None,
                    help="默认 CUDA 上 bf16、CPU 上 fp32（与官方一致）。CPU 支持 AMX/AVX512-BF16 时可用 bf16 提速（fp16 仅 CUDA）")
    ap.add_argument("--ctx-show", type=int, default=80, help="图里显示多少根历史 K 线")
    ap.add_argument("--out", type=Path, default=HERE / "outputs")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    market, scale = args.market, args.scale
    code = args.code.strip().upper()
    if args.file is None and market != "ashare":
        raise SystemExit("加密货币需要用 --file 指定数据（可先运行 fetch_crypto.py 下载）")
    path = args.file or (args.data_root / "ashare" / scale / f"{code}.parquet")
    df = load_bars(path, scale, market, open_time=args.open_time)
    ts = pd.DatetimeIndex(df["datetime"])
    end = anchor_end(args.anchor, market, ts)
    tdays = build_calendar(market, args.data_root, args.calendar, ts)

    model, cfg, instruments, stats = load_bundle(args.ckpt, args.data_root, args.seed)
    device = torch.device(args.device)
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    model.to(device).eval()

    Lc, Lh = cfg.data.window_for(scale)
    print(f"{code} {scale} ({market}): 预测锚点 {to_display(end, market)}{tz_note(market)}之后 {Lh} 根  device={device} precision={precision} K={args.paths}",
          flush=True)
    res = forecast_window(model, cfg, df, end, code=code, scale=scale, market=market, instruments=instruments,
                          stats=stats, tdays=tdays, paths=args.paths, steps=args.steps, seed=args.seed,
                          device=device, precision=precision)
    fut_ts, q, term, pred, prev_close = res["fut_ts"], res["q"], res["term"], res["pred"], res["prev_close"]
    if market == "ashare" and fut_ts[-1].normalize() > pd.Timestamp(tdays[-1]):
        print(f"[warn] 预测区间延伸到 {fut_ts[-1].date()}，超出交易日历（截至 {pd.Timestamp(tdays[-1]).date()}）："
              f"这部分只跳过周末、不跳节假日。可用 --calendar 提供最新交易日历")

    print(f"  历史截至 {to_display(res['anchor'], market)}，预测 {to_display(fut_ts[0], market)} .. "
          f"{to_display(fut_ts[-1], market)}{tz_note(market)}，共 {Lh} 根（时间均为 K 线收盘时间）")
    print(f"  最后收盘 {prev_close:g}；{Lh} 根后收盘的预测（{args.paths} 条路径）:")
    print(f"    中位数 {pct(np.median(term))}  均值 {pct(term.mean())}  "
          f"5%-95% 区间 [{pct(np.quantile(term, .05))}, {pct(np.quantile(term, .95))}]  "
          f"上涨路径占比 {(term > 0).mean():.0%}   用时 {res['seconds']:.0f}s")

    tag = f"{code}_{scale}_{to_display(res['anchor'], market):%Y%m%d_%H%M}"
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"datetime": to_display(fut_ts, market), "close_q05": q[0], "close_q25": q[1], "close_q50": q[2],
                          "close_q75": q[3], "close_q95": q[4],
                          **{f"rep_{k}": pred[k][res['k_rep']] for k in ("open", "high", "low", "close", "volume")}})
    if market == "crypto":
        table = table.rename(columns={"datetime": "close_time_utc8"})
        table.insert(1, "close_time_utc", fut_ts)
    fut_real = res["fut_real"]
    if len(fut_real):
        n = len(fut_real)
        table.loc[:n - 1, "actual_close"] = fut_real["close"].to_numpy()
        act = float(np.log(fut_real["close"].iloc[-1] / prev_close))
        inside = np.mean((fut_real["close"].to_numpy() >= q[0][:n]) & (fut_real["close"].to_numpy() <= q[4][:n]))
        print(f"  [对照] 文件里锚点之后有 {n} 根真实 K 线：实际 {pct(act)}，"
              f"{inside:.0%} 的真实收盘落在 5%-95% 预测区间内")
    table.to_csv(args.out / f"{tag}.csv", index=False)
    np.savez_compressed(args.out / f"{tag}.npz", **{f"pred_{k}": v.astype(np.float32) for k, v in pred.items()},
                        fut_ts=fut_ts.to_numpy(dtype="datetime64[ns]"), prev_close=np.float64(prev_close),
                        k_rep=res["k_rep"])
    if not args.no_plot:
        plot_result(res, args.out / f"{tag}.png", args.ctx_show)
    print(f"  输出: {args.out / tag}.{{png,csv,npz}}")


if __name__ == "__main__":
    main()
