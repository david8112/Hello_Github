#!/usr/bin/env python
"""把预测存档（outputs/*.npz）和 OKX 永续合约的实际 K 线对齐，生成"预测 vs 实际"评估数据集。

  python eval_dataset.py                                    # 实盘存档 outputs/ -> outputs/eval/live/
  python eval_dataset.py --split "2026-11-17 12:00"         # 另外按 UTC 时间切出 train/ 和 holdout/
  python eval_dataset.py --src outputs/hist --dst outputs/eval/hist --lag-min 5   # 历史样本外预测

时间全部是 UTC，K 线时间一律是收盘时间（1 分钟线另带 open_time）。输出:
  forecasts.parquet          每个预测一行（起点、生成时间、能下单的最早价格、终点分布摘要）
  pairs.parquet              每个（预测, 未来第 lead 根 K 线）一行：预测收盘价分位数 + 已收盘的实际 OHLC
  bars_{BTC|SOL}_{1m|15m|1h|2h|1d}.parquet   实际 K 线（只含已收盘的）
原始路径（paths × bars）仍在 npz 里，用 eval_dataset.load_paths(file) 读取。
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ASSETS = {"BTCUSDT": "BTC", "SOLUSDT": "SOL"}
OKX_INST = {"BTC": "BTC-USDT-SWAP", "SOL": "SOL-USDT-SWAP"}
OKX_BAR = {"1m": "1m", "15m": "15m", "1h": "1H", "2h": "2H", "1d": "1Dutc"}
STEP_MIN = {"1m": 1, "15m": 15, "1h": 60, "2h": 120, "1d": 1440}
SCALES = ("15m", "1h", "2h", "1d")
HIST_DAYS = {"1m": 0.5, "15m": 12, "1h": 30, "2h": 45, "1d": 200}     # 最早预测起点之前再多取多少天实际 K 线
QS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "kit-local-eval/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 4:
                raise
            time.sleep(1.5 ** attempt)


def okx_bars(asset: str, scale: str, start: pd.Timestamp, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """OKX 永续 [start, end] 之间开盘的已收盘 K 线，返回 open_time / t(收盘时间) / open high low close / vol。"""
    start_ms = int(start.value // 10**6)
    rows, after = {}, (None if end is None else int(end.value // 10**6) + 1)
    while True:
        q = dict(instId=OKX_INST[asset], bar=OKX_BAR[scale], limit=100)
        if after is not None:
            q["after"] = after
        js = get_json(f"https://www.okx.com/api/v5/market/history-candles?{urllib.parse.urlencode(q)}")
        if js.get("code") != "0":
            raise RuntimeError(f"OKX 返回错误 {js.get('code')} {js.get('msg')}")
        batch = js.get("data") or []
        for k in batch:
            if k[8] == "1":                                   # confirm=1：已收盘
                rows[int(k[0])] = (float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[6]))
        if not batch or int(batch[-1][0]) <= start_ms:
            break
        after = int(batch[-1][0])
        time.sleep(0.11)
    ms = sorted(t for t in rows if t >= start_ms)
    df = pd.DataFrame([rows[t] for t in ms], columns=["open", "high", "low", "close", "vol"])
    df.insert(0, "open_time", pd.to_datetime(ms, unit="ms"))
    df.insert(1, "t", df["open_time"] + pd.Timedelta(minutes=STEP_MIN[scale]))
    return df


def update_bars(path: Path, asset: str, scale: str, start: pd.Timestamp) -> pd.DataFrame:
    """增量更新本地缓存：已有的不重下，只补更早和更新的部分。"""
    old = pd.read_parquet(path) if path.exists() else None
    if old is not None and len(old) and old["open_time"].min() <= start:
        new = okx_bars(asset, scale, old["open_time"].max())
        df = pd.concat([old, new]).drop_duplicates("open_time", keep="last")
    else:
        df = okx_bars(asset, scale, start)
        if old is not None:
            df = pd.concat([old, df]).drop_duplicates("open_time", keep="last")
    df = df.sort_values("open_time").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def load_paths(file: str | Path) -> dict:
    """读一个预测 npz：close/open/high/low 是 (paths, bars) 的价格数组，fut_ts 是每根的 UTC 收盘时间。"""
    z = np.load(file, allow_pickle=False)
    out = {k: z[f"pred_{k}"].astype(float) for k in ("open", "high", "low", "close")}
    out["fut_ts"] = pd.DatetimeIndex(z["fut_ts"])
    out["prev_close"] = float(z["prev_close"])
    return out


def read_forecast(f: Path, lag_min: float | None) -> dict | None:
    name = f.stem.split("_")
    if len(name) != 4 or name[0] not in ASSETS or name[1] not in SCALES:
        return None
    z = np.load(f, allow_pickle=False)
    scale = name[1]
    step = pd.Timedelta(minutes=STEP_MIN[scale])
    fut = pd.DatetimeIndex(z["fut_ts"])
    anchor = pd.Timestamp(z["anchor"][()]) if "anchor" in z else fut[0] - step
    has_made = "made_at" in z
    made = pd.Timestamp(z["made_at"][()]) if has_made else pd.Timestamp(f.stat().st_mtime, unit="s")
    if lag_min is not None:                                    # 历史样本外：假设起点收盘后 lag_min 分钟拿到预测
        made = anchor + pd.Timedelta(minutes=lag_min)
    c = z["pred_close"].astype(float)
    return dict(file=str(f), code=name[0], asset=ASSETS[name[0]], scale=scale, step_min=STEP_MIN[scale], anchor=anchor,
                made_at=made, made_at_src=("lag" if lag_min is not None else "npz" if has_made else "mtime"),
                prev_close=float(z["prev_close"]), n_bars=len(fut), paths=int(c.shape[0]),
                steps=int(z["steps"][()]) if "steps" in z else 32,
                precision=str(z["precision"][()]) if "precision" in z else "", fut=fut,
                o=z["pred_open"].astype(float), h=z["pred_high"].astype(float), l=z["pred_low"].astype(float), c=c)


def build(src: Path, dst: Path, lag_min: float | None, quiet: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    dst.mkdir(parents=True, exist_ok=True)
    fcs = [r for r in (read_forecast(f, lag_min) for f in sorted(src.glob("*.npz"))) if r is not None]
    if not fcs:
        raise SystemExit(f"{src} 里没有 BTC/SOL 的预测 npz")
    first = min(r["anchor"] for r in fcs)
    bars = {}
    for asset in sorted({r["asset"] for r in fcs}):
        for scale in ("1m",) + SCALES:
            p = dst / f"bars_{asset}_{scale}.parquet"
            bars[(asset, scale)] = update_bars(p, asset, scale, (first - pd.Timedelta(days=HIST_DAYS[scale])).floor("D"))
            if not quiet:
                b = bars[(asset, scale)]
                print(f"  bars {asset} {scale}: {len(b)} 根  {b.t.min()} .. {b.t.max()}")
    frows, prows = [], []
    for r in fcs:
        asset, scale = r["asset"], r["scale"]
        fid = f"{asset}-{scale}-{r['anchor']:%Y%m%d%H%M}"
        act = bars[(asset, scale)].set_index("t")
        m1 = bars[(asset, "1m")]
        c, pc, fut = r["c"], r["prev_close"], r["fut"]
        q = np.quantile(c, QS, axis=0)
        hq = np.quantile(r["h"], [.5, .95], axis=0)
        lq = np.quantile(r["l"], [.05, .5], axis=0)
        cummax = np.maximum.accumulate(c, axis=1)              # 每条路径到第 lead 根为止的最高/最低收盘
        cummin = np.minimum.accumulate(c, axis=1)
        # 能下单的最早价格：生成时间之后第一根 1 分钟线的开盘价
        i = int(m1["open_time"].searchsorted(r["made_at"].ceil("min"), side="left"))
        entry_t, entry_px = (m1["open_time"].iloc[i], float(m1["open"].iloc[i])) if i < len(m1) else (pd.NaT, np.nan)
        anchor_ok = r["anchor"] in act.index
        anchor_close = float(act.loc[r["anchor"], "close"]) if anchor_ok else np.nan
        done = [t in act.index for t in fut]
        term = c[:, -1] / pc - 1
        frows.append(dict(
            fid=fid, asset=asset, scale=scale, step_min=r["step_min"], anchor=r["anchor"], made_at=r["made_at"],
            made_at_src=r["made_at_src"], lag_min=(r["made_at"] - r["anchor"]) / pd.Timedelta(minutes=1),
            prev_close=pc, anchor_close_okx=anchor_close, entry_t=entry_t, entry_px=entry_px, n_bars=r["n_bars"],
            n_done=int(sum(done)), end=fut[-1], paths=r["paths"], steps=r["steps"], precision=r["precision"],
            med_ret_end=float(np.median(term)), mean_ret_end=float(term.mean()), q05_ret_end=float(np.quantile(term, .05)),
            q95_ret_end=float(np.quantile(term, .95)), p_up_end=float((term > 0).mean()), file=r["file"]))
        for k, t in enumerate(fut):
            row = dict(fid=fid, asset=asset, scale=scale, anchor=r["anchor"], made_at=r["made_at"], lead=k + 1, t=t,
                       lead_min=(k + 1) * r["step_min"], prev_close=pc,
                       **{f"q{int(round(p * 100)):02d}": float(q[j, k]) for j, p in enumerate(QS)},
                       mean=float(c[:, k].mean()), std=float(c[:, k].std(ddof=1)), p_up=float((c[:, k] > pc).mean()),
                       hi_q50=float(hq[0, k]), hi_q95=float(hq[1, k]), lo_q05=float(lq[0, k]), lo_q50=float(lq[1, k]),
                       cummax_q50=float(np.median(cummax[:, k])), cummin_q50=float(np.median(cummin[:, k])),
                       done=bool(done[k]))
            if done[k]:
                a = act.loc[t]
                row.update(a_open=float(a.open), a_high=float(a.high), a_low=float(a.low), a_close=float(a.close))
            prows.append(row)
    F = pd.DataFrame(frows).sort_values(["made_at", "asset", "scale"]).reset_index(drop=True)
    P = pd.DataFrame(prows)
    for col in ("a_open", "a_high", "a_low", "a_close"):
        if col not in P:
            P[col] = np.nan
    return F, P, bars


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=HERE / "outputs", help="预测 npz 所在目录")
    ap.add_argument("--dst", type=Path, default=HERE / "outputs" / "eval" / "live")
    ap.add_argument("--lag-min", type=float, default=None, help="历史样本外预测用：假设起点收盘后多少分钟拿到预测（实盘存档不要给）")
    ap.add_argument("--split", default=None, help="UTC 时间，如 '2026-11-17 12:00'：另外写 train/（之前生成的预测 + 之前的实际）和 holdout/")
    args = ap.parse_args()

    F, P, bars = build(args.src, args.dst, args.lag_min)
    F.to_parquet(args.dst / "forecasts.parquet", index=False)
    P.to_parquet(args.dst / "pairs.parquet", index=False)
    d = P[P.done]
    print(f"{args.dst}: {len(F)} 个预测（{F.made_at.min()} .. {F.made_at.max()} UTC），{len(d)} 个已收盘的（预测, K线）对")
    print(F.groupby(["scale", "asset"]).agg(n=("fid", "size"), lag_min=("lag_min", "median"), done=("n_done", "sum")).to_string())
    if args.split:
        T = pd.Timestamp(args.split)
        for name, fm in (("train", F.made_at < T), ("holdout", F.made_at >= T)):
            sub = args.dst / name
            sub.mkdir(exist_ok=True)
            f = F[fm]
            p = P[P.fid.isin(f.fid)].copy()
            if name == "train":                                   # 训练段：切分时间之后的实际行情一律不可见
                late = p.t > T
                p.loc[late, ["a_open", "a_high", "a_low", "a_close"]] = np.nan
                p.loc[late, "done"] = False
                f = f.assign(n_done=f.fid.map(p[p.done].groupby("fid").size()).fillna(0).astype(int))
            f.to_parquet(sub / "forecasts.parquet", index=False)
            p.to_parquet(sub / "pairs.parquet", index=False)
            for (asset, scale), b in bars.items():
                (b[b.t <= T] if name == "train" else b).to_parquet(sub / f"bars_{asset}_{scale}.parquet", index=False)
            print(f"  {name}: {len(f)} 个预测，{int(p.done.sum())} 个已收盘对")
        (args.dst / "SPLIT.txt").write_text(f"{T} UTC\n", encoding="utf-8")


if __name__ == "__main__":
    main()
