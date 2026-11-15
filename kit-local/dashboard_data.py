#!/usr/bin/env python
"""生成「BTC·SOL 行情预测」看板的数据文件（JSON），给 Claude 上传到看板。

  python dashboard_data.py            # 读 outputs/ 里每个品种 × 周期最新的预测，拉实时行情，写 dashboard/*.json

输出（时间一律是北京时间 UTC+8，K 线收盘时间，格式 "YYYY-MM-DD HH:MM"）:
  dashboard/forecasts.json  预测区间：每根未来 K 线收盘价的 5/25/50/75/95% 分位（第一行是起点收盘价）
  dashboard/summary.json    每个预测一行：生成时间、起止时间、起点收盘价、终点涨跌幅分位、上涨路径占比
  dashboard/actuals.json    实际 K 线（OKX 永续合约 BTC-USDT-SWAP / SOL-USDT-SWAP，含正在走的那根，live=1）
  dashboard/backtest.json   日线样本外回测（有结果文件时才写）
每个文件是 {"rows": [...], "meta": [...]}，meta 里带本脚本源码，方便核对数字怎么来的。
"""
from __future__ import annotations

import glob
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DST = HERE / "dashboard"
ASSETS = {"BTCUSDT": "BTC", "SOLUSDT": "SOL"}
OKX_INST = {"BTCUSDT": "BTC-USDT-SWAP", "SOLUSDT": "SOL-USDT-SWAP"}     # 和你在 OKX 交易的永续合约一致
OKX_BAR = {"15m": "15m", "1h": "1H", "2h": "2H", "1d": "1Dutc"}          # 日线按 UTC 切分（北京时间 08:00 收盘）
SCALES = ("15m", "1h", "2h", "1d")
STEP_MIN = {"15m": 15, "1h": 60, "2h": 120, "1d": 1440}
HIST_BARS = {"15m": 96, "1h": 72, "2h": 60, "1d": 90}      # 预测起点之前画多少根历史 K 线
BJ = pd.Timedelta(hours=8)
FMT = "%Y-%m-%d %H:%M"


def bj(ts) -> str:
    """内部 UTC 时间 -> 北京时间字符串。"""
    return (pd.Timestamp(ts) + BJ).strftime(FMT)


def get_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "kit-local-dashboard/1.0"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def okx_klines(code: str, scale: str, start_ms: int) -> list[dict]:
    """OKX 永续合约从 start_ms（开盘时间）起到现在的 K 线，含正在走的那根（confirm=0）。从新到旧翻页，每页 100 根。"""
    out, after = {}, None
    while True:
        q = dict(instId=OKX_INST[code], bar=OKX_BAR[scale], limit=100)
        if after is not None:
            q["after"] = after
        js = get_json(f"https://www.okx.com/api/v5/market/history-candles?{urllib.parse.urlencode(q)}")
        if js.get("code") != "0":
            raise RuntimeError(f"OKX 返回错误 {js.get('code')} {js.get('msg')}")
        batch = js.get("data") or []
        for k in batch:
            out[int(k[0])] = dict(open_ms=int(k[0]), open=float(k[1]), high=float(k[2]), low=float(k[3]),
                                  close=float(k[4]), live=int(k[8] == "0"))
        if not batch or int(batch[-1][0]) <= start_ms:
            break
        after = int(batch[-1][0])
        time.sleep(0.12)
    return [out[t] for t in sorted(out) if t >= start_ms]


def latest_forecasts() -> dict:
    """每个 (品种, 周期) 取 outputs/ 里锚点最晚的那个 npz。"""
    best = {}
    for f in glob.glob(str(OUT / "*.npz")):
        name = Path(f).stem.split("_")
        if len(name) != 4 or name[0] not in ASSETS or name[1] not in SCALES:
            continue
        key = (name[0], name[1])
        stamp = name[2] + name[3]
        if key not in best or stamp > best[key][0]:
            best[key] = (stamp, f)
    return {k: v[1] for k, v in best.items()}


def main() -> None:
    DST.mkdir(exist_ok=True)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    code_meta = {"label": "dashboard_data.py", "value": Path(__file__).read_text(encoding="utf-8"),
                 "type": "code", "lang": "python"}
    bands, summary, actuals = [], [], []
    for (code, scale), f in sorted(latest_forecasts().items()):
        z = np.load(f, allow_pickle=False)
        c = z["pred_close"].astype(float)
        fut = pd.DatetimeIndex(z["fut_ts"])
        pc = float(z["prev_close"])
        anchor = pd.Timestamp(z["anchor"]) if "anchor" in z else fut[0] - pd.Timedelta(minutes=STEP_MIN[scale])
        made = pd.Timestamp(z["made_at"]) if "made_at" in z else pd.Timestamp(Path(f).stat().st_mtime, unit="s")
        precision = str(z["precision"]) if "precision" in z else ("fp32" if code == "BTCUSDT" else "bf16")
        q = np.quantile(c, [.05, .25, .5, .75, .95], axis=0)
        asset = ASSETS[code]
        bands.append(dict(asset=asset, scale=scale, key=f"{asset}-{scale}", step=0, t=bj(anchor),
                          q05=pc, q25=pc, q50=pc, q75=pc, q95=pc))
        for i, t in enumerate(fut):
            bands.append(dict(asset=asset, scale=scale, key=f"{asset}-{scale}", step=i + 1, t=bj(t),
                              q05=round(q[0, i], 4), q25=round(q[1, i], 4), q50=round(q[2, i], 4),
                              q75=round(q[3, i], 4), q95=round(q[4, i], 4)))
        term = c[:, -1] / pc - 1
        summary.append(dict(
            asset=asset, scale=scale, key=f"{asset}-{scale}", made_at=bj(made), anchor=bj(anchor),
            end=bj(fut[-1]), bars=len(fut), last_close=pc, paths=int(c.shape[0]),
            med_ret=round(float(np.median(term)), 6), q05_ret=round(float(np.quantile(term, .05)), 6),
            q95_ret=round(float(np.quantile(term, .95)), 6), p_up=round(float((term > 0).mean()), 4),
            precision=precision, file=Path(f).name))
        # 实际 K 线：预测起点前 HIST_BARS 根，到现在
        step = pd.Timedelta(minutes=STEP_MIN[scale])
        start = anchor - step * (HIST_BARS[scale])          # 第一根的开盘时间
        for k in okx_klines(code, scale, int(start.value // 10**6)):
            close_t = pd.Timestamp(k["open_ms"], unit="ms") + step
            actuals.append(dict(asset=asset, scale=scale, key=f"{asset}-{scale}", t=bj(close_t),
                                open=k["open"], high=k["high"], low=k["low"], close=k["close"], live=k["live"]))
        time.sleep(0.2)

    def dump(name, rows, extra=()):
        meta = [{"label": "生成时间（北京时间）", "value": bj(now)}, *extra, code_meta]
        (DST / name).write_text(json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(rows)} 行")

    model = {"label": "模型", "value": "KiT step_market_284M（训练截止 2025-12-31），每个预测采样 16-32 条路径，32 步欧拉积分"}
    dump("forecasts.json", bands, [model, {"label": "输入", "value": "outputs/*.npz（forecast.py 的输出）"}])
    dump("summary.json", summary, [model])
    dump("actuals.json", actuals, [{"label": "输入", "value": "OKX 永续合约公开行情 www.okx.com/api/v5/market/history-candles（BTC-USDT-SWAP、SOL-USDT-SWAP）"}])

    bt = sorted(glob.glob(str(OUT / "backtest_BTCUSDT_1d_*_bf16.csv")))
    if bt:
        t = pd.read_csv(bt[-1])
        anchor_col = "anchor_utc8" if "anchor_utc8" in t else "anchor"
        offs = pd.Timedelta(0) if anchor_col == "anchor_utc8" else BJ
        rows = [dict(window=i + 1, anchor=(pd.Timestamp(r[anchor_col]) + offs).strftime(FMT),
                     pred_median=round(float(np.expm1(r.pred_median)), 6), q05=round(float(np.expm1(r.q05)), 6),
                     q95=round(float(np.expm1(r.q95)), 6), actual=round(float(np.expm1(r.actual)), 6),
                     p_up=float(r.p_up), hit=int(np.sign(r.pred_median) == np.sign(r.actual)),
                     in_band=int(bool(r.end_in_band))) for i, r in t.iterrows()]
        dump("backtest.json", rows, [{"label": "输入", "value": Path(bt[-1]).name},
                                     {"label": "说明", "value": "BTC 日线，2026 年起每 20 天一个互不重叠的窗口，用窗口开始前的数据预测之后 20 天（简单涨跌幅）"}])


if __name__ == "__main__":
    main()
