#!/usr/bin/env python
"""从币安公开行情接口下载 K 线，存成 forecast.py / backtest.py 能直接读的 CSV。

  python fetch_binance.py --symbol BTCUSDT --interval 1d
  python fetch_binance.py --symbol ETHUSDT --interval 1h --bars 3000

- 数据源 https://data-api.binance.vision（币安的只读行情镜像，无需账号和密钥）
- datetime 列是 K 线「收盘时间」，UTC，无时区标记。这是 KiT 的约定：
  例如 2026-11-14 那根日线记为 2026-11-15 00:00，1h 线 09:00-10:00 那根记为 10:00
- 只保留已经收盘的 K 线，正在走的那根会被丢掉
- 默认下载量 = 模型需要的历史长度 + 成交量 EMA 预热，日线至少 1000 根
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"
STEP_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "1d": 1440}
LC = {"1m": 1200, "5m": 960, "15m": 800, "30m": 640, "1h": 400, "2h": 360, "1d": 250}   # 模型固定的历史窗口


def default_bars(interval: str) -> int:
    per_day = 1440 // STEP_MIN[interval]
    need = LC[interval] + 12 * 5 * per_day + 100     # 历史窗口 + 5 天半衰期 EMA 的 12 倍预热
    return max(need, 1000) if interval == "1d" else need


def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    step_ms = STEP_MIN[interval] * 60_000
    rows, cur = [], start_ms
    while cur < end_ms:
        q = urllib.parse.urlencode(dict(symbol=symbol, interval=interval, startTime=cur,
                                        endTime=end_ms, limit=1000))
        for attempt in range(4):
            try:
                with urllib.request.urlopen(f"{BASE}?{q}", timeout=30) as r:
                    batch = json.load(r)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    raise SystemExit(f"下载失败: {e}")
                time.sleep(2 ** attempt)
        if not batch:
            break
        rows += batch
        cur = int(batch[-1][0]) + step_ms
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="下载币安 K 线（收盘时间戳，UTC）")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1d", choices=list(STEP_MIN))
    ap.add_argument("--bars", type=int, default=None, help="下载多少根（默认按模型需要自动计算）")
    ap.add_argument("--out", type=Path, default=None, help="默认 data_crypto/<SYMBOL>_<interval>.csv")
    args = ap.parse_args()

    sym = args.symbol.upper()
    n = args.bars or default_bars(args.interval)
    now_ms = int(time.time() * 1000)
    step_ms = STEP_MIN[args.interval] * 60_000
    rows = fetch(sym, args.interval, now_ms - (n + 2) * step_ms, now_ms)
    if not rows:
        raise SystemExit(f"没有取到 {sym} {args.interval} 的数据（交易对名称对吗？）")
    df = pd.DataFrame(rows).iloc[:, :7]
    df.columns = ["open_ms", "open", "high", "low", "close", "volume", "close_ms"]
    df = df[df["close_ms"].astype("int64") < now_ms]                  # 丢掉还没收盘的那根
    df = df.drop_duplicates("open_ms").sort_values("open_ms")
    out = pd.DataFrame({
        "datetime": pd.to_datetime(df["open_ms"].astype("int64") + step_ms, unit="ms"),   # 收盘时间
        **{c: df[c].astype(float) for c in ("open", "high", "low", "close", "volume")},
    }).tail(n).reset_index(drop=True)

    path = args.out or Path(__file__).resolve().parent / "data_crypto" / f"{sym}_{args.interval}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"{sym} {args.interval}: {len(out)} 根，{out['datetime'].iloc[0]} .. {out['datetime'].iloc[-1]}（收盘时间 UTC）"
          f"，最新收盘价 {out['close'].iloc[-1]:g} -> {path}")


if __name__ == "__main__":
    main()
