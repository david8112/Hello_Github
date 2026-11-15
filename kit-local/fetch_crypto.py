#!/usr/bin/env python
"""从币安或 OKX 的公开行情接口下载 K 线，存成 forecast.py / backtest.py 能直接读的 CSV。

  python fetch_crypto.py --symbol BTCUSDT --interval 1d                  # 币安（默认）
  python fetch_crypto.py --exchange okx --symbol BTCUSDT --interval 1h   # OKX
  python fetch_crypto.py --symbol ETHUSDT --interval 2h --bars 3000

- 币安: https://data-api.binance.vision（只读行情镜像）；OKX: https://www.okx.com/api/v5/market/history-candles
  都不需要账号和密钥。
- 文件里的 datetime 列是 K 线「收盘时间」，UTC，无时区标记。这是 KiT 训练数据的约定，不要改：
  例如 UTC 2026-11-14 那根日线记为 2026-11-15 00:00（= 北京时间 11-15 08:00 收盘）
- 屏幕提示、forecast.py / backtest.py 的输出一律显示北京时间（UTC+8）
- OKX 的日线用 1Dutc（按 UTC 切日）；OKX 默认的 1D 是按香港时间切日，和模型的训练数据不一致
- 只保留已经收盘的 K 线，正在走的那根会被丢掉
- 默认下载量 = 模型需要的历史长度 + 成交量 EMA 预热，日线至少 1000 根
- KiT 的加密货币训练品种是 28 个 USDT 交易对（BTCUSDT、ETHUSDT、SOLUSDT …），代码写成 BTCUSDT 形式
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

STEP_MIN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "1d": 1440}
LC = {"1m": 1200, "5m": 960, "15m": 800, "30m": 640, "1h": 400, "2h": 360, "1d": 250}   # 模型固定的历史窗口
OKX_BAR = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H", "2h": "2H", "1d": "1Dutc"}
QUOTES = ("USDT", "USDC", "USD", "BTC", "ETH")


def default_bars(interval: str) -> int:
    per_day = 1440 // STEP_MIN[interval]
    need = LC[interval] + 12 * 5 * per_day + 100     # 历史窗口 + 5 天半衰期 EMA 的 12 倍预热
    return max(need, 1000) if interval == "1d" else need


def get_json(url: str):
    # OKX 会拒绝 Python 默认的 User-Agent（Python-urllib/x.y，返回 403），所以带一个通用的客户端标识
    req = urllib.request.Request(url, headers={"User-Agent": "kit-local-fetch/1.0", "Accept": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise SystemExit(f"下载失败: {e}（{url.split('?')[0]}）")
            time.sleep(2 ** attempt)


def fetch_binance(symbol: str, interval: str, n: int, now_ms: int) -> list[tuple]:
    """返回 [(open_ms, o, h, l, c, v)]，只含已收盘的 K 线。"""
    step_ms = STEP_MIN[interval] * 60_000
    rows, cur = [], now_ms - (n + 2) * step_ms
    while cur < now_ms:
        q = urllib.parse.urlencode(dict(symbol=symbol, interval=interval, startTime=cur, endTime=now_ms, limit=1000))
        batch = get_json(f"https://data-api.binance.vision/api/v3/klines?{q}")
        if not batch:
            break
        rows += [(int(k[0]), *map(float, k[1:6])) for k in batch if int(k[6]) < now_ms]
        cur = int(batch[-1][0]) + step_ms
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return rows


def okx_inst(symbol: str) -> str:
    s = symbol.upper()
    if "-" in s:
        return s
    for qt in QUOTES:
        if s.endswith(qt) and len(s) > len(qt):
            return f"{s[:-len(qt)]}-{qt}"
    raise SystemExit(f"无法把 {symbol} 转成 OKX 的 instId（例如 BTC-USDT）")


def fetch_okx(symbol: str, interval: str, n: int, now_ms: int) -> list[tuple]:
    """OKX history-candles 每次最多 100 根、从新到旧翻页；confirm=1 表示已收盘。"""
    inst, bar = okx_inst(symbol), OKX_BAR[interval]
    rows, after = [], now_ms
    while len(rows) < n:
        q = urllib.parse.urlencode(dict(instId=inst, bar=bar, after=after, limit=100))
        js = get_json(f"https://www.okx.com/api/v5/market/history-candles?{q}")
        if js.get("code") != "0":
            raise SystemExit(f"OKX 返回错误: {js.get('code')} {js.get('msg')}（instId={inst}）")
        batch = js.get("data") or []
        if not batch:
            break
        rows += [(int(k[0]), *map(float, k[1:6])) for k in batch if k[8] == "1"]
        after = int(batch[-1][0])
        time.sleep(0.12)                              # 限频 20 次 / 2 秒
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="下载加密货币 K 线（收盘时间戳，UTC）")
    ap.add_argument("--exchange", default="binance", choices=("binance", "okx"))
    ap.add_argument("--symbol", default="BTCUSDT", help="如 BTCUSDT（OKX 也可写 BTC-USDT）")
    ap.add_argument("--interval", default="1d", choices=list(STEP_MIN))
    ap.add_argument("--bars", type=int, default=None, help="下载多少根（默认按模型需要自动计算）")
    ap.add_argument("--out", type=Path, default=None, help="默认 data_crypto/<SYMBOL>_<interval>[_okx].csv")
    args = ap.parse_args()

    sym = args.symbol.upper().replace("-", "") if args.exchange == "binance" else args.symbol.upper()
    n = args.bars or default_bars(args.interval)
    now_ms = int(time.time() * 1000)
    rows = (fetch_binance if args.exchange == "binance" else fetch_okx)(sym, args.interval, n, now_ms)
    if not rows:
        raise SystemExit(f"没有取到 {args.exchange} {sym} {args.interval} 的数据（交易对名称对吗？）")
    step_ms = STEP_MIN[args.interval] * 60_000
    df = pd.DataFrame(rows, columns=["open_ms", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("open_ms").sort_values("open_ms")
    df = df[df["open_ms"] + step_ms <= now_ms]                         # 只要已收盘的
    out = pd.DataFrame({"datetime": pd.to_datetime(df["open_ms"] + step_ms, unit="ms"),   # 收盘时间
                        **{c: df[c] for c in ("open", "high", "low", "close", "volume")}}).tail(n).reset_index(drop=True)
    gaps = (out["datetime"].diff().dropna() != pd.Timedelta(minutes=STEP_MIN[args.interval])).sum()
    if gaps:
        print(f"[warn] 数据里有 {gaps} 处时间不连续（交易所停机或缺数据）")

    code = sym.replace("-", "")
    suffix = "" if args.exchange == "binance" else f"_{args.exchange}"
    path = args.out or Path(__file__).resolve().parent / "data_crypto" / f"{code}_{args.interval}{suffix}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    bj = out["datetime"] + pd.Timedelta(hours=8)
    print(f"{args.exchange} {code} {args.interval}: {len(out)} 根，{bj.iloc[0]} .. {bj.iloc[-1]}"
          f"（收盘时间，北京时间 UTC+8；文件内按 UTC 保存），最新收盘价 {out['close'].iloc[-1]:g} -> {path}")


if __name__ == "__main__":
    main()
