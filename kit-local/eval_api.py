#!/usr/bin/env python
"""策略能看到的数据接口：只暴露"某个时刻之前已经知道的东西"，用来写不偷看未来的交易规则。

    from eval_api import Dataset
    ds = Dataset("outputs/eval/live/train")      # eval_dataset.py 生成的目录
    ds.now                                        # 这个视图的信息截止时间（UTC）
    ds.forecasts                                  # 截止时间之前已经能在看板上看到的预测（visible_at <= now）
    ds.pairs                                      # 这些预测每根未来 K 线的分位数（只有预测，没有实际值）
    ds.paths(fid)                                 # 原始路径：close/open/high/low (paths, bars)、fut_ts、prev_close
    ds.bars("SOL", "15m")                         # 截止时间之前已经收盘的实际 K 线（1m/15m/1h/2h/1d）
    ds.last_price("SOL", t)                       # t 时刻最近一根已收盘 1 分钟线的收盘价（t 默认 now）
    ds.at(t)                                      # 截止时间提前到 t 的视图

策略文件约定（见 strategies/ 下的例子）:
    NAME = "..."; DESCRIPTION = "..."; PARAMS = {...}       # 参数写死在 PARAMS 里
    def signals(ds, **params) -> DataFrame                   # 列：asset, side(+1/-1), entry_time, exit_time[, tp, sl, tag]
每一行信号只能用 entry_time 及之前的信息来决定（包括方向、到期时间、止盈止损价）。eval_run.py 会把数据截断在若干时刻重跑
策略，检查"截断之前发出的信号"是否和完整数据上的一模一样，不一样就判为偷看未来。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eval_dataset import load_paths

PRED_COLS = ["fid", "asset", "scale", "anchor", "made_at", "visible_at", "lead", "t", "lead_min", "prev_close",
             "q05", "q10", "q25", "q50", "q75", "q90", "q95", "mean", "std", "p_up",
             "hi_q50", "hi_q95", "lo_q05", "lo_q50", "cummax_q50", "cummin_q50"]


def _ns(s: pd.Series) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(s).as_unit("ns"), index=s.index, name=s.name)


class Dataset:
    def __init__(self, directory: str | Path, now=None, _cache: dict | None = None):
        self.dir = Path(directory)
        self._c = _cache if _cache is not None else self._load()
        self.now = pd.Timestamp(now) if now is not None else self._c["end"]

    def _load(self) -> dict:
        F = pd.read_parquet(self.dir / "forecasts.parquet")
        for col in ("anchor", "made_at", "visible_at", "end"):
            F[col] = _ns(F[col])
        P = pd.read_parquet(self.dir / "pairs.parquet", columns=PRED_COLS)
        for col in ("anchor", "made_at", "visible_at", "t"):
            P[col] = _ns(P[col])
        bars = {}
        for p in sorted(self.dir.glob("bars_*_*.parquet")):
            _, asset, scale = p.stem.split("_")
            b = pd.read_parquet(p)
            for col in ("open_time", "t"):
                b[col] = _ns(b[col])
            bars[(asset, scale)] = b
        end = max(b["t"].max() for (a, s), b in bars.items() if s == "1m")
        return dict(F=F, P=P, bars=bars, end=end,
                    m1={a: (b["t"].to_numpy("datetime64[ns]").astype("int64"), b["close"].to_numpy(float))
                        for (a, s), b in bars.items() if s == "1m"})

    @property
    def end(self) -> pd.Timestamp:
        """数据里最后一根 1 分钟线的收盘时间。"""
        return self._c["end"]

    @property
    def assets(self) -> list:
        return sorted(self._c["m1"])

    @property
    def forecasts(self) -> pd.DataFrame:
        F = self._c["F"]
        cols = [c for c in F.columns if c not in ("n_done", "entry_t", "entry_px", "entry_t_vis", "entry_px_vis")]
        return F.loc[F["visible_at"] <= self.now, cols].reset_index(drop=True)

    @property
    def pairs(self) -> pd.DataFrame:
        P = self._c["P"]
        return P[P["visible_at"] <= self.now].reset_index(drop=True)

    def paths(self, fid: str) -> dict:
        F = self._c["F"]
        r = F[F["fid"] == fid]
        if not len(r) or r["visible_at"].iloc[0] > self.now:
            raise KeyError(f"{fid} 在 {self.now} 还看不到")
        return load_paths(r["file"].iloc[0], float(r["rebase"].iloc[0]) if "rebase" in r else 1.0)

    def bars(self, asset: str, scale: str) -> pd.DataFrame:
        b = self._c["bars"][(asset, scale)]
        return b[b["t"] <= self.now].reset_index(drop=True)

    def last_price(self, asset: str, t=None) -> float:
        t = self.now if t is None else pd.Timestamp(t)
        if t > self.now:
            raise ValueError(f"问的是 {t} 的价格，但这个视图只到 {self.now}")
        ts, close = self._c["m1"][asset]
        i = int(np.searchsorted(ts, t.as_unit("ns").value, side="right")) - 1
        return float(close[i]) if i >= 0 else float("nan")

    def at(self, now) -> "Dataset":
        return Dataset(self.dir, now, _cache=self._c)
