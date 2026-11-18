#!/usr/bin/env python
"""在一份评估数据集上原样运行一个策略文件，输出撮合结果、统计和"是否偷看未来"的检查。

  python eval_run.py strategies/fade_15m_4h.py --data outputs/eval/live/train
  python eval_run.py strategies/fade_15m_4h.py --data outputs/eval/live --from "2026-11-17 12:00"     # 只统计这之后进场的
  python eval_run.py strategies/x.py --data outputs/eval/hist --set hold_h=3                            # 临时改参数（敏感性检查用）

输出 JSON（同时写到 --out）：signals/trades 数量、causality（截断重跑检查）、one_position 和 all_signals 两种口径的
summary / block / baselines，以及两个压力口径：多等 15 分钟再进场（late15）、费用加倍（cost2x）。时间都是 UTC。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import eval_engine as E
from eval_api import Dataset

KEY = ["asset", "side", "entry_time", "exit_time", "tp", "sl"]


def load_strategy(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def run_signals(mod, ds: Dataset, params: dict) -> pd.DataFrame:
    S = mod.signals(ds, **params)
    S = pd.DataFrame(S).copy() if S is not None else pd.DataFrame()
    if not len(S):
        return pd.DataFrame(columns=KEY + ["tag"])
    for col in ("tp", "sl"):
        if col not in S:
            S[col] = np.nan
        S[col] = S[col].astype(float)
    if "tag" not in S:
        S["tag"] = ""
    S["side"] = S["side"].astype(int)
    for col in ("entry_time", "exit_time"):
        S[col] = pd.DatetimeIndex(pd.to_datetime(S[col])).as_unit("ns")
    bad = S[(S["exit_time"] <= S["entry_time"]) | ~S["side"].isin([-1, 1]) | ~S["asset"].isin(ds.assets)]
    if len(bad):
        raise SystemExit(f"策略输出了 {len(bad)} 行非法信号（到期不晚于进场 / side 不是 ±1 / 品种不认识）")
    return S.sort_values(["entry_time", "asset"]).reset_index(drop=True)


def canon(S: pd.DataFrame) -> set:
    return {(r.asset, int(r.side), r.entry_time.value, r.exit_time.value,
             None if not np.isfinite(r.tp) else round(float(r.tp), 8), None if not np.isfinite(r.sl) else round(float(r.sl), 8))
            for r in S.itertuples(index=False)}


def causality(mod, ds: Dataset, params: dict, full: pd.DataFrame, n_cuts: int = 12) -> dict:
    """把数据截断在若干个进场时刻重跑策略：截断时刻及之前发出的信号必须和完整数据上的完全一致。"""
    if not len(full):
        return dict(cuts=0, violations=0, examples=[])
    times = np.sort(full["entry_time"].unique())
    cuts = pd.DatetimeIndex(times[np.unique(np.linspace(0, len(times) - 1, min(n_cuts, len(times))).astype(int))])
    bad, examples = 0, []
    for T in cuts:
        part = run_signals(mod, ds.at(T), params)
        if len(part) and part["entry_time"].max() > T:
            bad += 1
            examples.append(f"截断在 {T} 时策略发出了 {part['entry_time'].max()} 的信号")
        a, b = canon(part[part["entry_time"] <= T]), canon(full[full["entry_time"] <= T])
        if a != b:
            bad += 1
            examples.append(f"截断在 {T}：截断视图 {len(a)} 条，完整数据 {len(b)} 条，不同的 {len(a ^ b)} 条")
    return dict(cuts=int(len(cuts)), violations=int(bad), examples=examples[:5])


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if k != "trades"}
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy", type=Path)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--from", dest="t_from", default=None, help="只统计这个 UTC 时间及之后进场的信号")
    ap.add_argument("--to", dest="t_to", default=None, help="只统计这个 UTC 时间之前进场的信号")
    ap.add_argument("--set", action="append", default=[], help="临时覆盖 PARAMS 里的参数，如 --set hold_h=3")
    ap.add_argument("--latency", type=float, default=0.0, help="看到信号后多少分钟才下单")
    ap.add_argument("--no-causality", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="结果 JSON 和 trades.csv 写到这个目录")
    args = ap.parse_args()

    mod = load_strategy(args.strategy)
    params = dict(getattr(mod, "PARAMS", {}))
    for kv in args.set:
        k, v = kv.split("=", 1)
        params[k] = type(params[k])(v) if k in params and not isinstance(params[k], bool) else json.loads(v)
    ds = Dataset(args.data)
    full = run_signals(mod, ds, params)
    caus = dict(skipped=True) if args.no_causality else causality(mod, ds, params, full)
    S = full
    if args.t_from:
        S = S[S["entry_time"] >= pd.Timestamp(args.t_from)]
    if args.t_to:
        S = S[S["entry_time"] < pd.Timestamp(args.t_to)]
    S = S.reset_index(drop=True)
    bars = E.load_1m(args.data)
    res = dict(strategy=getattr(mod, "NAME", args.strategy.stem), file=str(args.strategy), params=params, data=str(args.data),
               period=dict(t_from=args.t_from, t_to=args.t_to, data_end=str(ds.end)), n_signals=int(len(S)),
               long_share=float((S["side"] > 0).mean()) if len(S) else None, causality=caus,
               costs=dict(fee_bps_per_side=E.FEE_BPS, slip_bps_per_side=E.SLIP_BPS, latency_min=args.latency))
    if len(S):
        main_r = E.evaluate(S, bars, one_position=True, latency_min=args.latency)
        res["one_position"] = main_r
        res["all_signals"] = E.evaluate(S, bars, one_position=False, latency_min=args.latency)
        T = main_r["trades"]
        if len(T):
            res["by_asset"] = {a: E.summarize(g) for a, g in T.groupby("asset")}
            res["by_side"] = {("long" if s > 0 else "short"): E.summarize(g) for s, g in T.groupby("side")}
            res["by_reason"] = T["reason"].value_counts().to_dict()
        late = E.evaluate(S, bars, one_position=True, latency_min=args.latency + 15)
        cost2 = E.evaluate(S, bars, one_position=True, latency_min=args.latency, fee_bps=2 * E.FEE_BPS, slip_bps=2 * E.SLIP_BPS)
        res["stress"] = dict(late15=dict(summary=late["summary"], block=late["block"]),
                             cost2x=dict(summary=cost2["summary"], block=cost2["block"]))
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            T.to_csv(args.out / "trades.csv", index=False)
    out = clean(res)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
