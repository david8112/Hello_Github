#!/usr/bin/env python
"""在一份评估数据集上原样运行一个策略文件，输出撮合结果、统计和"是否偷看未来"的检查。

  python eval_run.py strategies/fade_15m_4h.py --data outputs/eval/live/train
  python eval_run.py strategies/fade_15m_4h.py --data outputs/eval/live --from "2026-11-17 12:00"     # 只统计这之后进场的
  python eval_run.py strategies/x.py --data outputs/eval/hist --set hold_h=3                            # 临时改参数（敏感性检查用）

输出 JSON（同时写到 --out）：signals/trades 数量、causality（截断重跑检查）、one_position 和 all_signals 两种口径的
summary / block / baselines，以及两个压力口径：多等 15 分钟再进场（late15）、费用加倍（cost2x）。时间都是 UTC。

偷看检查（causality）：在 --cuts 个进场时刻（默认 12；--cuts 0 = 每个进场时刻）把数据集物理截断（Dataset.cut）、重新加载
策略文件再跑一遍，截断时刻及之前的信号必须和完整数据上的一模一样，也不能发出晚于截断时刻的信号；策略在截断数据上报错
也记为一次违规。最终要采用的规则建议用 --cuts 0 再查一遍。--set 只能改 PARAMS 里已有的参数，类型不对或名字写错直接报错。
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


def _utc_naive(x) -> pd.DatetimeIndex:
    """时间列统一成不带时区的 UTC 纳秒（带时区的先换算到 UTC 再去掉时区）。"""
    t = pd.DatetimeIndex(pd.to_datetime(x))
    if t.tz is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.as_unit("ns")


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t


def parse_set(params: dict, kv: str) -> tuple:
    """--set name=value：name 必须是 PARAMS 里已有的参数；value 按原参数的类型解析（布尔认 true/false/1/0，
    None、列表、字典按 JSON）。写错名字或类型直接报错，免得"改了参数"其实没改。"""
    if "=" not in kv:
        raise SystemExit(f"--set 要写成 name=value，收到的是 {kv!r}")
    k, v = kv.split("=", 1)
    k, v = k.strip(), v.strip()
    if k not in params:
        raise SystemExit(f"--set {k}=...：PARAMS 里没有这个参数（有的是 {sorted(params)}）")
    old = params[k]
    try:
        if isinstance(old, bool):
            if v.lower() not in ("true", "false", "1", "0"):
                raise ValueError("布尔参数只认 true/false/1/0")
            return k, v.lower() in ("true", "1")
        if isinstance(old, int):
            return k, int(v)
        if isinstance(old, float):
            return k, float(v)
        if isinstance(old, str):
            return k, v
        return k, json.loads(v)                                   # None / 列表 / 字典：按 JSON
    except (ValueError, json.JSONDecodeError) as e:
        raise SystemExit(f"--set {k}={v!r} 解析不了（原值 {old!r}，类型 {type(old).__name__}）：{e}") from None


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
        S[col] = _utc_naive(S[col])
    bad = S[~(S["exit_time"] > S["entry_time"]) | ~S["side"].isin([-1, 1]) | ~S["asset"].isin(ds.assets)]
    if len(bad):
        raise SystemExit(f"策略输出了 {len(bad)} 行非法信号（时间缺失或到期不晚于进场 / side 不是 ±1 / 品种不认识）")
    return S.sort_values(["entry_time", "asset"]).reset_index(drop=True)


def canon(S: pd.DataFrame) -> set:
    return {(r.asset, int(r.side), r.entry_time.value, r.exit_time.value,
             None if not np.isfinite(r.tp) else round(float(r.tp), 8), None if not np.isfinite(r.sl) else round(float(r.sl), 8))
            for r in S.itertuples(index=False)}


def causality(mod, ds: Dataset, params: dict, full: pd.DataFrame, n_cuts: int = 12) -> dict:
    """把数据截断在若干个进场时刻重跑策略：截断时刻及之前发出的信号必须和完整数据上的完全一致。
    n_cuts <= 0 表示每个进场时刻都查一遍（慢，但不会漏掉只在个别信号上偷看的情况）。"""
    if not len(full):
        return dict(cuts=0, violations=0, examples=[])
    times = np.sort(full.loc[full["entry_time"] <= ds.now, "entry_time"].unique())      # 数据结束之后的进场时刻没法截断
    if not len(times):
        return dict(cuts=0, violations=0, examples=[])
    k = len(times) if n_cuts <= 0 else min(n_cuts, len(times))
    cuts = pd.DatetimeIndex(times[np.unique(np.linspace(0, len(times) - 1, k).astype(int))])
    bad, examples = 0, []
    src = getattr(mod, "__file__", None)
    for T in cuts:
        try:
            fresh = load_strategy(Path(src)) if src else mod       # 每次重新加载策略文件：完整数据那一遍留下的模块级缓存不能带过来
            part = run_signals(fresh, ds.cut(T), params)           # cut：之后的数据在这个对象里根本不存在
        except (Exception, SystemExit) as e:  # noqa: BLE001
            bad += 1
            examples.append(f"截断在 {T} 时策略报错（{type(e).__name__}: {str(e)[:120]}）")
            continue
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
    ap.add_argument("--cuts", type=int, default=12, help="偷看检查截断重跑的次数（默认 12；0 = 每个进场时刻都查）")
    ap.add_argument("--out", type=Path, default=None, help="结果 JSON 和 trades.csv 写到这个目录")
    args = ap.parse_args()

    mod = load_strategy(args.strategy)
    params = dict(getattr(mod, "PARAMS", {}))
    for kv in args.set:
        k, v = parse_set(params, kv)
        params[k] = v
    ds = Dataset(args.data)
    full = run_signals(mod, ds, params)
    caus = dict(skipped=True) if args.no_causality else causality(mod, ds, params, full, n_cuts=args.cuts)
    S = full
    if args.t_from:
        S = S[S["entry_time"] >= _ts(args.t_from)]
    if args.t_to:
        S = S[S["entry_time"] < _ts(args.t_to)]
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
