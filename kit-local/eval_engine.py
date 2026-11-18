#!/usr/bin/env python
"""信号回测引擎：把"什么时候、哪个方向、怎么离场"的交易清单放到 1 分钟实际 K 线上逐笔撮合。

约定（全部保守处理，避免把回测做得比实盘好看）:
  - 进场：entry_time 之后第一根 1 分钟线的开盘价（市价单）。entry_time 必须 >= 预测的生成时间 made_at。
  - 离场：先到者为准 —— 止盈价 tp / 止损价 sl / 到期 exit_time（到期按之后第一根 1 分钟线开盘价）。
    同一根 1 分钟线里止盈止损都碰到时，按止损成交。止损按止损价再让 slip_bps 的滑点。
  - 费用：每边 fee_bps（OKX 永续吃单 5bp），进场和离场各一次；另加每边 slip_bps 滑点。不含资金费。
  - 收益按 1 倍名义本金、不复利：net = side * (exit/entry - 1) - 费用。杠杆只是把它乘倍数（同时放大亏损和爆仓风险）。

用法:
    import eval_engine as E
    bars = E.load_1m("outputs/eval/live")                 # {"BTC": df, "SOL": df}
    trades = E.simulate(spec, bars)                       # spec: asset, side, entry_time, exit_time[, tp, sl, tag]
    print(E.summarize(trades))
    print(E.block_bootstrap(trades))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEE_BPS = 5.0        # 每边吃单手续费（OKX 永续普通用户 taker 0.05%）
SLIP_BPS = 1.0       # 每边滑点


def load_1m(dataset_dir: str | Path) -> dict:
    d = Path(dataset_dir)
    return {p.stem.split("_")[1]: pd.read_parquet(p) for p in sorted(d.glob("bars_*_1m.parquet"))}


def simulate(spec: pd.DataFrame, bars1m: dict, fee_bps: float = FEE_BPS, slip_bps: float = SLIP_BPS) -> pd.DataFrame:
    """逐笔撮合。spec 列：asset, side(+1 多/-1 空), entry_time, exit_time, 可选 tp, sl（价格）, tag。
    返回每笔：entry_t/entry_px/exit_t/exit_px/reason/gross/net/hold_min/mae/mfe。数据不够（进场后没有行情）的笔会被丢弃并计数在 attrs["dropped"]。"""
    out, dropped = [], 0
    arr = {a: (b["open_time"].to_numpy("datetime64[ns]"), b["open"].to_numpy(float), b["high"].to_numpy(float),
               b["low"].to_numpy(float), b["close"].to_numpy(float)) for a, b in bars1m.items()}
    cost = 2 * (fee_bps + slip_bps) / 1e4
    for r in spec.itertuples(index=False):
        ot, o, h, l, c = arr[r.asset]
        side = int(r.side)
        i0 = int(np.searchsorted(ot, np.datetime64(pd.Timestamp(r.entry_time).ceil("min"), "ns"), side="left"))
        i1 = int(np.searchsorted(ot, np.datetime64(pd.Timestamp(r.exit_time).ceil("min"), "ns"), side="left"))
        if i0 >= len(ot) or i1 >= len(ot) or i1 <= i0:           # 到期时间之后还没有行情：这笔还没走完，不计
            dropped += 1
            continue
        entry = o[i0]
        tp = getattr(r, "tp", np.nan)
        sl = getattr(r, "sl", np.nan)
        tp = np.nan if tp is None else float(tp)
        sl = np.nan if sl is None else float(sl)
        hh, ll = h[i0:i1], l[i0:i1]
        if side > 0:
            hit_tp = hh >= tp if np.isfinite(tp) else np.zeros(len(hh), bool)
            hit_sl = ll <= sl if np.isfinite(sl) else np.zeros(len(hh), bool)
        else:
            hit_tp = ll <= tp if np.isfinite(tp) else np.zeros(len(hh), bool)
            hit_sl = hh >= sl if np.isfinite(sl) else np.zeros(len(hh), bool)
        k_tp = int(np.argmax(hit_tp)) if hit_tp.any() else len(hh)
        k_sl = int(np.argmax(hit_sl)) if hit_sl.any() else len(hh)
        if k_sl <= k_tp and k_sl < len(hh):                       # 同一根里都碰到 -> 按止损
            k, reason, px = k_sl, "sl", sl
            px = min(px, o[i0 + k]) if side > 0 else max(px, o[i0 + k])      # 跳空越过止损价：按开盘价成交
        elif k_tp < len(hh):
            k, reason, px = k_tp, "tp", tp
        else:
            k, reason, px = i1 - i0, "time", o[i1]
        gross = side * (px / entry - 1)
        seg_h, seg_l = h[i0:i0 + k + 1], l[i0:i0 + k + 1]
        fav = (seg_h.max() / entry - 1) if side > 0 else (1 - seg_l.min() / entry)
        adv = (1 - seg_l.min() / entry) if side > 0 else (seg_h.max() / entry - 1)
        d = r._asdict()
        d.update(entry_t=pd.Timestamp(ot[i0]), entry_px=float(entry), exit_t=pd.Timestamp(ot[min(i0 + k, len(ot) - 1)]),
                 exit_px=float(px), reason=reason, gross=float(gross), net=float(gross - cost),
                 hold_min=float(k), mfe=float(fav), mae=float(adv))
        out.append(d)
    T = pd.DataFrame(out)
    T.attrs["dropped"] = dropped
    return T


def no_overlap(spec: pd.DataFrame) -> pd.DataFrame:
    """同一品种同一时间只持一笔：按进场时间排序，前一笔到期之前的新信号跳过（用 exit_time 判断，保守）。"""
    keep, busy = [], {}
    for i, r in spec.sort_values("entry_time").iterrows():
        if r.asset in busy and pd.Timestamp(r.entry_time) < busy[r.asset]:
            continue
        busy[r.asset] = pd.Timestamp(r.exit_time)
        keep.append(i)
    return spec.loc[keep]


def summarize(T: pd.DataFrame) -> dict:
    if not len(T):
        return dict(n=0)
    net = T["net"].to_numpy()
    eq = np.cumsum(T.sort_values("exit_t")["net"].to_numpy())
    dd = float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())
    wins, losses = net[net > 0], net[net <= 0]
    return dict(n=int(len(T)), win_rate=float((net > 0).mean()), avg_net=float(net.mean()), median_net=float(np.median(net)),
                total_net=float(net.sum()), avg_gross=float(T["gross"].mean()),
                profit_factor=float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
                avg_win=float(wins.mean()) if len(wins) else 0.0, avg_loss=float(losses.mean()) if len(losses) else 0.0,
                max_drawdown=dd, avg_hold_min=float(T["hold_min"].mean()),
                t_stat=float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))) if len(net) > 1 and net.std(ddof=1) > 0 else 0.0)


def block_bootstrap(T: pd.DataFrame, block_hours: float = 8.0, n_boot: int = 4000, seed: int = 0) -> dict:
    """按进场时间分块（默认 8 小时一块，同一块里 BTC/SOL 和相邻信号一起抽）做自助抽样，给出平均净收益和胜率的 90% 区间，
    以及"平均净收益 <= 0"的概率 p_le0。块数太少（< 6）时区间没有意义，返回 n_blocks 供判断。"""
    if not len(T):
        return dict(n_blocks=0)
    blk = (pd.DatetimeIndex(T["entry_t"]).view("int64") // int(block_hours * 3600 * 1e9))
    groups = [g["net"].to_numpy() for _, g in T.groupby(blk)]
    rng = np.random.default_rng(seed)
    means, wrs = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        x = np.concatenate([groups[i] for i in pick])
        means.append(x.mean()); wrs.append((x > 0).mean())
    means, wrs = np.array(means), np.array(wrs)
    return dict(n_blocks=len(groups), avg_net_lo=float(np.quantile(means, .05)), avg_net_hi=float(np.quantile(means, .95)),
                win_lo=float(np.quantile(wrs, .05)), win_hi=float(np.quantile(wrs, .95)), p_le0=float((means <= 0).mean()))


def baselines(spec: pd.DataFrame, bars1m: dict, n_random: int = 200, seed: int = 0, **kw) -> dict:
    """同样的进出场时间，换成"全做多 / 全做空 / 随机方向"的结果，用来判断方向信号本身有没有贡献。
    止盈止损按方向镜像（保持与进场价的相对距离）。"""
    def flip(s: pd.DataFrame, side: np.ndarray) -> pd.DataFrame:
        s = s.copy()
        changed = side != s["side"].to_numpy()
        if "tp" in s and "sl" in s and changed.any():
            ref = simulate(s.assign(tp=np.nan, sl=np.nan), bars1m, **kw)
            px = dict(zip(zip(ref["asset"], ref["entry_time"]), ref["entry_px"]))
            e = np.array([px.get((a, t), np.nan) for a, t in zip(s["asset"], s["entry_time"])])
            tp, sl = s["tp"].to_numpy(float), s["sl"].to_numpy(float)
            s["tp"] = np.where(changed, 2 * e - tp, tp)
            s["sl"] = np.where(changed, 2 * e - sl, sl)
        s["side"] = side
        return s
    res = {}
    for name, sd in (("all_long", 1), ("all_short", -1)):
        res[name] = summarize(simulate(flip(spec, np.full(len(spec), sd)), bars1m, **kw))
    rng = np.random.default_rng(seed)
    avg, wr = [], []
    for _ in range(n_random):
        s = summarize(simulate(flip(spec, rng.choice([-1, 1], len(spec))), bars1m, **kw))
        if s.get("n"):
            avg.append(s["avg_net"]); wr.append(s["win_rate"])
    res["random"] = dict(avg_net_mean=float(np.mean(avg)), avg_net_q95=float(np.quantile(avg, .95)),
                         win_rate_mean=float(np.mean(wr)), win_rate_q95=float(np.quantile(wr, .95))) if avg else {}
    return res


def evaluate(spec: pd.DataFrame, bars1m: dict, one_position: bool = True, **kw) -> dict:
    """一步到位：可选"同品种同时只持一笔"，撮合，汇总，自助区间，方向基线。"""
    s = no_overlap(spec) if one_position else spec
    T = simulate(s, bars1m, **kw)
    return dict(summary=summarize(T), bootstrap=block_bootstrap(T), baselines=baselines(s, bars1m, **kw) if len(T) else {},
                dropped=T.attrs.get("dropped", 0), trades=T)
