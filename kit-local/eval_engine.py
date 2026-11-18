#!/usr/bin/env python
"""信号回测引擎：把"什么时候、哪个方向、怎么离场"的交易清单放到 1 分钟实际 K 线上逐笔撮合，并给出不自欺的统计。

撮合约定（全部保守处理，避免把回测做得比实盘好看）:
  - 进场：entry_time（可再加 latency_min 的反应时间）之后第一根 1 分钟线的开盘价（市价单）。
  - 离场：先到者为准 —— 止盈价 tp / 止损价 sl / 到期 exit_time（到期按之后第一根 1 分钟线开盘价）。
    同一根 1 分钟线里止盈止损都碰到时按止损成交；跳空越过止损价按开盘价成交。
    进场时止盈价已经在不利一侧（行情已经走过目标）则按进场价立即离场，只亏费用。
  - 费用：每边 fee_bps（OKX 永续吃单 5bp）加每边 slip_bps 滑点，进场离场各一次。不含资金费（每 8 小时一次，通常 ±1bp 量级）。
  - 收益按 1 倍名义本金、不复利：net = side * (exit/entry - 1) - 费用。杠杆只是把它乘倍数（同时放大亏损和爆仓风险）。
  - 到期时间之后还没有行情的交易不计（dropped）。

统计约定:
  - 相邻信号、BTC 和 SOL、同一批预测之间高度相关，独立样本远少于笔数。block_stats 按进场时间分块（块长不短于最长持仓时间，
    至少 8 小时），对"块"做刀切法（jackknife）得到平均净收益和胜率的 90% t 区间；不足 6 块时只给点估计，不给区间。
  - 方向基线：同样的进出场时间，全做多 / 全做空 / 按前 60 分钟动量 / 随机方向。随机方向是"每个时间簇（默认 2 小时内的
    信号算一簇，BTC 和 SOL 同簇）抽一个方向"，不是每笔独立抽，否则基线太窄、太容易被"跑赢"。

用法:
    import eval_engine as E
    bars = E.load_1m("outputs/eval/live")                 # {"BTC": df, "SOL": df}
    R = E.evaluate(spec, bars)                            # spec: asset, side, entry_time, exit_time[, tp, sl, tag]
    R["summary"], R["block"], R["baselines"], R["trades"]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEE_BPS = 5.0        # 每边吃单手续费（OKX 永续普通用户 taker 0.05%）
SLIP_BPS = 1.5       # 每边滑点（SOL 一个最小报价单位约 1.4bp）
MIN_BLOCKS = 6       # 少于这么多个独立时间块，不给置信区间


def _ns(x) -> pd.DatetimeIndex:
    """任何时间列统一成纳秒精度（parquet 读出来的可能是微秒，直接取 int64 会差 1000 倍）。"""
    return pd.DatetimeIndex(pd.to_datetime(x)).as_unit("ns")


def load_1m(dataset_dir: str | Path) -> dict:
    d = Path(dataset_dir)
    return {p.stem.split("_")[1]: pd.read_parquet(p) for p in sorted(d.glob("bars_*_1m.parquet"))}


def _arrays(bars1m: dict) -> dict:
    return {a: (_ns(b["open_time"]).asi8, b["open"].to_numpy(float), b["high"].to_numpy(float),
                b["low"].to_numpy(float), b["close"].to_numpy(float)) for a, b in bars1m.items()}


def _one(ot, o, h, l, side, i0, i1, tp, sl):
    """一笔交易在 [i0, i1) 这些 1 分钟线上的结果，返回 (k, reason, exit_px)。k 是离场所在的 bar 相对 i0 的偏移。"""
    entry = o[i0]
    if np.isfinite(tp) and ((side > 0 and tp <= entry) or (side < 0 and tp >= entry)):
        return 0, "tp_at_entry", entry
    hh, ll = h[i0:i1], l[i0:i1]
    n = len(hh)
    if side > 0:
        hit_tp = hh >= tp if np.isfinite(tp) else np.zeros(n, bool)
        hit_sl = ll <= sl if np.isfinite(sl) else np.zeros(n, bool)
    else:
        hit_tp = ll <= tp if np.isfinite(tp) else np.zeros(n, bool)
        hit_sl = hh >= sl if np.isfinite(sl) else np.zeros(n, bool)
    k_tp = int(np.argmax(hit_tp)) if hit_tp.any() else n
    k_sl = int(np.argmax(hit_sl)) if hit_sl.any() else n
    if k_sl <= k_tp and k_sl < n:                                # 同一根里都碰到 -> 按止损
        px = min(sl, o[i0 + k_sl]) if side > 0 else max(sl, o[i0 + k_sl])   # 跳空越过止损价：按开盘价成交
        return k_sl, "sl", px
    if k_tp < n:
        return k_tp, "tp", tp
    return n, "time", o[i1]


def simulate(spec: pd.DataFrame, bars1m: dict, fee_bps: float = FEE_BPS, slip_bps: float = SLIP_BPS,
             latency_min: float = 0.0) -> pd.DataFrame:
    """逐笔撮合。spec 列：asset, side(+1 多/-1 空), entry_time, exit_time, 可选 tp, sl（价格）, tag 等附加列（原样带出）。
    返回每笔：entry_t/entry_px/exit_t/exit_px/reason/gross/net/hold_min/mfe/mae。attrs["dropped"] 是没走完而不计的笔数。"""
    spec = spec.reset_index(drop=True)
    arr = _arrays(bars1m)
    cost = 2 * (fee_bps + slip_bps) / 1e4
    lat = pd.Timedelta(minutes=latency_min)
    out, dropped = [], 0
    for r in spec.itertuples(index=False):
        ot, o, h, l, c = arr[r.asset]
        side = int(r.side)
        t_in = (pd.Timestamp(r.entry_time) + lat).ceil("min").as_unit("ns").value
        t_out = pd.Timestamp(r.exit_time).ceil("min").as_unit("ns").value
        i0, i1 = int(np.searchsorted(ot, t_in, side="left")), int(np.searchsorted(ot, t_out, side="left"))
        if i0 >= len(ot) or i1 >= len(ot) or i1 <= i0:           # 到期时间之后还没有行情：这笔还没走完，不计
            dropped += 1
            continue
        tp = getattr(r, "tp", np.nan)
        sl = getattr(r, "sl", np.nan)
        tp = np.nan if tp is None else float(tp)
        sl = np.nan if sl is None else float(sl)
        entry = o[i0]
        k, reason, px = _one(ot, o, h, l, side, i0, i1, tp, sl)
        gross = side * (px / entry - 1)
        hi = max([entry, px] + ([float(h[i0:i0 + k].max())] if k > 0 else []))    # 只看离场之前走完的那些分钟线
        lo = min([entry, px] + ([float(l[i0:i0 + k].min())] if k > 0 else []))
        d = r._asdict()
        t_exit = int(ot[min(i0 + k, len(ot) - 1)])
        d.update(entry_t=pd.Timestamp(int(ot[i0])), entry_px=float(entry), exit_t=pd.Timestamp(t_exit), exit_px=float(px),
                 reason=reason, gross=float(gross), net=float(gross - cost), hold_min=float((t_exit - int(ot[i0])) / 6e10),
                 mfe=float(hi / entry - 1 if side > 0 else 1 - lo / entry),
                 mae=float(1 - lo / entry if side > 0 else hi / entry - 1))
        out.append(d)
    T = pd.DataFrame(out)
    T.attrs["dropped"] = dropped
    return T


def no_overlap(spec: pd.DataFrame) -> pd.DataFrame:
    """同一品种同一时间只持一笔：按进场时间排序，前一笔到期（exit_time）之前的新信号跳过。"""
    spec = spec.reset_index(drop=True)
    keep, busy = [], {}
    order = np.argsort(_ns(spec["entry_time"]).asi8, kind="stable")
    for i in order:
        r = spec.iloc[i]
        if r["asset"] in busy and pd.Timestamp(r["entry_time"]) < busy[r["asset"]]:
            continue
        busy[r["asset"]] = pd.Timestamp(r["exit_time"])
        keep.append(i)
    return spec.iloc[keep].reset_index(drop=True)


def summarize(T: pd.DataFrame) -> dict:
    """点估计。max_drawdown 只按已平仓交易的累计净收益算（不含持仓中的浮亏），逐分钟的见 mtm_drawdown。"""
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
                max_drawdown=dd, avg_hold_min=float(T["hold_min"].mean()))


def mtm_drawdown(T: pd.DataFrame, bars1m: dict, fee_bps: float = FEE_BPS, slip_bps: float = SLIP_BPS) -> float:
    """逐分钟盯市的最大回撤（1 倍名义本金、各笔相加）：持仓中按当时收盘价计浮动盈亏，平仓后计已实现净收益。"""
    if not len(T):
        return 0.0
    arr = _arrays(bars1m)
    t0, t1 = _ns(T["entry_t"]).asi8.min(), _ns(T["exit_t"]).asi8.max()
    grid = np.arange(t0, t1 + 60_000_000_000, 60_000_000_000)
    eq = np.zeros(len(grid))
    half = (fee_bps + slip_bps) / 1e4
    for r in T.itertuples(index=False):
        ot, o, h, l, c = arr[r.asset]
        a, b = pd.Timestamp(r.entry_t).as_unit("ns").value, pd.Timestamp(r.exit_t).as_unit("ns").value
        ia, ib = int(np.searchsorted(grid, a)), int(np.searchsorted(grid, b))
        j = np.clip(np.searchsorted(ot, grid[ia:ib], side="right") - 1, 0, len(ot) - 1)
        eq[ia:ib] += int(r.side) * (c[j] / r.entry_px - 1) - half
        eq[ib:] += r.net
    return float((np.maximum.accumulate(np.r_[0, eq]) - np.r_[0, eq]).max())


def _t_cdf(x: float, df: int) -> float:
    g = np.linspace(-80, 80, 400001)
    pdf = (1 + g * g / df) ** (-(df + 1) / 2)
    cdf = np.cumsum(pdf)
    return float(np.interp(x, g, cdf / cdf[-1]))


def _t_ppf(p: float, df: int) -> float:
    lo, hi = -80.0, 80.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _t_cdf(mid, df) < p else (lo, mid)
    return (lo + hi) / 2


def block_stats(T: pd.DataFrame, block_hours: float | None = None, min_blocks: int = MIN_BLOCKS) -> dict:
    """按进场时间分块后的刀切法区间。块长默认取 max(8 小时, 最长持仓时间向上取整到 4 小时)。
    返回 n_blocks、block_hours、平均净收益和胜率的点估计；块数 >= min_blocks 时再给 90% 区间（avg_net_lo/hi, win_lo/hi）
    和单侧 p 值 p_le0（"真实平均净收益 <= 0"假设下看到这么好结果的概率）。reliable=False 时不要引用区间。"""
    if not len(T):
        return dict(n_blocks=0, reliable=False)
    if block_hours is None:
        block_hours = max(8.0, 4.0 * np.ceil(float(T["hold_min"].max()) / 240.0))
    blk = _ns(T["entry_t"]).asi8 // int(block_hours * 3600 * 1e9)
    g = pd.DataFrame(dict(b=blk, net=T["net"].to_numpy(), win=(T["net"].to_numpy() > 0).astype(float))).groupby("b")
    s, w, n = g["net"].sum().to_numpy(), g["win"].sum().to_numpy(), g["net"].size().to_numpy()
    B = len(n)
    out = dict(n_blocks=int(B), block_hours=float(block_hours), avg_net=float(s.sum() / n.sum()),
               win_rate=float(w.sum() / n.sum()), blocks_positive=int((s > 0).sum()), reliable=bool(B >= min_blocks))
    if B < 2:
        return out
    def jk(x):
        th = (x.sum() - x) / (n.sum() - n)
        return float(np.sqrt((B - 1) / B * ((th - th.mean()) ** 2).sum()))
    se_m, se_w = jk(s), jk(w)
    out.update(se_avg_net=se_m, se_win=se_w)
    if B >= min_blocks:
        tq = _t_ppf(0.95, B - 1)
        out.update(avg_net_lo=out["avg_net"] - tq * se_m, avg_net_hi=out["avg_net"] + tq * se_m,
                   win_lo=out["win_rate"] - tq * se_w, win_hi=out["win_rate"] + tq * se_w,
                   p_le0=float(1 - _t_cdf(out["avg_net"] / se_m, B - 1)) if se_m > 0 else float(out["avg_net"] <= 0))
    return out


def block_bootstrap(T: pd.DataFrame, block_hours: float | None = None, **_) -> dict:
    """旧名字，等同 block_stats（原来的 8 小时行自助法在块少、持仓跨块时区间偏窄，已换成刀切法）。"""
    return block_stats(T, block_hours)


def _clusters(spec: pd.DataFrame, cluster_minutes: float) -> np.ndarray:
    """进场时间相邻不超过 cluster_minutes 的信号（不分品种）归为一簇，返回每行的簇编号。"""
    t = _ns(spec["entry_time"]).asi8
    order = np.argsort(t, kind="stable")
    cid = np.zeros(len(t), int)
    c, last = -1, None
    for i in order:
        if last is None or t[i] - last > cluster_minutes * 6e10:
            c += 1
        cid[i], last = c, t[i]
    return cid


def baselines(spec: pd.DataFrame, bars1m: dict, observed_avg_net: float | None = None, cluster_minutes: float = 120.0,
              n_random: int = 2000, seed: int = 0, **kw) -> dict:
    """同样的进出场时间，换方向的结果：all_long / all_short / mom60（方向 = 进场前 60 分钟涨跌）/ random（每个时间簇抽一个方向）。
    换方向时止盈止损各自按进场价镜像。给了 observed_avg_net 时 random 里多一个 p_ge：随机方向不低于它的比例（置换检验 p 值）。"""
    spec = spec.reset_index(drop=True)
    arr = _arrays(bars1m)
    lat = pd.Timedelta(minutes=kw.get("latency_min", 0.0))
    e, past = np.full(len(spec), np.nan), np.full(len(spec), np.nan)
    for i, r in enumerate(spec.itertuples(index=False)):
        ot, o = arr[r.asset][0], arr[r.asset][1]
        t_in = (pd.Timestamp(r.entry_time) + lat).ceil("min").as_unit("ns").value
        i0 = int(np.searchsorted(ot, t_in, side="left"))
        if i0 < len(ot):
            e[i] = o[i0]
            j = int(np.searchsorted(ot, int(ot[i0]) - 60 * 60_000_000_000, side="left"))
            past[i] = o[j]
    side0 = spec["side"].to_numpy(int)

    def with_side(side: np.ndarray) -> pd.DataFrame:
        s = spec.copy()
        changed = side != side0
        for col in ("tp", "sl"):
            if col in s:
                v = s[col].to_numpy(float)
                s[col] = np.where(changed, 2 * e - v, v)
        s["side"] = side
        s["row_id"] = np.arange(len(s))
        return s

    def net_by_row(side: np.ndarray) -> np.ndarray:
        T = simulate(with_side(side), bars1m, **kw)
        v = np.full(len(spec), np.nan)
        if len(T):
            v[T["row_id"].to_numpy(int)] = T["net"].to_numpy()
        return v

    long_net, short_net = net_by_row(np.ones(len(spec), int)), net_by_row(-np.ones(len(spec), int))
    ok = np.isfinite(long_net) & np.isfinite(short_net)

    def stat(v: np.ndarray) -> dict:
        v = v[np.isfinite(v)]
        return dict(n=int(len(v)), win_rate=float((v > 0).mean()), avg_net=float(v.mean())) if len(v) else dict(n=0)

    mom = np.where(e >= past, 1, -1)
    res = dict(all_long=stat(long_net[ok]), all_short=stat(short_net[ok]),
               mom60=stat(np.where(mom > 0, long_net, short_net)[ok]))
    cid = _clusters(spec, cluster_minutes)
    rng = np.random.default_rng(seed)
    draws = rng.choice([-1, 1], size=(n_random, cid.max() + 1))[:, cid]          # 每簇一个方向
    nets = np.where(draws > 0, long_net, short_net)[:, ok]
    if nets.shape[1]:
        avg, wr = nets.mean(axis=1), (nets > 0).mean(axis=1)
        res["random"] = dict(n_clusters=int(cid.max() + 1), avg_net_mean=float(avg.mean()), avg_net_q05=float(np.quantile(avg, .05)),
                             avg_net_q95=float(np.quantile(avg, .95)), win_rate_mean=float(wr.mean()),
                             win_rate_q95=float(np.quantile(wr, .95)))
        if observed_avg_net is not None:
            res["random"]["p_ge"] = float((avg >= observed_avg_net - 1e-15).mean())
    return res


def evaluate(spec: pd.DataFrame, bars1m: dict, one_position: bool = True, **kw) -> dict:
    """一步到位：可选"同品种同时只持一笔"，撮合，点估计，分块区间，方向基线，逐分钟回撤。kw 传给 simulate（fee_bps/slip_bps/latency_min）。"""
    s = no_overlap(spec) if one_position else spec.reset_index(drop=True)
    T = simulate(s, bars1m, **kw)
    S = summarize(T)
    if len(T):
        S["mtm_drawdown"] = mtm_drawdown(T, bars1m, kw.get("fee_bps", FEE_BPS), kw.get("slip_bps", SLIP_BPS))
    return dict(summary=S, block=block_stats(T), bootstrap=block_stats(T),
                baselines=baselines(s, bars1m, observed_avg_net=S.get("avg_net"), **kw) if len(T) else {},
                dropped=T.attrs.get("dropped", 0), trades=T)
