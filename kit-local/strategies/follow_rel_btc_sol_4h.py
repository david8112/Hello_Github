"""Relative follower BTC vs SOL (control, least bad on design data). Frozen by strategist "follow" on outputs/eval/live/train only.

Removes the model's standing bearish drift by construction: only the DIFFERENCE between the two assets' readings is traded,
as a pair (long one, short the other, equal notional). The readings are re-anchored path increments, as in follow_reanchor_4h.
On the design data it lost slightly less than the costs; it is frozen as a control, not as a candidate.
"""
import numpy as np
import pandas as pd

NAME = "follow_rel_btc_sol_4h"
DESCRIPTION = ("Whenever a new 15m/1h/2h forecast of BTC or SOL becomes visible and no pair is open: for each asset pool the newest fresh "
               "(<= 60 min old) forecast of each scale (at least 2 scales per asset), z = median/std over the 16 paths of the path's own "
               "log increment from now to now+4h, averaged over the scales; if z_BTC > z_SOL go long BTC and short SOL, if z_BTC < z_SOL "
               "short BTC and long SOL; hold both legs 4 hours, no stop.")
PARAMS = {
    "scales": "15m,1h,2h",   # scales that are read
    "hold_h": 4.0,           # holding time in hours = horizon of the path increment that is read
    "fresh_min": 60.0,       # a forecast is used only while its visible_at is at most this many minutes old
    "min_scales": 2,         # at least this many scales per asset must be fresh and usable
    "s_min": 0.0,            # trade only if |z_BTC - z_SOL| > 0 and >= s_min
}


def _z(ds, f, t0, hold_h):
    """median / std (ddof=1) over paths of log P(t0 + hold) - log P(t0); each path is the piecewise-linear (in time) log close
    through (anchor, prev_close) and (bar close time, predicted close). None if [t0, t0 + hold] is not inside the path."""
    p = ds.paths(f.fid)
    ts = np.r_[pd.Timestamp(f.anchor).as_unit("ns").value, p["fut_ts"].asi8].astype(float)
    lc = np.log(np.c_[np.full(p["close"].shape[0], p["prev_close"]), p["close"]])
    x0 = float(pd.Timestamp(t0).as_unit("ns").value)
    x1 = x0 + hold_h * 3.6e12
    if x0 < ts[0] or x1 > ts[-1]:
        return None
    inc = np.array([np.interp(x1, ts, r) - np.interp(x0, ts, r) for r in lc])
    sd = inc.std(ddof=1)
    return float(np.median(inc) / sd) if sd > 0 else None


def _pooled(ds, F, asset, t0, hold_h, fresh_min, min_scales):
    """average z over the newest fresh forecast of each scale; None if fewer than min_scales are usable."""
    G = F[(F.asset == asset) & (F.visible_at <= t0) & (F.visible_at >= t0 - pd.Timedelta(minutes=fresh_min))]
    G = G.sort_values(["visible_at", "anchor"]).groupby("scale").tail(1)
    zs = [z for z in (_z(ds, f, t0, hold_h) for f in G.itertuples(index=False)) if z is not None]
    return float(np.mean(zs)) if len(zs) >= min_scales else None


def signals(ds, scales="15m,1h,2h", hold_h=4.0, fresh_min=60.0, min_scales=2, s_min=0.0):
    sc = [s.strip() for s in str(scales).split(",")]
    F = ds.forecasts
    F = F[F.scale.isin(sc) & F.asset.isin(["BTC", "SOL"])]
    rows, busy = [], None                                      # busy = exit time of the open pair
    for t0 in sorted(F["visible_at"].unique()):
        t0 = pd.Timestamp(t0)
        if busy is not None and t0 < busy:
            continue
        zb = _pooled(ds, F, "BTC", t0, hold_h, fresh_min, int(min_scales))
        zs = _pooled(ds, F, "SOL", t0, hold_h, fresh_min, int(min_scales))
        if zb is None or zs is None:
            continue
        s = zb - zs
        if s == 0 or abs(s) < s_min:
            continue
        busy = t0 + pd.Timedelta(hours=hold_h)
        side = 1 if s > 0 else -1
        tag = f"zB={zb:+.3f} zS={zs:+.3f}"
        rows.append(dict(asset="BTC", side=side, entry_time=t0, exit_time=busy, tag=tag))
        rows.append(dict(asset="SOL", side=-side, entry_time=t0, exit_time=busy, tag=tag))
    return pd.DataFrame(rows)
