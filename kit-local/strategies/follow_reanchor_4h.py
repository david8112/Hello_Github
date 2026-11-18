"""Careful directional follower (control). Frozen by strategist "follow" on design data outputs/eval/live/train only.

Reads the model the way a careful user would: never the stale price levels, only the path increments from "now" to
"now + hold", pooled over the scales that have just been refreshed. No de-drifting, no stops. On the design data this
rule LOST (see the panel answer); it is frozen as the honest control for "following the forecast, done carefully".
"""
import numpy as np
import pandas as pd

NAME = "follow_reanchor_4h"
DESCRIPTION = ("Whenever a new 15m/1h/2h forecast of an asset becomes visible: pool the newest fresh (<= 60 min old) forecast of each "
               "scale (at least 2 scales), z = median/std over the 16 paths of the path's own log increment from now to now+4h, "
               "average z over the scales, go long if the average is > 0 and short if < 0, hold 4 hours, no stop, one position per asset.")
PARAMS = {
    "scales": "15m,1h,2h",   # scales that are read
    "hold_h": 4.0,           # holding time in hours = horizon of the path increment that is read
    "fresh_min": 60.0,       # a forecast is used only while its visible_at is at most this many minutes old
    "min_scales": 2,         # at least this many scales must be fresh and usable
    "z_min": 0.0,            # trade only if |average z| > 0 and >= z_min
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


def signals(ds, scales="15m,1h,2h", hold_h=4.0, fresh_min=60.0, min_scales=2, z_min=0.0):
    sc = [s.strip() for s in str(scales).split(",")]
    F = ds.forecasts
    F = F[F.scale.isin(sc)]
    rows = []
    for asset in sorted(F.asset.unique()):
        busy = None                                            # exit time of the open position of this asset
        for t0 in sorted(F.loc[F.asset == asset, "visible_at"].unique()):
            t0 = pd.Timestamp(t0)
            if busy is not None and t0 < busy:
                continue
            z = _pooled(ds, F, asset, t0, hold_h, fresh_min, int(min_scales))
            if z is None or z == 0 or abs(z) < z_min:
                continue
            busy = t0 + pd.Timedelta(hours=hold_h)
            rows.append(dict(asset=asset, side=1 if z > 0 else -1, entry_time=t0, exit_time=busy, tag=f"z={z:+.3f}"))
    return pd.DataFrame(rows)
