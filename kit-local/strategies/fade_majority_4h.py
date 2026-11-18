"""Fade the model's own direction from the current moment (control / least-bad fade). Frozen by strategist "fade" on the design
data outputs/eval/live/train only (forecasts visible before 2026-11-17 12:00 UTC).

What is faded: not the stale price levels of the fan, but the direction of each sampled path from "now" to "now + 4 hours"
(path increments, so the delay between the anchor bar and the moment the forecast becomes visible does not matter). If a clear
majority of the paths of a newly visible forecast go one way, the rule takes the other side for 4 hours.

When it does NOT trade: the newly visible forecasts have no clear majority (6 to 10 of 16 paths up), or two newly visible
forecasts of the same asset have clear majorities in opposite directions, or a position of this rule is still open on the asset.

Honest status: on the design data this looked good at this horizon (+29 bp per trade net, 71% wins, 14 trades), but about 70%
of its gross profit is what "always long at the same times" earned, a third of the net profit is one forecast batch
(2026-11-16 13:59 UTC), only 4 of the 14 trades were shorts (2 won), and the sign is positive only for holds of 3 to 5 hours
(2 h, 2.5 h, 6 h and 8 h are negative). It was the best-looking of about 40 variants tried on 46 hours of data. It is frozen as
a pre-registered test of "the model's direction is worse than a coin", not as a finding; the expectation is that it fails.
"""
import numpy as np
import pandas as pd

NAME = "fade_majority_4h"
DESCRIPTION = ("Whenever a new 15m/1h/2h forecast of an asset becomes visible and the asset has no open position of this rule: for each "
               "forecast that became visible at that moment, count the share of its sampled paths whose own log close rises from now to "
               "now+4h (path interpolated linearly in time through the anchor close and the predicted closes). Share <= 0.32 (at most 5 of "
               "16 paths up) = model says down, share >= 0.68 (at least 11 of 16 up) = model says up. If at least one new forecast is "
               "one-sided and all one-sided ones agree, trade AGAINST the model: long if it says down, short if it says up. Enter at the "
               "moment the forecast is visible, exit 4 hours later, no stop, no take-profit, one position per asset.")
PARAMS = {
    "scales": "15m,1h,2h",   # scales that are read (each forecast is read on its own, no pooling)
    "hold_h": 4.0,           # holding time in hours = horizon over which the path direction is read
    "one_sided": 0.68,       # a forecast is one-sided if the share of paths going up is >= one_sided or <= 1 - one_sided
}


def _up_share(ds, f, t0, hold_h):
    """share of the sampled paths whose log close at t0 + hold is above their own log close at t0. Each path is the piecewise-linear
    (in time) log close through (anchor, prev_close) and (bar close time, predicted close). None if [t0, t0 + hold] is not covered."""
    p = ds.paths(f.fid)
    ts = np.r_[pd.Timestamp(f.anchor).as_unit("ns").value, p["fut_ts"].asi8].astype(float)
    lc = np.log(np.c_[np.full(p["close"].shape[0], p["prev_close"]), p["close"]])
    x0 = float(pd.Timestamp(t0).as_unit("ns").value)
    x1 = x0 + hold_h * 3.6e12
    if x0 < ts[0] or x1 > ts[-1]:
        return None
    inc = np.array([np.interp(x1, ts, r) - np.interp(x0, ts, r) for r in lc])
    return float((inc > 0).mean())


def signals(ds, scales="15m,1h,2h", hold_h=4.0, one_sided=0.68):
    sc = [s.strip() for s in str(scales).split(",")]
    F = ds.forecasts
    F = F[F.scale.isin(sc)]
    lo, hi = 1.0 - one_sided, one_sided
    rows = []
    for asset in sorted(F.asset.unique()):
        Fa = F[F.asset == asset]
        busy = None                                            # exit time of the open position of this asset
        for t0 in sorted(Fa.visible_at.unique()):
            t0 = pd.Timestamp(t0)
            if busy is not None and t0 < busy:
                continue
            dirs, tags = [], []                                # model direction of every one-sided forecast that appeared at t0
            for f in Fa[Fa.visible_at == t0].sort_values("fid").itertuples(index=False):
                q = _up_share(ds, f, t0, hold_h)
                if q is None:
                    continue
                if q <= lo + 1e-9:
                    dirs.append(-1)
                    tags.append(f"{f.scale}:{q:.2f}")
                elif q >= hi - 1e-9:
                    dirs.append(+1)
                    tags.append(f"{f.scale}:{q:.2f}")
            if not dirs or len(set(dirs)) > 1:                 # nothing one-sided, or one-sided forecasts disagree: no trade
                continue
            busy = t0 + pd.Timedelta(hours=hold_h)
            rows.append(dict(asset=asset, side=-dirs[0], entry_time=t0, exit_time=busy, tag=" ".join(tags)))
    return pd.DataFrame(rows)
