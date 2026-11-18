"""Fade a lopsided fan as it is seen on the dashboard (control, expected to fail). Frozen by strategist "fade" on the design data
outputs/eval/live/train only (forecasts visible before 2026-11-17 12:00 UTC).

Stage 1 reported that lopsided fans (<= 20% or >= 80% of the paths on one side of the price) pointed the right way only 35.8% of
the time on the design period. This file is the traded form of that statement, read the way a person reads the dashboard: the
price LEVELS of the fan 4 hours from now against the price right now (so the reading includes whatever the market did between
the anchor bar and the moment the forecast became visible).

When it does NOT trade: no newly visible forecast is lopsided at the 20/80 level, or two newly visible forecasts of the asset are
lopsided in opposite directions, or a position of this rule is still open on the asset.

Honest status: on the design data it earned +12 bp per trade net with 50% wins on 8 trades, LESS than "always long at the same
times" (+17 bp), its short trades lost, it turned negative when entered 15 minutes late, and the sign flips with the holding time
(2 h and 6 h negative, 3 h and 8 h positive). It is frozen as a control for the stage-1 statement, not as a candidate.
"""
import numpy as np
import pandas as pd

NAME = "fade_lopsided_level_4h"
DESCRIPTION = ("Whenever a new 15m/1h/2h forecast of an asset becomes visible and the asset has no open position of this rule: for each "
               "forecast that became visible at that moment, count the share of its sampled paths whose close 4 hours from now (path "
               "interpolated linearly in time through the anchor close and the predicted closes) is ABOVE the current price (last closed "
               "1-minute close). Share <= 0.20 = fan lopsided down, share >= 0.80 = fan lopsided up. If at least one new forecast is "
               "lopsided and all lopsided ones agree, trade AGAINST the fan: long if it is lopsided down, short if lopsided up. Enter at "
               "the moment the forecast is visible, exit 4 hours later, no stop, no take-profit, one position per asset.")
PARAMS = {
    "scales": "15m,1h,2h",   # scales that are read (each forecast is read on its own, no pooling)
    "hold_h": 4.0,           # holding time in hours = how far ahead the fan is read
    "lopsided": 0.80,        # a fan is lopsided if the share of paths above the current price is >= lopsided or <= 1 - lopsided
}


def _share_above(ds, f, t0, hold_h, px):
    """share of the sampled paths whose close at t0 + hold is above px. Each path is the piecewise-linear (in time) log close
    through (anchor, prev_close) and (bar close time, predicted close). None if t0 + hold is beyond the end of the path."""
    p = ds.paths(f.fid)
    ts = np.r_[pd.Timestamp(f.anchor).as_unit("ns").value, p["fut_ts"].asi8].astype(float)
    lc = np.log(np.c_[np.full(p["close"].shape[0], p["prev_close"]), p["close"]])
    x0 = float(pd.Timestamp(t0).as_unit("ns").value)
    x1 = x0 + hold_h * 3.6e12
    if x0 < ts[0] or x1 > ts[-1]:
        return None
    lvl = np.array([np.interp(x1, ts, r) for r in lc])
    return float((lvl > np.log(px)).mean())


def signals(ds, scales="15m,1h,2h", hold_h=4.0, lopsided=0.80):
    sc = [s.strip() for s in str(scales).split(",")]
    F = ds.forecasts
    F = F[F.scale.isin(sc)]
    lo, hi = 1.0 - lopsided, lopsided
    rows = []
    for asset in sorted(F.asset.unique()):
        Fa = F[F.asset == asset]
        busy = None                                            # exit time of the open position of this asset
        for t0 in sorted(Fa.visible_at.unique()):
            t0 = pd.Timestamp(t0)
            if busy is not None and t0 < busy:
                continue
            px = ds.at(t0).last_price(asset)                   # last closed 1-minute close at the decision moment
            if not np.isfinite(px) or px <= 0:
                continue
            dirs, tags = [], []                                # direction of every lopsided fan that appeared at t0
            for f in Fa[Fa.visible_at == t0].sort_values("fid").itertuples(index=False):
                q = _share_above(ds, f, t0, hold_h, px)
                if q is None:
                    continue
                if q <= lo + 1e-9:
                    dirs.append(-1)
                    tags.append(f"{f.scale}:{q:.2f}")
                elif q >= hi - 1e-9:
                    dirs.append(+1)
                    tags.append(f"{f.scale}:{q:.2f}")
            if not dirs or len(set(dirs)) > 1:                 # nothing lopsided, or lopsided fans disagree: no trade
                continue
            busy = t0 + pd.Timedelta(hours=hold_h)
            rows.append(dict(asset=asset, side=-dirs[0], entry_time=t0, exit_time=busy, tag=" ".join(tags)))
    return pd.DataFrame(rows)
