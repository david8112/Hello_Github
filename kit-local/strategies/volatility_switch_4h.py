"""Regime switch on the model's predicted volatility: follow the last hour's move when the model expects busy hours, fade it when
the model expects quiet hours. Frozen by strategist "volatility" on the design data outputs/eval/live/train only (forecasts
visible before 2026-11-17 12:00 UTC). Frozen as a CONTROL: the plainest test of "volatility tells momentum from reversal".

The model's median is never read. Direction = the sign of the price move of the last 60 minutes (the same quantity as the
engine's mom60 baseline), kept in the busy regime and reversed in the quiet regime. So on identical entries and exits:
quiet=follow gives the always-follow twin (= mom60), busy=fade gives the always-fade twin, and both together give the inverted
switch.

The rule, to trade by hand (all times UTC):
 1. Moment of decision t0: a new 15m, 1h or 2h forecast of the asset becomes visible and the asset has no open position of this
    rule (a position stays open for exactly 4 hours).
 2. Predicted volatility V: for each scale (15m, 1h, 2h) take the newest forecast visible at t0 that is not older than 6 hours
    (visible_at >= t0 - 6h) and whose paths reach t0 + 4h. For each of its 16 sampled paths take the one-bar log returns of the
    closes (first bar from the anchor close) of the bars that overlap (t0, t0 + 4h], and the root mean square of them; average
    the 16 values; divide by sqrt(bar minutes); multiply by the scale's level factor (15m 0.85, 1h 0.75, 2h 0.70).
    V = geometric mean over the scales that are available.
 3. Realised volatility B: root mean square of the log(close/open) of the 15-minute bars that closed in (t0 - 24h, t0], divided
    by sqrt(15). At least 76 of the 96 bars must exist, otherwise no trade.
 4. Last-hour move: last price at t0 (close of the last 1-minute bar) against the last price at t0 - 60 minutes. Equal: no trade.
 5. R = V / B. R >= 1.0 (busy): trade WITH the last-hour move. R < 1.0 (quiet): trade AGAINST it.
 6. Enter at market at t0, exit at market 4 hours later. No stop, no take-profit.

Honest status on the design data (46 hours, 16 trades, 6 eight-hour blocks): +20.4 bp per trade net, 69% wins, always-fade twin
-0.1 bp, always-follow twin -25.9 bp. Do not trust it: (a) the whole difference to the always-fade twin is ONE forecast batch
(2026-11-16 13:59, the only busy call that was traded; BTC had moved +2 bp in the hour before, so the "direction" there was a
coin flip that came up right); (b) 13 of 16 trades were longs in a rebounding market; (c) it is a parameter cliff: look-back 30,
120 and 240 minutes all lose 17 to 26 bp per trade, holding 2 or 3 hours loses too. The bar history of the two weeks before the
forecasts says that even a PERFECT volatility forecast earns this switch only about +2 bp per trade after costs, and a forecast
of the model's quality (rank correlation about 0.5) about -7 bp. Expected on unseen data: a small loss.
"""
import numpy as np
import pandas as pd

NAME = "volatility_switch_4h"
DESCRIPTION = ("When a new 15m/1h/2h forecast of an asset becomes visible and the asset has no open position of this rule: V = predicted "
               "volatility of the next 4 hours (mean over the 16 paths of the RMS one-bar log return of the bars overlapping the window, "
               "per sqrt(minute), times a level factor 0.85/0.75/0.70 for 15m/1h/2h, geometric mean over the newest forecast of each "
               "scale not older than 6 h); B = realised volatility of the past 24 hours (15-minute bars). If V/B >= 1.0 (busy) trade in "
               "the direction of the price move of the last 60 minutes, if V/B < 1.0 (quiet) trade against it. Enter at market when the "
               "forecast is visible, exit 4 hours later, no stop, no take-profit, one position per asset.")
PARAMS = {
    "hold_h": 4.0,        # holding time in hours
    "vol_h": 4.0,         # window of the volatility forecast in hours
    "lookback_min": 60,   # the move that is followed / faded: last price now against last price this many minutes ago
    "thr": 1.0,           # regime threshold on R = V / B: R >= thr is "busy", R < thr is "quiet"
    "busy": "follow",     # what to do in the busy regime: follow / fade / skip
    "quiet": "fade",      # what to do in the quiet regime: follow / fade / skip
    "cal_15m": 0.85,      # level factors: realised / predicted volatility, medians on the design data
    "cal_1h": 0.75,
    "cal_2h": 0.70,
    "max_age_h": 6.0,     # a forecast older than this (since visible_at) is not read
    "base_h": 24.0,       # look-back of the realised volatility B in hours
}
STEP_MIN = {"15m": 15.0, "1h": 60.0, "2h": 120.0}


def _pred_sigma(ds, fid, scale, t0, vol_h):
    """mean over the sampled paths of the RMS one-bar log return (closes) of the bars overlapping (t0, t0 + vol_h], per sqrt(minute).
    nan if the paths do not reach t0 + vol_h."""
    p = ds.paths(fid)
    ts = p["fut_ts"].asi8                                         # close times of the predicted bars
    step = STEP_MIN[scale]
    x0 = pd.Timestamp(t0).as_unit("ns").value
    x1 = x0 + int(vol_h * 3.6e12)
    if ts[-1] < x1:
        return np.nan
    lc = np.log(np.c_[np.full(p["close"].shape[0], p["prev_close"]), p["close"]])
    r = np.diff(lc, axis=1)                                       # (paths, bars)
    m = (ts > x0) & (ts - int(step * 6e10) < x1)                  # bars that overlap the window
    if not m.any():
        return np.nan
    return float(np.sqrt((r[:, m] ** 2).mean(axis=1)).mean() / np.sqrt(step))


def _real_sigma(b15, t_lo, t_hi, min_bars):
    """RMS of log(close/open) of the 15-minute bars closing in (t_lo, t_hi], per sqrt(minute)."""
    b = b15[(b15["t"] > t_lo) & (b15["t"] <= t_hi)]
    if len(b) < min_bars:
        return np.nan
    r = np.log(b["close"].to_numpy(float) / b["open"].to_numpy(float))
    return float(np.sqrt((r ** 2).mean()) / np.sqrt(15.0))


def vol_state(ds, F, asset, t0, vol_h, cal, max_age_h, base_h):
    """(V, B) at t0: V = geometric mean over scales of the level-corrected predicted volatility of the next vol_h hours, read from the
    newest forecast of each scale visible at t0 and not older than max_age_h; B = realised volatility of the last base_h hours."""
    vs = []
    for s, c in cal.items():
        Fa = F[(F["asset"] == asset) & (F["scale"] == s) & (F["visible_at"] <= t0) & (F["visible_at"] >= t0 - pd.Timedelta(hours=max_age_h))]
        if not len(Fa):
            continue
        f = Fa.sort_values(["visible_at", "anchor"]).iloc[-1]
        v = _pred_sigma(ds, f["fid"], s, t0, vol_h)
        if np.isfinite(v) and v > 0:
            vs.append(v * c)
    if not vs:
        return np.nan, np.nan
    V = float(np.exp(np.mean(np.log(vs))))
    B = _real_sigma(ds.bars(asset, "15m"), t0 - pd.Timedelta(hours=base_h), t0, int(base_h * 4 * 0.8))
    return V, B


def signals(ds, hold_h=4.0, vol_h=4.0, lookback_min=60, thr=1.0, busy="follow", quiet="fade", cal_15m=0.85, cal_1h=0.75, cal_2h=0.70,
            max_age_h=6.0, base_h=24.0):
    cal = {"15m": cal_15m, "1h": cal_1h, "2h": cal_2h}
    F = ds.forecasts
    F = F[F["scale"].isin(list(cal))]
    rows = []
    for asset in sorted(F["asset"].unique()):
        free_at = None                                            # exit time of the open position of this asset
        for t0 in sorted(F[F["asset"] == asset]["visible_at"].unique()):
            t0 = pd.Timestamp(t0)
            if free_at is not None and t0 < free_at:
                continue
            V, B = vol_state(ds, F, asset, t0, vol_h, cal, max_age_h, base_h)
            if not (np.isfinite(V) and np.isfinite(B) and B > 0):
                continue
            R = V / B
            mode = busy if R >= thr else quiet
            if mode not in ("follow", "fade"):
                continue
            px = ds.last_price(asset, t0)
            past = ds.last_price(asset, t0 - pd.Timedelta(minutes=lookback_min))
            if not (np.isfinite(px) and np.isfinite(past)) or px == past:
                continue
            mom = 1 if px > past else -1
            free_at = t0 + pd.Timedelta(hours=hold_h)
            rows.append(dict(asset=asset, side=mom if mode == "follow" else -mom, entry_time=t0, exit_time=free_at, tag=f"{mode} R={R:.2f}"))
    return pd.DataFrame(rows)
