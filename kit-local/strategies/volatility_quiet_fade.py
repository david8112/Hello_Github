"""Fade stretched moves only while the model expects quiet hours, with every distance set by the model's predicted volatility.
Frozen by strategist "volatility" on the design data outputs/eval/live/train only (forecasts visible before 2026-11-17 12:00 UTC).

Idea. The one part of the KiT forecasts that stage 1 found informative is the volatility of the sampled paths: it ranks the
realised volatility of the next 4 hours. Volatility alone has no direction, so the direction comes from price: in hours that
turn out quiet, a move away from the starting price tends to come back; in hours that turn out busy it tends to keep going.
The rule therefore uses the model twice and its median never: (1) the regime: trade only when the predicted volatility of the
next 4 hours is below the volatility realised over the past 24 hours; (2) the geometry: entry trigger, take-profit and stop
are multiples of the predicted 4-hour sigma, and take-profit and stop are the SAME distance from the trigger (symmetric, so the
win rate is not bought with a small target and a wide stop).

The rule, to trade by hand (all times UTC):
 1. Moment of decision t0: a new 15m, 1h or 2h forecast of the asset becomes visible and the asset has no open position of this
    rule. Any bracket still pending from an earlier forecast is cancelled at t0 (replace = True).
 2. Predicted volatility V: for each scale (15m, 1h, 2h) take the newest forecast visible at t0 that is not older than 6 hours
    (visible_at >= t0 - 6h) and whose paths reach t0 + 4h. For each of its 16 sampled paths take the one-bar log returns of the
    closes (first bar from the anchor close) of the bars that overlap (t0, t0 + 4h], and the root mean square of them; average
    the 16 values; divide by sqrt(bar minutes); multiply by the scale's level factor (15m 0.85, 1h 0.75, 2h 0.70: the paths
    of the longer scales are too wide). V = geometric mean over the scales that are available. S = V * sqrt(240) is the
    predicted one-sigma move of the next 4 hours.
 3. Realised volatility B: root mean square of the log(close/open) of the 15-minute bars that closed in (t0 - 24h, t0], divided
    by sqrt(15). At least 76 of the 96 bars must exist, otherwise no trade.
 4. Regime: R = V / B. R >= 1.0 ("busy"): do nothing until the next forecast. R < 1.0 ("quiet"): place a bracket around the
    last price P (close of the last 1-minute bar at t0): upper trigger P*exp(+0.75*S), lower trigger P*exp(-0.75*S).
 5. The bracket is active from t0 until the first of: t0 + 4h, or the next new forecast of the asset. The first 1-minute bar
    (opening at or after t0) whose high reaches the upper trigger or whose low reaches the lower trigger starts the trade (if one
    bar does both: cancel). Upper trigger touched: SHORT; lower trigger touched: LONG. Entry at market right after that bar.
 6. Exits, fixed at entry: take-profit at P (the centre of the bracket), stop at 0.75*S beyond the trigger
    (trigger*exp(+0.75*S) for a short, trigger*exp(-0.75*S) for a long), time stop 4 hours after entry. While the trade's
    4-hour window runs, new forecasts are ignored (one position per asset).

Controls built in (use --set): quiet=follow turns every trade into the breakout twin on the same brackets (enter WITH the
touch, stop at P, no target); busy=fade or busy=follow trade the busy regime too; thr=99 removes the regime filter.

Honest status on the design data (46 hours, 7 trades in 4 eight-hour blocks, BTC and SOL usually together): +41.5 bp per trade
net, 6 of 7 trades won, both sides won; the breakout twin on the same brackets lost 68 bp per trade. This is far too little to
mean anything (random direction does as well in 18% of draws). The reason it is frozen at all is the bar history of the two
weeks before the forecasts (15-minute bars 2026-11-04 .. 11-17, no model): with PERFECT knowledge of which 4-hour windows turn
out quiet, this bracket earned about +16 bp per trade net and lost about 30 bp in the busy windows; without any regime
knowledge it lost about 5 bp. So everything depends on how well the model's "quiet" call keeps the busy windows out, which 46
hours cannot show. Expected on unseen data: close to zero, more likely a small plus than a large one; not adoptable before it
has at least 6 independent blocks.
"""
import numpy as np
import pandas as pd

NAME = "volatility_quiet_fade"
DESCRIPTION = ("When a new 15m/1h/2h forecast of an asset becomes visible and the asset has no open position of this rule: V = predicted "
               "volatility of the next 4 hours (mean over the 16 paths of the RMS one-bar log return of the bars overlapping the window, "
               "per sqrt(minute), times a level factor 0.85/0.75/0.70 for 15m/1h/2h, geometric mean over the newest forecast of each "
               "scale not older than 6 h); B = realised volatility of the past 24 hours (15-minute bars); S = V*sqrt(240). If V/B >= 1.0 "
               "(busy) do nothing. If V/B < 1.0 (quiet) put a bracket at P*exp(+-0.75*S) around the last price P, active for 4 hours or "
               "until the next new forecast. First 1-minute bar touching the upper trigger: short; lower trigger: long (market, right "
               "after that bar). Take-profit at P, stop 0.75*S beyond the trigger, time stop 4 hours after entry. One position per asset.")
PARAMS = {
    "k": 0.75,          # bracket half-width, in predicted 4-hour sigmas S
    "thr": 1.0,         # regime threshold on R = V / B: R >= thr is "busy", R < thr is "quiet"
    "quiet": "fade",    # what to do in the quiet regime: fade / follow / skip
    "busy": "skip",     # what to do in the busy regime: fade / follow / skip
    "wait_h": 4.0,      # hours a bracket stays active
    "hold_h": 4.0,      # time stop, hours after entry
    "vol_h": 4.0,       # window of the volatility forecast in hours (also the horizon of S)
    "tp_frac": 1.0,     # fade take-profit: fraction of the way from the trigger back to the centre P (1.0 = at P)
    "sl_mult": 1.0,     # fade stop: sl_mult * k * S beyond the trigger (1.0 = same distance as the take-profit)
    "replace": True,    # True: a newer forecast cancels a pending bracket; False: a pending bracket blocks newer forecasts until it expires
    "cal_15m": 0.85,    # level factors: realised / predicted volatility, medians on the design data
    "cal_1h": 0.75,
    "cal_2h": 0.70,
    "max_age_h": 6.0,   # a forecast older than this (since visible_at) is not read
    "base_h": 24.0,     # look-back of the realised volatility B in hours
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


def signals(ds, k=0.75, thr=1.0, quiet="fade", busy="skip", wait_h=4.0, hold_h=4.0, vol_h=4.0, tp_frac=1.0, sl_mult=1.0, replace=True,
            cal_15m=0.85, cal_1h=0.75, cal_2h=0.70, max_age_h=6.0, base_h=24.0):
    cal = {"15m": cal_15m, "1h": cal_1h, "2h": cal_2h}
    F = ds.forecasts
    F = F[F["scale"].isin(list(cal))]
    rows = []
    for asset in sorted(F["asset"].unique()):
        m1 = ds.bars(asset, "1m")
        ot = m1["open_time"].to_numpy("datetime64[ns]").astype("int64")
        ct = m1["t"].to_numpy("datetime64[ns]").astype("int64")
        hi, lo = m1["high"].to_numpy(float), m1["low"].to_numpy(float)
        times = [pd.Timestamp(t) for t in sorted(F[F["asset"] == asset]["visible_at"].unique())]
        free_at = None                                            # position open (or, with replace=False, bracket pending) until then
        for n, t0 in enumerate(times):
            if free_at is not None and t0 < free_at:
                continue
            V, B = vol_state(ds, F, asset, t0, vol_h, cal, max_age_h, base_h)
            P = ds.last_price(asset, t0)
            if not (np.isfinite(V) and np.isfinite(B) and B > 0 and np.isfinite(P)):
                continue
            R = V / B
            mode = busy if R >= thr else quiet
            if mode not in ("fade", "follow"):
                continue
            S = V * np.sqrt(vol_h * 60.0)                         # predicted one-sigma move over vol_h hours (log units)
            U, L = P * np.exp(k * S), P * np.exp(-k * S)
            end = t0 + pd.Timedelta(hours=wait_h)
            if replace:
                if n + 1 < len(times):
                    end = min(end, times[n + 1])                  # the next new forecast cancels this bracket
            else:
                free_at = end                                     # replace=False: the bracket blocks newer forecasts until it expires
            i0 = int(np.searchsorted(ot, t0.as_unit("ns").value, side="left"))
            i1 = int(np.searchsorted(ct, end.as_unit("ns").value, side="right"))
            for j in range(i0, i1):
                up, dn = hi[j] >= U, lo[j] <= L
                if up and dn:
                    break                                         # both triggers inside one minute: cancel
                if up or dn:
                    brk = 1 if up else -1
                    trig = U if up else L
                    te = pd.Timestamp(int(ct[j]))                 # close time of the bar that touched the trigger
                    row = dict(asset=asset, entry_time=te, exit_time=te + pd.Timedelta(hours=hold_h), tag=f"{mode} R={R:.2f} S={S * 1e4:.0f}bp")
                    if mode == "fade":
                        row.update(side=-brk, tp=trig * np.exp(-brk * tp_frac * k * S), sl=trig * np.exp(brk * sl_mult * k * S))
                    else:                                         # breakout twin: with the touch, stop at the centre, no target
                        row.update(side=brk, tp=np.nan, sl=P)
                    rows.append(row)
                    free_at = row["exit_time"]
                    break
    return pd.DataFrame(rows)
