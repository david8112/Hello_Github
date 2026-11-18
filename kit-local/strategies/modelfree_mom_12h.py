"""Model-free trend control: at 00:00 and 12:00 UTC go with the last 12 hours, hold 12 hours. No forecasts are read.
Frozen by strategist "modelfree" on the design data outputs/eval/pre only (1-minute bars 2026-09-01 .. 2026-10-13 and the hourly
bars of the same directory back to 2026-07-15; no forecasts).

The rule, to trade by hand (all times UTC), separately for BTC and for SOL:
 1. Decision moments: 00:00 and 12:00 (every step_min = 720 minutes from midnight, shifted by phase_min = 0).
 2. At a decision moment T take the asset's last 1-minute close at T and at T - 720 minutes (look_min).
    Price now above the price 12 hours ago: go LONG. Below: go SHORT. Exactly equal: no trade.
 3. Entry at market at T, exit at market at T + 720 minutes (hold_min), where the next decision is taken. No take-profit, no stop.
    One position per asset, 2 trades per asset per day.

Why this one. Of about 185 model-free variant evaluations (momentum and reversal at several look-backs and holds, volatility-scaled
extremes, quiet/busy regimes, time of day, session-open momentum, BTC leading SOL, dip in a trend), nothing clears the 13 bp
round-trip cost with any reliability. The 12-hour momentum held for 12 hours is the least bad, because the longest allowed hold makes
the move large against the cost and because it is the only cell whose gross sign held in every third of the 91 days of hourly bars in
the design directory: SOL +24.7 bp gross per 12 hours (hourly grid, day-block t = 2.5, thirds +7.5/+40.7/+25.5), BTC +9.3 bp
(t = 1.3, thirds -7.9/+20.2/+15.5). After costs and averaged over all 12 possible clock phases that is about +12 bp per trade on SOL
and -4 bp on BTC, +4 bp together.

Design numbers of the frozen clock: engine, 1-minute window 2026-09-02 .. 2026-10-13: +22.6 bp net per trade, 166 trades, 50% wins,
profit factor 1.47, first half +43 bp, second half +1.4 bp. Hourly closes of the 48 days before (2026-07-15 .. 2026-08-31, not used
to find the rule): +11.6 bp net, 188 trades, 47% wins, BTC -12.2, SOL +35.4. Read this as a control, not as an edge:
 (a) the frozen clock (00:00/12:00) is the natural one (daily candle, funding times) and was fixed before looking, but on the design
     data it turned out to be the luckiest of the 12 phases (91 days: +17 bp per trade for both assets together against +4 bp for
     the average phase; engine: the clock shifted by 3 hours gives -4.6 bp, by 6 hours +2.8 bp, by 9 hours +21.6 bp). The design
     result therefore overstates even the in-sample effect, and the phase is a parameter cliff;
 (b) BTC has no net effect at any look-back from 6 to 24 hours or hold from 6 to 12 hours when averaged over phases (-4 to -12 bp);
 (c) on SOL the phase-averaged net is positive for look-backs of 8 to 18 hours at a 12-hour hold (+3 to +12 bp), about zero at an
     8-hour hold and negative at a 6-hour hold or with a 24-hour look-back (engine, 24-hour look-back: -4.0 bp for both assets);
 (d) the design months drifted up: in the 48 earlier days "always long" on the same clock earned +12.6 bp against the rule's +11.6;
 (e) the profit is a handful of trend days: 3 of 42 days carry 87% of the engine result (20 of 42 days are positive), and the rule
     wins only half of its trades;
 (f) it was picked after about 185 looks at the same data, which alone is worth several bp of upward bias.
Expected on unseen data: about zero (between -10 and +10 bp per trade), win rate 45-50%. One month of unseen bars has a standard
error of roughly 16 bp per trade for this rule, so it can neither confirm nor refute an effect of a few bp; a good unseen month would
most likely be one or two trend days (the 2026-11-12..15 sell-off is such a candidate) and must be checked for that before it is
believed. The live window (26 hours) holds at most 2 decision moments: this rule can only be tested on the historical bars.
Controls built in (use --set): fade=true is the mirror; phase_min shifts the clock; look_min / hold_min+step_min are the neighbours.
"""
import numpy as np
import pandas as pd

NAME = "modelfree_mom_12h"
DESCRIPTION = ("Model-free trend control, per asset (BTC, SOL): at 00:00 and 12:00 UTC compare the last 1-minute close with the close 720 "
               "minutes earlier; higher: long, lower: short, equal: no trade. Market entry at the decision moment, market exit 720 minutes "
               "later, no take-profit, no stop, one position per asset. Reads no forecasts.")
PARAMS = {
    "look_min": 720,     # look-back of the return whose sign is traded, minutes
    "hold_min": 720,     # holding time, minutes
    "step_min": 720,     # decision moments every step_min minutes counted from 00:00 UTC (must be >= hold_min: one position per asset)
    "phase_min": 0,      # shift of the decision moments, minutes after the step boundary
    "fade": False,       # True flips every trade (mirror control)
}
MIN_NS = 60 * 1_000_000_000
MAX_STALE_NS = 5 * MIN_NS                  # a price older than 5 minutes is treated as missing (data gap): no trade


def _asof_logclose(ts, lc, t_ns):
    """log of the last 1-minute close at or before each time in t_ns; nan where that bar is older than 5 minutes."""
    i = np.searchsorted(ts, t_ns, side="right") - 1
    j = np.clip(i, 0, len(ts) - 1)
    good = (i >= 0) & (t_ns - ts[j] <= MAX_STALE_NS)
    out = np.full(len(t_ns), np.nan)
    out[good] = lc[j[good]]
    return out


def signals(ds, look_min=720, hold_min=720, step_min=720, phase_min=0, fade=False):
    if step_min < hold_min:
        raise ValueError("step_min must be >= hold_min (one position per asset)")
    cols = ["asset", "side", "entry_time", "exit_time", "tag"]
    rows = []
    t_last = pd.Timestamp(ds.now).as_unit("ns").value
    step, phase = step_min * MIN_NS, phase_min * MIN_NS
    for a in ("BTC", "SOL"):
        m = ds.bars(a, "1m")
        if not len(m):
            continue
        ts, lc = m["t"].to_numpy("datetime64[ns]").astype("int64"), np.log(m["close"].to_numpy(float))
        k0 = -(-(ts[0] + look_min * MIN_NS - phase) // step)                      # first decision moment with a full look-back
        T = np.arange(k0 * step + phase, t_last + 1, step, dtype="int64")
        if not len(T):
            continue
        r = _asof_logclose(ts, lc, T) - _asof_logclose(ts, lc, T - look_min * MIN_NS)
        for i in np.flatnonzero(np.isfinite(r) & (r != 0)):
            t = pd.Timestamp(int(T[i]))
            rows.append(dict(asset=a, side=int(np.sign(r[i])) * (-1 if fade else 1), entry_time=t,
                             exit_time=t + pd.Timedelta(minutes=hold_min), tag=f"r{look_min}={r[i] * 1e4:.0f}bp"))
    return pd.DataFrame(rows, columns=cols)
