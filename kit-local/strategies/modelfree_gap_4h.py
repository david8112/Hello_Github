"""Model-free control: trade both perps in the direction of the BTC-minus-SOL 4-hour gap. No forecasts are read.
Frozen by strategist "modelfree" on the design data outputs/eval/pre only (bars 2026-09-01 .. 2026-10-13, no forecasts).

The rule, to trade by hand (all times UTC):
 1. Decision moments: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 (every step_min = 240 minutes from midnight, shifted by phase_min = 0).
 2. At a decision moment T take the last 1-minute close of BTC and of SOL at T and at T - 240 minutes (look_min) and the
    log returns rB = ln(BTC_T / BTC_T-240), rS = ln(SOL_T / SOL_T-240).
 3. gap = beta * rB - rS with beta = 1.4 (SOL moves about 1.4 times as much as BTC).
    gap > 0 (SOL did worse than BTC would suggest):  go LONG  BTC and LONG  SOL.
    gap < 0 (SOL did better than BTC would suggest): go SHORT BTC and SHORT SOL.
    gap = 0: no trade.
 4. Optional strength filter (off in the frozen rule: thr = 0): z = |gap| / (sigma_SOL * sqrt(look_min)), where sigma_SOL is the RMS of
    log(close/open) of the SOL 15-minute bars that closed in (T - 24h, T], divided by sqrt(15); trade only if z >= thr.
 5. Entry at market at T, exit at market at T + 240 minutes (hold_min). No take-profit, no stop. One position per asset: the next
    decision moment is the exit moment of the previous trade.

What the design data say (and do not say). This is the best model-free cell found at a 4-hour holding time, and it LOSES money:
 - On a 15-minute grid over the six weeks of 1-minute bars the sign of this gap had the only t-value near 2 among about 90 cells with
   holds of 1 to 12 hours (momentum/reversal at 4 look-backs x 4 holds, volatility-scaled extremes, quiet/busy regimes, time of day,
   session-open momentum, BTC leading SOL): +9.5 bp gross per 4 hours on SOL and +6.7 bp on BTC, same sign in both halves of the
   window and at the neighbouring holds of 2 and 8 hours. One t of 2 among 90 cells is what chance produces.
 - The gross effect is below the 13 bp round-trip cost: on the frozen clock the engine gives -3.6 bp net per trade (510 trades, 46.9%
   wins, 9.4 bp gross). No neighbour is positive: thr 0.25/0.5/0.75 give -10.9/-5.8/-12.3 bp, an 8-hour hold -10.6 bp, the clock
   shifted by 1/2/3 hours -1.6/-3.7/-9.7 bp, beta 1.0/1.8 -8.6/-5.1 bp.
 - On the 48 earlier days of hourly bars in the same directory (2026-07-15 .. 2026-08-31), which were not used to find the rule, the
   same clock made +1.6 bp gross and -11.4 bp net per trade (572 trades, 42% wins): the effect was not there.
 - It is not a spread effect (long SOL / short BTC earns nothing); it is a weak market-direction lean with no mechanism I would defend.
It is frozen as a frequency-matched control (six 4-hour trades a day per asset, direction from prices only): it shows what a rule
without skill costs at the holding time the model-based rules use, and what becomes of a "t = 2 in the landscape" out of sample.
Expected on unseen data: about -8 bp per trade (between minus the full cost and zero), win rate 44-48%.
"""
import numpy as np
import pandas as pd

NAME = "modelfree_gap_4h"
DESCRIPTION = ("Model-free control. Every 4 hours from 00:00 UTC: rB, rS = log returns of BTC and SOL over the last 240 minutes (1-minute "
               "closes); gap = 1.4*rB - rS. gap > 0: long BTC and long SOL; gap < 0: short BTC and short SOL. Market entry at the decision "
               "moment, market exit 240 minutes later, no take-profit, no stop, one position per asset. Reads no forecasts.")
PARAMS = {
    "look_min": 240,     # look-back of the two returns, minutes
    "hold_min": 240,     # holding time, minutes
    "step_min": 240,     # decision moments every step_min minutes counted from 00:00 UTC (must be >= hold_min for one position per asset)
    "phase_min": 0,      # shift of the decision moments, minutes after the step boundary
    "beta": 1.4,         # SOL move per unit of BTC move
    "thr": 0.0,          # minimum |gap| in units of sigma_SOL*sqrt(look_min); 0 = trade every decision moment
    "assets": "both",    # both / BTC / SOL: which perps are traded on the signal
    "fade": False,       # True flips every trade (mirror control)
}
MAX_STALE_NS = 5 * 60 * 1_000_000_000      # a price older than 5 minutes is treated as missing (data gap): no trade


def _asof_logclose(ts, lc, t_ns):
    """log of the last 1-minute close at or before each time in t_ns; nan where that bar is older than 5 minutes."""
    i = np.searchsorted(ts, t_ns, side="right") - 1
    ok = i >= 0
    out = np.full(len(t_ns), np.nan)
    j = np.clip(i, 0, len(ts) - 1)
    good = ok & (t_ns - ts[j] <= MAX_STALE_NS)
    out[good] = lc[j[good]]
    return out


def _sigma_15m(b15, t_ns):
    """RMS of log(close/open) of the 15-minute bars closing in (T - 24h, T], per sqrt(minute); nan with fewer than 76 bars."""
    ct = b15["t"].to_numpy("datetime64[ns]").astype("int64")
    r2 = np.log(b15["close"].to_numpy(float) / b15["open"].to_numpy(float)) ** 2
    cs = np.r_[0.0, np.cumsum(r2)]
    hi = np.searchsorted(ct, t_ns, side="right")
    lo = np.searchsorted(ct, t_ns - 24 * 3600 * 1_000_000_000, side="right")
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt((cs[hi] - cs[lo]) / n) / np.sqrt(15.0)
    return np.where(n >= 76, s, np.nan)


def signals(ds, look_min=240, hold_min=240, step_min=240, phase_min=0, beta=1.4, thr=0.0, assets="both", fade=False):
    if step_min < hold_min:
        raise ValueError("step_min must be >= hold_min (one position per asset)")
    px = {}
    for a in ("BTC", "SOL"):
        m = ds.bars(a, "1m")
        if not len(m):
            return pd.DataFrame(columns=["asset", "side", "entry_time", "exit_time", "tag"])
        px[a] = (m["t"].to_numpy("datetime64[ns]").astype("int64"), np.log(m["close"].to_numpy(float)))
    t_first = max(px["BTC"][0][0], px["SOL"][0][0])
    t_last = pd.Timestamp(ds.now).as_unit("ns").value
    step = step_min * 60 * 1_000_000_000
    k0 = -(-(t_first + look_min * 60 * 1_000_000_000 - phase_min * 60 * 1_000_000_000) // step)      # first grid point with a full look-back
    T = np.arange(k0 * step + phase_min * 60 * 1_000_000_000, t_last + 1, step, dtype="int64")
    if not len(T):
        return pd.DataFrame(columns=["asset", "side", "entry_time", "exit_time", "tag"])
    look = look_min * 60 * 1_000_000_000
    rB = _asof_logclose(*px["BTC"], T) - _asof_logclose(*px["BTC"], T - look)
    rS = _asof_logclose(*px["SOL"], T) - _asof_logclose(*px["SOL"], T - look)
    gap = beta * rB - rS
    ok = np.isfinite(gap) & (gap != 0)
    z = np.full(len(T), np.nan)
    if thr > 0:
        sig = _sigma_15m(ds.bars("SOL", "15m"), T)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.abs(gap) / (sig * np.sqrt(look_min))
        ok &= np.isfinite(z) & (z >= thr)
    trade = ("BTC", "SOL") if assets == "both" else (assets,)
    rows = []
    for i in np.flatnonzero(ok):
        side = int(np.sign(gap[i])) * (-1 if fade else 1)
        t = pd.Timestamp(int(T[i]))
        for a in trade:
            rows.append(dict(asset=a, side=side, entry_time=t, exit_time=t + pd.Timedelta(minutes=hold_min),
                             tag=f"gap={gap[i] * 1e4:.0f}bp rB={rB[i] * 1e4:.0f} rS={rS[i] * 1e4:.0f}"))
    return pd.DataFrame(rows, columns=["asset", "side", "entry_time", "exit_time", "tag"])
