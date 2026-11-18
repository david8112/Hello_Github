"""Voting committee, natural direction (CONTROL, expected to fail). Frozen by strategist "committee" on the design data
outputs/eval/live/train only (forecasts visible before 2026-11-17 12:00 UTC).

Question this rule tests: following the model's direction lost on the design period. Does it still lose when the readings on
the dashboard agree with each other, i.e. when a strong majority of four equal voters points the same way?

The four voters (equal weight, each votes up / down / abstains):
  1-3. one voter per scale 15m, 1h, 2h: the direction in which the majority of that scale's 16 sampled paths moves from
       "now" to "now + 4 hours". Only the paths' own increments are used (never the stale price levels of the fan), so the
       delay between the anchor bar and the moment the forecast becomes visible does not enter the reading.
  4.   the tape: the sign of the price change over the last 60 minutes.

The rule, to trade by hand (all times UTC):
  1. Decision moment t0: a new 15m, 1h or 2h forecast of the asset becomes visible on the dashboard and this rule has no open
     position in that asset (a position stays open for exactly 4 hours; a decision exactly at its exit time is allowed).
  2. Scale voters: for each scale take the forecast with the latest anchor among those of that scale that are visible at t0 and
     became visible no more than 6 hours before t0. Read each of its sampled paths at t0 and at t0 + 4 h by linear
     interpolation of the logarithm of the path's closes between bar-close times (the anchor bar's close at the anchor time is
     the first point). up_share = share of paths that are higher at t0 + 4 h than at t0. The voter says "up" if
     up_share > 0.5, "down" if up_share < 0.5, abstains at exactly 0.5 (8 of 16). A scale with no such forecast, or whose
     paths end before t0 + 4 h, casts no vote. (model_sign = +1: the vote is the model's own direction.)
  3. Tape voter: "up" if the last closed 1-minute price at t0 is above the last closed 1-minute price at t0 - 60 min, "down"
     if below, abstains if equal.
  4. Trade only on a strong majority: LONG if at least 3 voters say "up" and at most 1 says "down"; SHORT if at least 3 say
     "down" and at most 1 says "up"; otherwise no trade.
  5. Enter at the market at t0, exit at the market at t0 + 4 h. No take-profit, no stop (the result is the plain 4-hour move).

What the committee does in practice: its direction is always the majority of the three scales. It trades when the scales are
unanimous (whatever the tape says) or when two scales agree, none or one disagrees, and the tape sides with the two; in every
other case it stays out. On the design data there were 25 decision moments (13 BTC, 12 SOL); the scales were unanimous at 7
of them (28%; three independent coins give 25%), and the rule produced 12 trades = about 3.3 per asset per day.

Honest status (design data, 44 hours, one position per asset, 5 bp fee + 1.5 bp slippage per side): 12 trades, 42% wins,
-10.7 bp per trade net, below "always long at the same times" (+26.8 bp). Holds of 2, 3, 5, 6, 8 hours: -9.6, -31.0, -18.0,
+1.1, -10.2 bp. The average over the six holds is about minus the round-trip cost: no gross edge. The committee is better than
each of its model members taken alone on the design data (-62, -23, -22 bp) only in the sense of losing less, and it is no
better than its tape member (60-minute momentum at the same times: -12.3 bp). It is frozen as the control for "trade when
everything lines up"; its mirror image (same voters, model votes inverted) is committee_consensus_fade_4h.py.
"""
import numpy as np
import pandas as pd

NAME = "committee_consensus_follow_4h"
DESCRIPTION = ("When a new 15m/1h/2h forecast of an asset becomes visible and no position is open: 4 equal voters = for each scale "
               "15m, 1h, 2h (latest-anchor forecast visible now and not more than 6 h on the dashboard) the direction of the majority "
               "of its 16 paths' own moves from now to now+4h (paths read by log-linear interpolation; 8/16 = abstain), plus the sign "
               "of the last 60 minutes' price change. Long if >= 3 voters say up and <= 1 says down, short if >= 3 say down and <= 1 "
               "says up, else no trade. Hold exactly 4 h, no TP/SL. Control: expected to fail.")
PARAMS = {
    "hold_h": 4.0,        # holding time in hours; also the horizon over which each path's move is read
    "model_sign": 1,      # +1 = scale voters vote the model's own direction, -1 = the opposite direction
    "tape_min": 60,       # lookback of the tape voter in minutes
    "min_agree": 3,       # at least this many voters on the trade's side
    "max_dissent": 1,     # at most this many voters on the other side
    "max_age_h": 6.0,     # a forecast votes only if it became visible no more than this many hours before the decision
    "lean": 0.5,          # a scale votes up if up_share > lean, down if up_share < 1 - lean, abstains in between
}
SCALES = ("15m", "1h", "2h")


def _path_moves(view, f, t0, t1):
    """Log move of every sampled path from t0 to t1; each path is read by linear interpolation of log close between its bar
    closes, starting from the anchor bar's close at the anchor time. None if [t0, t1] is not inside the path."""
    p = view.paths(f.fid)
    ts = np.r_[pd.Timestamp(f.anchor).as_unit("ns").value, pd.DatetimeIndex(p["fut_ts"]).as_unit("ns").asi8].astype(float)
    c = np.asarray(p["close"], float)
    lp = np.log(np.c_[np.full(len(c), float(p["prev_close"])), c])
    a, b = float(t0.as_unit("ns").value), float(t1.as_unit("ns").value)
    if a < ts[0] or b > ts[-1]:
        return None
    return np.array([np.interp(b, ts, row) - np.interp(a, ts, row) for row in lp])


def signals(ds, hold_h=4.0, model_sign=1, tape_min=60, min_agree=3, max_dissent=1, max_age_h=6.0, lean=0.5):
    cols = ["asset", "side", "entry_time", "exit_time", "tag"]
    F = ds.forecasts
    F = F[F.scale.isin(SCALES)]
    hold = pd.Timedelta(hours=hold_h)
    rows = []
    for asset in ds.assets:
        Fa = F[F.asset == asset]
        busy_until = None
        for t0 in sorted(Fa.visible_at.unique()):
            t0 = pd.Timestamp(t0)
            if busy_until is not None and t0 < busy_until:      # one position per asset
                continue
            view = ds.at(t0)                                     # nothing after the decision moment is read
            votes, names = [], []
            for s in SCALES:
                Fs = Fa[(Fa.scale == s) & (Fa.visible_at <= t0) & (Fa.visible_at >= t0 - pd.Timedelta(hours=max_age_h))]
                if not len(Fs):
                    continue
                f = Fs.sort_values("anchor").iloc[-1]
                mv = _path_moves(view, f, t0, t0 + hold)
                if mv is None:
                    continue
                up = float((mv > 0).mean())
                votes.append(int(model_sign) * (1 if up > lean else -1 if up < 1 - lean else 0))
                names.append(f"{s}:{up:.2f}")
            px = view.last_price(asset)
            past = view.last_price(asset, t0 - pd.Timedelta(minutes=tape_min))
            if np.isfinite(px) and np.isfinite(past) and px != past:
                votes.append(1 if px > past else -1)
                names.append(f"tape:{(px / past - 1) * 1e4:+.0f}bp")
            n_up, n_dn = sum(v > 0 for v in votes), sum(v < 0 for v in votes)
            side = 1 if (n_up >= min_agree and n_dn <= max_dissent) else -1 if (n_dn >= min_agree and n_up <= max_dissent) else 0
            if side == 0:
                continue
            rows.append(dict(asset=asset, side=side, entry_time=t0, exit_time=t0 + hold,
                             tag=f"{n_up}up{n_dn}dn " + " ".join(names)))
            busy_until = t0 + hold
    return pd.DataFrame(rows, columns=cols)
