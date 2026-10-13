"""Trade simulation for SOMA plans.

Each Turn / Shotgun signal is evaluated independently (overlapping trades are allowed),
so the stats measure signal quality, not a single-position account.

Rules
- Entry: next bar's open after the signal bar closes. Skipped if that open is already
  at/through the stop.
- Stop: static; fills on any touch (gap-through fills at the open). stop_mode="close"
  instead exits on a close beyond the stop, like the script's plan invalidation.
- Same bar touches the stop and an un-hit target: stop assumed first (conservative).
- Plan end: T3 hit, stop, plan expiry (Scalp 30/60, TrendRun 120/180 bars) or, on
  intraday charts, the new trading day -> remainder exits at that bar's open.
- Win = T1 reached before the stop / plan end. T2 / T3 reach rates are also recorded.
- R = results in units of actual risk |fill - stop|, net of round-trip costs.
  Three exit styles: all at T1, thirds at T1/T2/T3, all at T3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


LIMIT_WAIT_BARS = 5


def simulate(df: pd.DataFrame, sig: pd.DataFrame, is_intraday: bool, cost_price,
             stop_mode: str = "touch", entry_mode: str = "next_open",
             session_reset: bool = True) -> pd.DataFrame:
    """cost_price(fill) -> round-trip cost in price units.
    entry_mode: "next_open" = market at the next bar's open;
                "limit" = limit at the plan's ENTRY price, valid LIMIT_WAIT_BARS bars,
                cancelled if T1 or the stop trades first.
    session_reset: wipe the plan at a new trading day on intraday charts (script behavior)."""
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    new_day = sig.attrs["new_day"] if (is_intraday and session_reset) else np.zeros(len(df), bool)
    n = len(df)
    out = []
    for s in sig.itertuples(index=False):
        if s.engine not in ("Turn", "Shotgun") or np.isnan(s.stop) or np.isnan(s.t1):
            continue
        d = s.dir
        skip = lambda why: out.append(dict(time=s.time, engine=s.engine, dir=d, skipped=True, skip_reason=why))
        j0 = s.bar + 1
        if j0 >= n:
            continue
        if new_day[j0]:                          # plan is wiped at the new session before entry
            skip("session_reset")
            continue
        if entry_mode == "next_open":
            fill = o[j0]
        else:
            fill = None
            for j in range(j0, min(n, j0 + LIMIT_WAIT_BARS)):
                if j > j0 and new_day[j]:
                    break
                if d * (o[j] - s.entry) <= 0:     # opened at/through the entry: fill at the open
                    fill = o[j]
                elif (l[j] <= s.entry) if d == 1 else (h[j] >= s.entry):
                    fill = s.entry
                if fill is not None:
                    j0 = j
                    break
                if (h[j] >= s.t1) if d == 1 else (l[j] <= s.t1):
                    break                         # ran to T1 without us
            if fill is None:
                skip("limit_not_filled")
                continue
        risk = d * (fill - s.stop)
        if risk <= 0:
            skip("gap_through_stop")
            continue
        targets = [s.t1, s.t2, s.t3]
        hit = [None, None, None]
        for k, t in enumerate(targets):          # already through a target at the fill
            if d * (fill - t) >= 0:
                hit[k] = fill
        exit_px, reason, ambiguous, j = None, None, False, j0
        while j < n:
            if j > j0 and (j - s.bar > s.max_bars or new_day[j]):
                exit_px, reason = o[j], "timeout"
                break
            stop_touch = (l[j] <= s.stop) if d == 1 else (h[j] >= s.stop)
            stopped = stop_touch if stop_mode == "touch" else (d * (c[j] - s.stop) <= 0)
            pending = [k for k in range(3) if hit[k] is None and ((h[j] >= targets[k]) if d == 1 else (l[j] <= targets[k]))]
            if stopped:
                ambiguous = bool(pending)
                if stop_mode == "touch":
                    exit_px = o[j] if d * (o[j] - s.stop) <= 0 else s.stop
                else:
                    exit_px = c[j]
                reason = "stop"
                break
            for k in pending:
                hit[k] = targets[k]
            if hit[2] is not None:
                exit_px, reason = hit[2], "t3"
                break
            j += 1
        if exit_px is None:                       # data ended with the plan still open
            exit_px, reason = c[-1], "open"
        cost_r = cost_price(fill) / risk
        rr = lambda px: d * (px - fill) / risk
        exit_r = rr(exit_px)
        r_t1 = (rr(hit[0]) if hit[0] is not None else exit_r) - cost_r
        r_t3 = (rr(hit[2]) if hit[2] is not None else exit_r) - cost_r
        r_thirds = sum((rr(hit[k]) if hit[k] is not None else exit_r) for k in range(3)) / 3 - cost_r
        out.append(dict(
            time=s.time, engine=s.engine, dir=d, skipped=False, skip_reason="", entry_plan=s.entry, fill=fill,
            stop=s.stop, t1=s.t1, t2=s.t2, t3=s.t3, risk=risk, win=hit[0] is not None,
            hit_t2=hit[1] is not None, hit_t3=hit[2] is not None, exit=reason, bars=j - j0,
            ambiguous=ambiguous, r_all_t1=r_t1, r_thirds=r_thirds, r_all_t3=r_t3))
    return pd.DataFrame(out)


def continuation_stats(df: pd.DataFrame, sig: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """Continuation signals have no plan: did price move +1 ATR before -1 ATR
    (from the next bar's open) within `horizon` bars?"""
    o, h, l = (df[k].to_numpy(float) for k in ("open", "high", "low"))
    atr = sig.attrs["atr"]
    rows = []
    for s in sig[sig.engine == "Continuation"].itertuples(index=False):
        j0 = s.bar + 1
        a = atr[s.bar]
        if j0 >= len(df) or np.isnan(a):
            continue
        up, dn = o[j0] + s.dir * a, o[j0] - s.dir * a
        res = "none"
        for j in range(j0, min(len(df), j0 + horizon)):
            fav = h[j] >= up if s.dir == 1 else l[j] <= up
            adv = l[j] <= dn if s.dir == 1 else h[j] >= dn
            if adv:
                res = "adverse"
                break
            if fav:
                res = "favorable"
                break
        rows.append(dict(time=s.time, dir=s.dir, result=res))
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    t = trades[~trades.skipped] if len(trades) else trades
    if not len(t):
        return dict(trades=0)
    def pf(r):
        g, lo = r[r > 0].sum(), -r[r < 0].sum()
        return g / lo if lo > 0 else np.inf
    eq = t.sort_values("time")["r_thirds"].cumsum()
    return dict(
        trades=len(t),
        win_t1=t.win.mean(), hit_t2=t.hit_t2.mean(), hit_t3=t.hit_t3.mean(),
        stopped=(t.exit == "stop").mean(), timeout=(t.exit.isin(["timeout", "open"])).mean(),
        avgR_all_t1=t.r_all_t1.mean(), avgR_thirds=t.r_thirds.mean(), avgR_all_t3=t.r_all_t3.mean(),
        pf_thirds=pf(t.r_thirds), totalR_thirds=t.r_thirds.sum(),
        maxDD_R_thirds=(eq.cummax() - eq).max(),
        ambiguous=t.ambiguous.mean(), skipped=int(trades.skipped.sum()),
    )
