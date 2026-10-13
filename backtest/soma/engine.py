"""Bar-by-bar Python port of Soma_Trendrun.pine (signal + plan logic only).

Mirrors the Pine script's Turn, Shotgun and Continuation engines, the structure
zones, session levels (LIS / PMZ / ORB) and higher-timeframe edges that feed the
Shotgun targets. Drawing/visual-only code (bands, boxes, labels, tables) is not
ported. Every signal is decided on a closed bar with no future data.

Usage:
    sig = run_engine(df, market="crypto", tf_sec=86400, tick=0.001)
    sig is a DataFrame of one row per Turn / Shotgun / Continuation signal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"

# Core settings (Pine constants)
SUPER_FAST_LEN, FAST_LEN, SLOW_LEN, TREND_LEN, ATR_LEN = 9, 13, 22, 44, 14
REQ_EMA_STRUCTURE = False
REQ_EMA_TURNS = True
TURN_SETUP_VALIDITY_BARS = 8
TURN_COOLDOWN_BARS = 4
REQ_EMA_SHOTGUN = False
REQ_TURN_CONF_SHOTGUN = False
TURN_CONF_LOOKBACK = 3
REQ_ZONE_TOUCH_SHOTGUN = True
ZONE_TOUCH_LOOKBACK = 20
SHOTGUN_COOLDOWN_BARS = 3
CONT_PULLBACK_LOOKBACK = 3
CONT_COOLDOWN_BARS = 6
CONT_MAX_EXTENSION_ATR = 1.75
AGGRESSIVE_BODY_BREAK = False
USE_ZONE_BOUNDARY_STOP = False
USE_CLASSIC_ZONES = False
INVALIDATE_ZONE_ON_CLOSE = True
SQZ_LEN = 20

PREMARKET_START = 7 * 60 + 30
REGULAR_OPEN = 9 * 60 + 30
REGULAR_CLOSE = 16 * 60
PMZ_RESET_TIME = 15 * 60 + 55
ORB_END = 9 * 60 + 45
TARGET_MIN_SPACING_TICKS = 2
RECENT_SWING_LOOKBACK = 20

TF_D, TF_W, TF_M, TF_12M, TF_4H = 86400, 604800, 2628000, 31536000, 14400


@dataclass
class Mode:
    scalp: bool = False
    cooldown_from_signal: bool = True   # "Last Fired Signal" (default) vs TOS "Last Setup"

    @property
    def turn_stop_atr(self): return 0.75 if self.scalp else 1.25
    @property
    def turn_t_atr(self): return (0.5, 1.0, 1.5) if self.scalp else (1.0, 2.0, 4.0)
    @property
    def shotgun_stop_atr(self): return 0.5 if self.scalp else 1.0
    @property
    def shotgun_t_r(self): return (0.5, 1.0, 1.5) if self.scalp else (1.0, 2.0, 4.0)
    @property
    def turn_max_bars(self): return 30 if self.scalp else 120
    @property
    def shotgun_max_bars(self): return 60 if self.scalp else 180


# ----------------------------------------------------------------------------- Pine-equivalent series helpers
def shift(a: np.ndarray, k: int = 1, fill=np.nan) -> np.ndarray:
    out = np.full_like(a, fill, dtype=a.dtype if a.dtype == bool else float)
    if k < len(a):
        out[k:] = a[:-k] if k else a
    return out


def pine_ema(x: np.ndarray, n: int) -> np.ndarray:
    """ta.ema: seeded with the SMA of the first n values."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    alpha = 2.0 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def pine_rma(x: np.ndarray, n: int) -> np.ndarray:
    """ta.rma: seeded with the SMA of the first n values."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    alpha = 1.0 / n
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rolling_max(x, n):
    return pd.Series(x).rolling(n, min_periods=n).max().to_numpy()


def rolling_min(x, n):
    return pd.Series(x).rolling(n, min_periods=n).min().to_numpy()


def pine_linreg(src: np.ndarray, n: int) -> np.ndarray:
    """ta.linreg(src, n, 0): least-squares line value at the newest bar."""
    out = np.full(len(src), np.nan)
    if len(src) < n:
        return out
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    w = np.lib.stride_tricks.sliding_window_view(src, n)
    ym = w.mean(axis=1)
    slope = ((w - ym[:, None]) * (x - xm)).sum(axis=1) / denom
    out[n - 1:] = ym + slope * (n - 1 - xm)
    return out


def f_tick(x: float, t: float) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    return math.floor(x / t + 0.5) * t      # Pine math.round: ties round up


def nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ----------------------------------------------------------------------------- calendar keys
def calendar_keys(idx: pd.DatetimeIndex, market: str) -> dict:
    """Trading-day / week / month / year / 4h period keys, matching TradingView's
    session conventions: CME days start 18:00 ET, crypto days start 00:00 UTC."""
    utc = idx.tz_convert("UTC")
    if market == "cme":
        et = idx.tz_convert(NY)
        tday = (et + pd.Timedelta(hours=6)).normalize().tz_localize(None)   # 18:00 ET -> next date
        mins_into = ((et - (et + pd.Timedelta(hours=6)).normalize() + pd.Timedelta(hours=6))
                     .total_seconds() // 60).astype(int)  # minutes since the 18:00 ET open
        four = tday.astype("int64") // 86_400_000_000_000 * 10 + (np.asarray(mins_into) // 240)
    else:
        tday = utc.normalize().tz_localize(None)
        four = utc.floor("4h").tz_localize(None).astype("int64")
    tday = pd.DatetimeIndex(tday)
    iso = tday.isocalendar()
    return {
        "day": tday.astype("int64").to_numpy(),
        "week": (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy(),
        "month": (tday.year * 100 + tday.month).to_numpy(),
        "year": tday.year.to_numpy(),
        "4h": np.asarray(four),
    }


def prev_period_hl(keys: np.ndarray, high: np.ndarray, low: np.ndarray):
    """request.security(tf, [high[1], low[1]], lookahead_on): H/L of the previous
    completed period for every bar."""
    s = pd.DataFrame({"k": keys, "h": high, "l": low})
    order = pd.unique(keys)
    g = s.groupby("k", sort=False).agg(h=("h", "max"), l=("l", "min")).reindex(order)
    prev_h = pd.Series(g["h"].shift(1).to_numpy(), index=order)
    prev_l = pd.Series(g["l"].shift(1).to_numpy(), index=order)
    return prev_h.reindex(keys).to_numpy(), prev_l.reindex(keys).to_numpy()


# ----------------------------------------------------------------------------- engine
def run_engine(df: pd.DataFrame, market: str, tf_sec: int, tick: float,
               mode: Mode | None = None) -> pd.DataFrame:
    """df: tz-aware DatetimeIndex (bar open time), columns open/high/low/close.
    Returns one row per fired signal with its plan (entry/stop/targets)."""
    mode = mode or Mode()
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    safe_tick = tick if tick > 0 else 0.01
    is_intraday = tf_sec < TF_D

    # --- vector indicators
    ema9, ema13 = pine_ema(c, SUPER_FAST_LEN), pine_ema(c, FAST_LEN)
    ema22, ema44 = pine_ema(c, SLOW_LEN), pine_ema(c, TREND_LEN)
    c1 = shift(c)
    tr = np.where(np.isnan(c1), h - l, np.maximum(h - l, np.maximum(abs(h - c1), abs(l - c1))))
    atr = pine_rma(tr, ATR_LEN)
    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    ema_bull = (ema9 > ema13) & (ema13 > ema22)
    ema_bear = (ema9 < ema13) & (ema13 < ema22)

    h1, l1 = shift(h), shift(l)
    hh2, hh3 = rolling_max(h1, 2), rolling_max(h1, 3)
    ll2, ll3 = rolling_min(l1, 2), rolling_min(l1, 3)

    sqz_mid = (rolling_max(h, SQZ_LEN) + rolling_min(l, SQZ_LEN)) / 2 + pine_ema(c, SQZ_LEN)
    sqz = pine_linreg(c - sqz_mid / 2, SQZ_LEN)
    recent_swing_high = shift(rolling_max(h, RECENT_SWING_LOOKBACK))
    recent_swing_low = shift(rolling_min(l, RECENT_SWING_LOOKBACK))

    # --- time
    et = df.index.tz_convert(NY)
    bar_min = (et.hour * 60 + et.minute).to_numpy()
    keys = calendar_keys(df.index, market)
    new_day = np.r_[False, keys["day"][1:] != keys["day"][:-1]]
    blk5 = (df.index.asi8 // 1_000_000) // 300_000

    # --- higher-timeframe edges
    def edges(key, has):
        if not has:
            return np.full(n, np.nan), np.full(n, np.nan)
        return prev_period_hl(keys[key], h, l)

    pdh, pdl = edges("day", tf_sec <= TF_D)
    pwh, pwl = edges("week", tf_sec <= TF_W)
    pmh, pml = edges("month", tf_sec <= TF_M)
    pyh, pyl = edges("year", tf_sec <= TF_12M)
    p4h, p4l = edges("4h", tf_sec <= TF_4H)
    lvl = lambda lo, hi, f: lo + (hi - lo) * f
    d_lo_e, d_up_e = lvl(pdl, pdh, 0.113), lvl(pdl, pdh, 0.887)
    w_lo_e, w_up_e = lvl(pwl, pwh, 0.113), lvl(pwl, pwh, 0.887)
    m_lo_e, m_up_e = lvl(pml, pmh, 0.113), lvl(pml, pmh, 0.887)
    y_lo_e, y_up_e = lvl(pyl, pyh, 0.113), lvl(pyl, pyh, 0.887)
    fh_lo, fh_up = lvl(p4l, p4h, 0.113), lvl(p4l, p4h, 0.887)

    # --- structure (vector part)
    up, dn = c > o, c < o
    R = (h1 < h) & up
    D = (l1 > l) & dn
    o1, o2, cl1, cl2 = shift(o), shift(o, 2), c1, shift(c, 2)
    Rp = shift(R, fill=False)
    RB = Rp & (o1 <= c) & (c <= cl1)
    RBp = shift(RB, fill=False)
    RBB = RBp & (o2 <= c) & (c <= cl2)
    Dp = shift(D, fill=False)
    DB = Dp & (o1 >= c) & (c >= cl1)
    DBp = shift(DB, fill=False)
    DBB = DBp & (o2 >= c) & (c >= cl2)
    RBBp, DBBp = shift(RBB, fill=False), shift(DBB, fill=False)
    RBR = RBp & up & (h >= hh2)
    RBBR = RBBp & up & (h >= hh3)
    DBD = DBp & dn & (l <= ll2)
    DBBD = DBBp & dn & (l <= ll3)
    DBR = DBp & up & (h >= hh2)
    DBBR = DBBp & up & (h >= hh3)
    RBD = RBp & dn & (l <= ll2)
    RBBD = RBBp & dn & (l <= ll3)
    bull_struct = (RBR | RBBR | DBR | DBBR) & (~np.bool_(REQ_EMA_STRUCTURE) | ema_bull)
    bear_struct = (DBD | DBBD | RBD | RBBD) & (~np.bool_(REQ_EMA_STRUCTURE) | ema_bear)
    bull_f_low = np.where(RBBR | DBBR, ll3, ll2)
    bull_f_high = np.where(RBBR | DBBR, hh3, hh2)
    bear_f_low = np.where(DBBD | RBBD, ll3, ll2)
    bear_f_high = np.where(DBBD | RBBD, hh3, hh2)
    new_dem_low = bull_f_low
    new_dem_high = bull_f_high if USE_CLASSIC_ZONES else np.maximum(bull_f_low, l)
    new_sup_low = bear_f_low if USE_CLASSIC_ZONES else np.minimum(bear_f_high, h)
    new_sup_high = bear_f_high

    # continuation pullback flags (vector)
    ema13_1, ema44_1 = shift(ema13), shift(ema44)
    bull_pb = ((l1 <= ema13_1) & (c1 > ema44_1)).astype(float)
    bear_pb = ((h1 >= ema13_1) & (c1 < ema44_1)).astype(float)
    recent_bull_pb = rolling_max(bull_pb, CONT_PULLBACK_LOOKBACK) > 0
    recent_bear_pb = rolling_max(bear_pb, CONT_PULLBACK_LOOKBACK) > 0
    ll2c = rolling_min(l1, 2)

    # --- state
    dem_lo = dem_hi = sup_lo = sup_hi = np.nan
    lis = np.nan
    blk5_open = np.nan
    pm_hi, pm_lo, pm_above_lis = 0.0, 1_000_000.0, False
    orb_hi = orb_lo = np.nan
    up_age = dn_age = 10000
    since_turn_edge = since_turn_sig = 100000
    since_cont_edge = since_cont_sig = 100000
    since_sg_edge = since_sg_sig = 100000
    dem_touch_age = sup_touch_age = 10000
    prev_q_up_only = prev_q_dn_only = False
    prev_raw_cont_l = prev_raw_cont_s = False
    prev_raw_sg_l = prev_raw_sg_s = False
    turn_up_hist = np.zeros(n, bool)
    turn_dn_hist = np.zeros(n, bool)
    t_stop_atr = mode.turn_stop_atr
    t1a, t2a, t3a = mode.turn_t_atr
    s1r, s2r, s3r = mode.shotgun_t_r
    spacing = safe_tick * TARGET_MIN_SPACING_TICKS
    rows = []

    for i in range(n):
        # ---- structure zones
        dem_invalid = INVALIDATE_ZONE_ON_CLOSE and c[i] < dem_lo
        sup_invalid = INVALIDATE_ZONE_ON_CLOSE and c[i] > sup_hi
        if bull_struct[i]:
            dem_lo, dem_hi = new_dem_low[i], new_dem_high[i]
        elif dem_invalid:
            dem_lo = dem_hi = np.nan
        if bear_struct[i]:
            sup_hi, sup_lo = new_sup_high[i], new_sup_low[i]
        elif sup_invalid:
            sup_hi = sup_lo = np.nan

        # ---- LIS
        bm = bar_min[i]
        if is_intraday and ((REGULAR_CLOSE <= bm < REGULAR_CLOSE + 5) if tf_sec < 300 else bm == REGULAR_CLOSE):
            lis = c[i]
        # ---- PMZ
        if i == 0 or blk5[i] != blk5[i - 1]:
            blk5_open = o[i]
        in_pm = is_intraday and PREMARKET_START <= bm < REGULAR_OPEN
        if bm == PMZ_RESET_TIME:
            pm_hi, pm_lo = 0.0, 1_000_000.0
        elif in_pm:
            pm_hi, pm_lo = max(pm_hi, h[i]), min(pm_lo, l[i])
        if in_pm:
            pm_above_lis = bool(blk5_open >= lis)
        pm_rng = pm_hi - pm_lo
        pmz_valid = is_intraday and pm_hi > pm_lo and pm_hi > 0 and pm_lo < 1_000_000
        pmz_hi_v = pm_hi - pm_rng * 0.20 if pm_above_lis else pm_lo + pm_rng * 0.40
        pmz_lo_v = pm_hi - pm_rng * 0.40 if pm_above_lis else pm_lo + pm_rng * 0.20
        v_pmz_hi = pmz_hi_v if pmz_valid else np.nan
        v_pmz_lo = pmz_lo_v if pmz_valid else np.nan
        v_pm_hi = pm_hi if pmz_valid else np.nan
        v_pm_lo = pm_lo if pmz_valid else np.nan
        # ---- ORB
        in_orb = is_intraday and REGULAR_OPEN <= bm < ORB_END
        if bm == REGULAR_OPEN:
            orb_hi, orb_lo = h[i], l[i]
        elif new_day[i]:
            orb_hi = orb_lo = np.nan
        elif in_orb:
            orb_hi, orb_lo = max(orb_hi, h[i]) if not nan(orb_hi) else np.nan, \
                min(orb_lo, l[i]) if not nan(orb_lo) else np.nan

        # ---- Turn
        s0, s1, s2 = sqz[i], (sqz[i - 1] if i >= 1 else np.nan), (sqz[i - 2] if i >= 2 else np.nan)
        raw_dn = s0 > 0 and s0 < s1 and s2 < s1
        raw_up = s0 < 0 and s0 > s1 and s2 > s1
        up_age = 0 if raw_up else min(up_age + 1, 10000)
        dn_age = 0 if raw_dn else min(dn_age + 1, 10000)
        rec_up = up_age <= TURN_SETUP_VALIDITY_BARS and up_age < dn_age
        rec_dn = dn_age <= TURN_SETUP_VALIDITY_BARS and dn_age < up_age
        e22_up = i >= 1 and ema22[i] > ema22[i - 1]
        e22_dn = i >= 1 and ema22[i] < ema22[i - 1]
        tr_bull = mode.scalp or (c[i] > ema44[i] and e22_up)
        tr_bear = mode.scalp or (c[i] < ema44[i] and e22_dn)
        e9_up = i >= 1 and ema9[i] > ema9[i - 1]
        e9_dn = i >= 1 and ema9[i] < ema9[i - 1]
        q_up = rec_up and (not REQ_EMA_TURNS or ema_bull[i]) and i >= 1 and c[i] > h[i - 1] and e9_up and tr_bull
        q_dn = rec_dn and (not REQ_EMA_TURNS or ema_bear[i]) and i >= 1 and c[i] < l[i - 1] and e9_dn and tr_bear
        q_up_only, q_dn_only = q_up and not q_dn, q_dn and not q_up
        up_edge, dn_edge = q_up_only and not prev_q_up_only, q_dn_only and not prev_q_dn_only
        prev_q_up_only, prev_q_dn_only = q_up_only, q_dn_only
        prev_since = since_turn_edge
        since_turn_edge = 0 if (up_edge or dn_edge) else min(since_turn_edge + 1, 100000)
        turn_ready = (since_turn_sig >= TURN_COOLDOWN_BARS) if mode.cooldown_from_signal \
            else (prev_since >= TURN_COOLDOWN_BARS)
        turn_up, turn_dn = up_edge and turn_ready, dn_edge and turn_ready
        since_turn_sig = 0 if (turn_up or turn_dn) else min(since_turn_sig + 1, 100000)
        turn_up_hist[i], turn_dn_hist[i] = turn_up, turn_dn
        if turn_up or turn_dn:
            d = 1 if turn_up else -1
            e, a = c[i], atr[i]
            rows.append(dict(
                bar=i, engine="Turn", dir=d, entry=e,
                stop=f_tick(e - d * a * t_stop_atr, safe_tick),
                t1=f_tick(e + d * a * t1a, safe_tick), t2=f_tick(e + d * a * t2a, safe_tick),
                t3=f_tick(e + d * a * t3a, safe_tick), max_bars=mode.turn_max_bars))

        # ---- Continuation
        ext_bull = (c[i] - ema22[i]) / max(atr[i], safe_tick) if not nan(atr[i]) else np.nan
        ext_bear = (ema22[i] - c[i]) / max(atr[i], safe_tick) if not nan(atr[i]) else np.nan
        raw_cl = bool(i >= 1 and ema_bull[i] and ema_bull[i - 1] and c[i] > ema44[i] and e22_up
                      and recent_bull_pb[i] and c[i] > hh2[i] and up[i] and ext_bull <= CONT_MAX_EXTENSION_ATR)
        raw_cs = bool(i >= 1 and ema_bear[i] and ema_bear[i - 1] and c[i] < ema44[i] and e22_dn
                      and recent_bear_pb[i] and c[i] < ll2c[i] and dn[i] and ext_bear <= CONT_MAX_EXTENSION_ATR)
        cl_edge, cs_edge = raw_cl and not prev_raw_cont_l, raw_cs and not prev_raw_cont_s
        prev_raw_cont_l, prev_raw_cont_s = raw_cl, raw_cs
        prev_since = since_cont_edge
        since_cont_edge = 0 if (cl_edge or cs_edge) else min(since_cont_edge + 1, 100000)
        cont_ready = (since_cont_sig >= CONT_COOLDOWN_BARS) if mode.cooldown_from_signal \
            else (prev_since >= CONT_COOLDOWN_BARS)
        cont_l, cont_s = cl_edge and cont_ready, cs_edge and cont_ready
        since_cont_sig = 0 if (cont_l or cont_s) else min(since_cont_sig + 1, 100000)
        if cont_l or cont_s:
            rows.append(dict(bar=i, engine="Continuation", dir=1 if cont_l else -1, entry=c[i],
                             stop=np.nan, t1=np.nan, t2=np.nan, t3=np.nan, max_bars=0, atr=atr[i]))

        # ---- Shotgun
        prev_dem_age, prev_sup_age = dem_touch_age, sup_touch_age
        dem_touch = not nan(dem_lo) and l[i] <= dem_hi and h[i] >= dem_lo
        sup_touch = not nan(sup_lo) and h[i] >= sup_lo and l[i] <= sup_hi
        dem_touch_age = 0 if dem_touch else min(dem_touch_age + 1, 10000)
        sup_touch_age = 0 if sup_touch else min(sup_touch_age + 1, 10000)
        if i < 2:
            continue
        lz = (not REQ_ZONE_TOUCH_SHOTGUN) or prev_dem_age <= ZONE_TOUCH_LOOKBACK
        sz = (not REQ_ZONE_TOUCH_SHOTGUN) or prev_sup_age <= ZONE_TOUCH_LOOKBACK
        hl, lh = l[i - 1] > l[i - 2], h[i - 1] < h[i - 2]
        lb = (h[i] > body_hi[i - 1]) if AGGRESSIVE_BODY_BREAK else (c[i] > body_hi[i - 1])
        sb = (l[i] < body_lo[i - 1]) if AGGRESSIVE_BODY_BREAK else (c[i] < body_lo[i - 1])
        lo_i = max(0, i - TURN_CONF_LOOKBACK + 1)
        l_turn_conf = turn_up_hist[lo_i:i + 1].any()
        s_turn_conf = turn_dn_hist[lo_i:i + 1].any()
        lf = (not REQ_EMA_SHOTGUN or ema_bull[i]) and (not REQ_TURN_CONF_SHOTGUN or l_turn_conf)
        sf = (not REQ_EMA_SHOTGUN or ema_bear[i]) and (not REQ_TURN_CONF_SHOTGUN or s_turn_conf)
        raw_l = lz and hl and lb and lf
        raw_s = sz and lh and sb and sf
        raw_l_only, raw_s_only = raw_l and not raw_s, raw_s and not raw_l
        l_edge, s_edge = raw_l_only and not prev_raw_sg_l, raw_s_only and not prev_raw_sg_s
        prev_raw_sg_l, prev_raw_sg_s = raw_l_only, raw_s_only
        prev_since = since_sg_edge
        since_sg_edge = 0 if (l_edge or s_edge) else min(since_sg_edge + 1, 100000)
        sg_ready = (since_sg_sig >= SHOTGUN_COOLDOWN_BARS) if mode.cooldown_from_signal \
            else (prev_since >= SHOTGUN_COOLDOWN_BARS)
        sg_l, sg_s = l_edge and sg_ready, s_edge and sg_ready
        since_sg_sig = 0 if (sg_l or sg_s) else min(since_sg_sig + 1, 100000)
        if not (sg_l or sg_s):
            continue

        d = 1 if sg_l else -1
        e = body_hi[i - 1] if sg_l else body_lo[i - 1]
        wick = l[i - 1] if sg_l else h[i - 1]
        zone_stop = dem_lo if sg_l else sup_hi
        valid_zone = not nan(zone_stop) and ((sg_l and zone_stop < e) or (sg_s and zone_stop > e))
        base_stop = zone_stop if (USE_ZONE_BOUNDARY_STOP and valid_zone) else wick
        min_dist = atr[i] * mode.shotgun_stop_atr
        raw_stop = min(base_stop, e - min_dist) if sg_l else max(base_stop, e + min_dist)
        stop = f_tick(raw_stop, safe_tick)
        risk = abs(e - stop)
        base = [v_pmz_lo, v_pmz_hi, v_pm_lo, v_pm_hi, orb_lo, orb_hi, lis, fh_lo[i], fh_up[i],
                d_lo_e[i], d_up_e[i]]
        if sg_l:
            def above(thr, extra):
                best = 1_000_000_000.0
                for v in base + extra:
                    if not nan(v) and v > thr:
                        best = min(best, v)
                return best
            r1 = above(e, [w_lo_e[i], w_up_e[i], pdh[i], sup_lo, recent_swing_high[i]])
            r2 = above(r1 + spacing, [w_lo_e[i], w_up_e[i], pdh[i], sup_hi, recent_swing_high[i]])
            r3 = above(r2 + spacing, [m_lo_e[i], m_up_e[i], y_lo_e[i], y_up_e[i], sup_hi])
            fb = [e + risk * s1r, e + risk * s2r, e + risk * s3r]
            c1_, c2_, c3_ = (r if r < 500_000_000 else f for r, f in zip((r1, r2, r3), fb))
            t1 = max(c1_, fb[0])
            t2 = max(max(c2_, t1 + spacing), fb[1])
            t3 = max(max(c3_, t2 + spacing), fb[2])
        else:
            def below(thr, extra):
                best = -1_000_000_000.0
                for v in base + extra:
                    if not nan(v) and v < thr:
                        best = max(best, v)
                return best
            r1 = below(e, [w_lo_e[i], w_up_e[i], pdl[i], dem_hi, recent_swing_low[i]])
            r2 = below(r1 - spacing, [w_lo_e[i], w_up_e[i], pdl[i], dem_lo, recent_swing_low[i]])
            r3 = below(r2 - spacing, [m_lo_e[i], m_up_e[i], y_lo_e[i], y_up_e[i], dem_lo])
            fb = [e - risk * s1r, e - risk * s2r, e - risk * s3r]
            c1_, c2_, c3_ = (r if r > -500_000_000 else f for r, f in zip((r1, r2, r3), fb))
            t1 = min(c1_, fb[0])
            t2 = min(min(c2_, t1 - spacing), fb[1])
            t3 = min(min(c3_, t2 - spacing), fb[2])
        rows.append(dict(bar=i, engine="Shotgun", dir=d, entry=e, stop=stop,
                         t1=f_tick(t1, safe_tick), t2=f_tick(t2, safe_tick), t3=f_tick(t3, safe_tick),
                         max_bars=mode.shotgun_max_bars))

    out = pd.DataFrame(rows, columns=["bar", "engine", "dir", "entry", "stop", "t1", "t2", "t3",
                                      "max_bars", "atr"])
    if len(out):
        out["time"] = df.index[out["bar"].to_numpy()]
    out.attrs["new_day"] = new_day
    out.attrs["atr"] = atr
    return out
