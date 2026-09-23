#!/usr/bin/env python3
"""
IBKR TrendQuality Web Dashboard with SQLite bar cache + Fresh BUY alerts + Call/Put Walls

Features
- Pulls historical bars from IBKR for symbols passed on command line
- Stores/updates bars in SQLite
- Computes TrendQuality BUY/SELL state on: 1d, 1h, 30m, 5m, 1m
- Serves a compact transposed HTML dashboard at http://127.0.0.1:8050
- Fresh BUY alert strip at the top
- Symbol-level Call Wall / Put Wall option metrics in column headers

Install
  pip install ib_insync pandas numpy flask

Run
  python3 ibkr_tq_web_dashboard_with_walls.py --symbols KORU SOXL TQQQ NVDA AMD AAPL TSLA SPY --paper

Disable option walls if needed
  python3 ibkr_tq_web_dashboard_with_walls.py --symbols KORU SOXL --paper --wall-refresh-minutes 0
"""

import argparse
import asyncio
import math
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template_string
from ib_insync import IB, Stock, util


# ============================================================
# Timeframes
# ============================================================

TIMEFRAMES = {
    "1d": {
        "label": "Daily",
        "ib_bar_size": "1 day",
        "seed_duration": "3 Y",
        "update_duration": "30 D",
        "min_bars": 320,
    },
    "1h": {
        "label": "1 Hour",
        "ib_bar_size": "1 hour",
        "seed_duration": "90 D",
        "update_duration": "5 D",
        "min_bars": 320,
    },
    "30m": {
        "label": "30 Min",
        "ib_bar_size": "30 mins",
        "seed_duration": "60 D",
        "update_duration": "3 D",
        "min_bars": 320,
    },
    "5m": {
        "label": "5 Min",
        "ib_bar_size": "5 mins",
        "seed_duration": "15 D",
        "update_duration": "2 D",
        "min_bars": 320,
    },
    "1m": {
        "label": "1 Min",
        "ib_bar_size": "1 min",
        "seed_duration": "5 D",
        "update_duration": "1 D",
        "min_bars": 320,
    },
}

TF_DISPLAY_ORDER = ["1m", "5m", "30m", "1h", "1d"]


# ============================================================
# TrendQuality params
# ============================================================

@dataclass
class TQParams:
    fast_length: int = 20
    slow_length: int = 50
    trend_length: int = 4
    noise_type: str = "linear"
    noise_length: int = 250
    correction_factor: float = 2.0
    threshold_value: float = 3.0
    exit_mode: str = "OnZeroCross"   # OnTurnRed, OnLeaveGreen, OnZeroCross
    use_max_loss_stop: bool = False
    max_loss_pct: float = 12.0


# ============================================================
# Global runtime state
# ============================================================

APP_STATE = {
    "last_update": None,
    "last_cycle_seconds": None,
    "connected": False,
    "errors": {},
    "is_updating": False,
    "walls": {},
    "last_wall_update": None,
}

DB_LOCK = threading.Lock()
STATE_LOCK = threading.Lock()


# ============================================================
# SQLite
# ============================================================

def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            bar_time TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            average REAL,
            bar_count INTEGER,
            source TEXT,
            updated_at TEXT,
            PRIMARY KEY (symbol, timeframe, bar_time)
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_bars_symbol_tf_time
        ON bars(symbol, timeframe, bar_time)
    """)
    conn.commit()
    return conn


def count_bars(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int:
    with DB_LOCK:
        cur = conn.execute(
            "SELECT COUNT(*) FROM bars WHERE symbol=? AND timeframe=?",
            (symbol.upper(), timeframe),
        )
        return int(cur.fetchone()[0])


def load_bars(conn: sqlite3.Connection, symbol: str, timeframe: str) -> pd.DataFrame:
    with DB_LOCK:
        df = pd.read_sql_query(
            """
            SELECT bar_time, open, high, low, close, volume
            FROM bars
            WHERE symbol=? AND timeframe=?
            ORDER BY bar_time ASC
            """,
            conn,
            params=(symbol.upper(), timeframe),
        )

    if df.empty:
        return df

    df["bar_time"] = pd.to_datetime(df["bar_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["bar_time"]).sort_values("bar_time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def normalize_bar_time(x) -> str:
    if isinstance(x, pd.Timestamp):
        ts = x
    elif isinstance(x, datetime):
        ts = pd.Timestamp(x)
    elif isinstance(x, date):
        ts = pd.Timestamp(datetime(x.year, x.month, x.day, tzinfo=timezone.utc))
    else:
        ts = pd.to_datetime(x, errors="coerce", utc=True)

    if pd.isna(ts):
        raise ValueError(f"Could not parse IB bar date: {x}")

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts.isoformat()


def upsert_bars(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    source: str = "IBKR",
) -> int:
    if df is None or df.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    rows = []

    for _, r in df.iterrows():
        try:
            bar_time = normalize_bar_time(r["date"])
        except Exception:
            continue

        rows.append((
            symbol.upper(),
            timeframe,
            bar_time,
            float(r.get("open", np.nan)) if not pd.isna(r.get("open", np.nan)) else None,
            float(r.get("high", np.nan)) if not pd.isna(r.get("high", np.nan)) else None,
            float(r.get("low", np.nan)) if not pd.isna(r.get("low", np.nan)) else None,
            float(r.get("close", np.nan)) if not pd.isna(r.get("close", np.nan)) else None,
            float(r.get("volume", 0) or 0),
            float(r.get("average", 0) or 0),
            int(r.get("barCount", 0) or 0),
            source,
            now,
        ))

    if not rows:
        return 0

    with DB_LOCK:
        conn.executemany(
            """
            INSERT INTO bars (
                symbol, timeframe, bar_time,
                open, high, low, close, volume,
                average, bar_count, source, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, timeframe, bar_time)
            DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                average=excluded.average,
                bar_count=excluded.bar_count,
                source=excluded.source,
                updated_at=excluded.updated_at
            """,
            rows,
        )
        conn.commit()

    return len(rows)


# ============================================================
# IBKR bars
# ============================================================

def connect_ib(host: str, port: int, client_id: int) -> IB:
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=20)
    return ib


def fetch_ib_bars(
    ib: IB,
    symbol: str,
    timeframe: str,
    duration: str,
    what_to_show: str,
    use_rth: bool,
) -> pd.DataFrame:
    cfg = TIMEFRAMES[timeframe]
    contract = Stock(symbol.upper(), "SMART", "USD")

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=duration,
        barSizeSetting=cfg["ib_bar_size"],
        whatToShow=what_to_show,
        useRTH=1 if use_rth else 0,
        formatDate=2,
        keepUpToDate=False,
    )

    if not bars:
        return pd.DataFrame()

    return util.df(bars)


def update_symbol_timeframe(
    ib: IB,
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    what_to_show: str,
    use_rth: bool,
    request_pause: float,
) -> Tuple[int, str]:
    existing = count_bars(conn, symbol, timeframe)
    cfg = TIMEFRAMES[timeframe]
    duration = cfg["seed_duration"] if existing < cfg["min_bars"] else cfg["update_duration"]

    try:
        df = fetch_ib_bars(
            ib=ib,
            symbol=symbol,
            timeframe=timeframe,
            duration=duration,
            what_to_show=what_to_show,
            use_rth=use_rth,
        )
        n = upsert_bars(conn, symbol, timeframe, df)
        time.sleep(request_pause)
        return n, ""
    except Exception as e:
        return 0, str(e)


# ============================================================
# Option walls
# ============================================================

def is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def fnum(x, default=0.0):
    return float(x) if is_num(x) else default


def fetch_call_put_walls(
    ib: IB,
    symbol: str,
    sectype: str = "STK",
    exchange: str = "SMART",
    currency: str = "USD",
    expiries: int = 1,
    band: float = 0.15,
    mode: str = "oi",
    batch: int = 40,
    wait: float = 8.0,
) -> Dict:
    """
    Symbol-level option wall snapshot.

    Call wall = strike with largest call OI at/above spot.
    Put wall  = strike with largest put OI at/below spot.

    Open interest comes from IBKR generic tick 101. It is usually a slow-moving
    exchange-reported snapshot, not an intraday live metric.
    """
    from ib_insync import Index, Option

    result = {
        "symbol": symbol,
        "spot": None,
        "call_wall_oi": None,
        "call_wall_oi_value": None,
        "put_wall_oi": None,
        "put_wall_oi_value": None,
        "call_wall_gamma": None,
        "put_wall_gamma": None,
        "seen_gamma": False,
        "expirations": [],
        "strikes_count": 0,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error": "",
    }

    try:
        und = Stock(symbol, exchange, currency) if sectype == "STK" else Index(symbol, exchange, currency)
        ib.qualifyContracts(und)

        if not und.conId:
            raise RuntimeError(f"Could not qualify {symbol}")

        ticker = ib.reqTickers(und)[0]
        spot = ticker.marketPrice()

        if not is_num(spot) or spot <= 0:
            spot = fnum(ticker.close) or fnum(ticker.last)

        if not spot or spot <= 0:
            raise RuntimeError("Could not get spot price")

        result["spot"] = float(spot)

        chains = ib.reqSecDefOptParams(und.symbol, "", und.secType, und.conId)
        if not chains:
            raise RuntimeError("No option chain returned")

        chain = next((c for c in chains if c.exchange == "SMART"), max(chains, key=lambda c: len(c.strikes)))
        expirations = sorted(e for e in chain.expirations if e)[:max(1, expiries)]

        lo = spot * (1 - band)
        hi = spot * (1 + band)
        strikes = sorted(k for k in chain.strikes if lo <= k <= hi)

        if not strikes:
            raise RuntimeError("No strikes inside band")

        result["expirations"] = expirations
        result["strikes_count"] = len(strikes)

        opts = []
        for exp in expirations:
            for k in strikes:
                for right in ("C", "P"):
                    opts.append(
                        Option(
                            symbol,
                            exp,
                            k,
                            right,
                            exchange,
                            multiplier="100",
                            currency=currency,
                            tradingClass=chain.tradingClass,
                        )
                    )

        ib.qualifyContracts(*opts)
        opts = [o for o in opts if o.conId]
        if not opts:
            raise RuntimeError("No option contracts qualified")

        call_oi = defaultdict(float)
        put_oi = defaultdict(float)
        call_gw = defaultdict(float)
        put_gw = defaultdict(float)
        seen_gamma = False

        for i in range(0, len(opts), batch):
            chunk = opts[i:i + batch]
            tickers = [
                ib.reqMktData(
                    o,
                    genericTickList="100,101,104,106",
                    snapshot=False,
                )
                for o in chunk
            ]

            ib.sleep(wait)

            for o, tk in zip(chunk, tickers):
                oi = fnum(tk.callOpenInterest if o.right == "C" else tk.putOpenInterest)
                gamma = tk.modelGreeks.gamma if tk.modelGreeks and is_num(tk.modelGreeks.gamma) else float("nan")
                gamma_weight = oi * gamma * 100 if is_num(gamma) else 0.0

                if is_num(gamma):
                    seen_gamma = True

                if o.right == "C":
                    call_oi[o.strike] += oi
                    call_gw[o.strike] += gamma_weight
                else:
                    put_oi[o.strike] += oi
                    put_gw[o.strike] += gamma_weight

            for o in chunk:
                ib.cancelMktData(o)

        def wall(profile, side):
            sub = {
                k: v
                for k, v in profile.items()
                if (k >= spot if side == "call" else k <= spot) and v > 0
            }
            pool = sub or {k: v for k, v in profile.items() if v > 0}
            return max(pool, key=pool.get) if pool else None

        cw_oi = wall(call_oi, "call")
        pw_oi = wall(put_oi, "put")
        cw_g = wall(call_gw, "call")
        pw_g = wall(put_gw, "put")

        result.update({
            "call_wall_oi": float(cw_oi) if cw_oi else None,
            "call_wall_oi_value": int(call_oi.get(cw_oi, 0)) if cw_oi else None,
            "put_wall_oi": float(pw_oi) if pw_oi else None,
            "put_wall_oi_value": int(put_oi.get(pw_oi, 0)) if pw_oi else None,
            "call_wall_gamma": float(cw_g) if cw_g else None,
            "put_wall_gamma": float(pw_g) if pw_g else None,
            "seen_gamma": bool(seen_gamma),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        return result

    except Exception as e:
        result["error"] = str(e)
        result["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return result


# ============================================================
# TrendQuality calculation
# ============================================================

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def calculate_tq(df: pd.DataFrame, p: TQParams) -> pd.DataFrame:
    out = df.copy()

    if out.empty or len(out) < max(p.slow_length, p.noise_length, p.trend_length) + 5:
        return out

    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)

    # Approximation of ThinkScript TrendPeriods(fast, slow): regime changes when fast EMA crosses slow EMA.
    fast_ema = ema(close, p.fast_length)
    slow_ema = ema(close, p.slow_length)
    spread = fast_ema - slow_ema

    reversal = np.where(spread > 0, 1, np.where(spread < 0, -1, 0))
    reversal = pd.Series(reversal, index=out.index)

    smf = 2.0 / (1.0 + p.trend_length)
    cpc_vals = np.zeros(len(out))
    trend_vals = np.zeros(len(out))

    for i in range(len(out)):
        if i == 0:
            cpc_vals[i] = 0.0
            trend_vals[i] = 0.0
            continue

        if reversal.iloc[i - 1] != reversal.iloc[i]:
            cpc_vals[i] = 0.0
            trend_vals[i] = 0.0
        else:
            close_delta = close.iloc[i] - close.iloc[i - 1]
            cpc_vals[i] = cpc_vals[i - 1] + close_delta
            trend_vals[i] = trend_vals[i - 1] * (1.0 - smf) + cpc_vals[i] * smf

    out["reversal"] = reversal
    out["cpc"] = cpc_vals
    out["trend"] = trend_vals

    diff = (out["cpc"] - out["trend"]).abs()

    if p.noise_type.lower() == "squared":
        noise = p.correction_factor * np.sqrt(
            (diff * diff).rolling(p.noise_length, min_periods=p.noise_length).mean()
        )
    else:
        noise = p.correction_factor * diff.rolling(
            p.noise_length,
            min_periods=p.noise_length,
        ).mean()

    out["noise"] = noise
    out["TQ"] = np.where((noise == 0) | noise.isna(), np.nan, out["trend"] / noise)

    tq = out["TQ"]
    tq_prev = tq.shift(1)

    out["green"] = tq > p.threshold_value
    out["red"] = tq < -p.threshold_value
    out["neutral"] = ~(out["green"] | out["red"])

    out["turned_green"] = out["green"] & (tq_prev <= p.threshold_value)
    out["turned_red"] = out["red"] & (tq_prev >= -p.threshold_value)
    out["left_green"] = (tq < p.threshold_value) & (tq_prev >= p.threshold_value)
    out["cross_down_zero"] = (tq < 0) & (tq_prev >= 0)

    if p.exit_mode == "OnTurnRed":
        exit_trigger = out["turned_red"]
    elif p.exit_mode == "OnLeaveGreen":
        exit_trigger = out["left_green"]
    else:
        exit_trigger = out["cross_down_zero"]

    out["exit_trigger"] = exit_trigger

    entry_vals = np.zeros(len(out))
    entering = np.zeros(len(out), dtype=bool)
    exiting = np.zeros(len(out), dtype=bool)
    stop_exit = np.zeros(len(out), dtype=bool)

    for i in range(len(out)):
        if i == 0:
            entry_vals[i] = 0.0
            continue

        prev_entry = entry_vals[i - 1]

        if prev_entry == 0:
            if bool(out["turned_green"].iloc[i]):
                entry_vals[i] = close.iloc[i]
                entering[i] = True
            else:
                entry_vals[i] = 0.0
        else:
            hard_stop = False
            if p.use_max_loss_stop:
                hard_stop = low.iloc[i] <= prev_entry * (1.0 - p.max_loss_pct / 100.0)

            if bool(exit_trigger.iloc[i]) or hard_stop:
                entry_vals[i] = 0.0
                exiting[i] = True
                stop_exit[i] = hard_stop
            else:
                entry_vals[i] = prev_entry

    out["entry"] = entry_vals
    out["in_trade"] = out["entry"] > 0
    out["entering"] = entering
    out["exiting"] = exiting
    out["stop_exit"] = stop_exit

    entry_ref = []
    last_entry = np.nan

    for i in range(len(out)):
        if entering[i]:
            last_entry = close.iloc[i]

        if out["in_trade"].iloc[i]:
            entry_ref.append(last_entry)
        elif exiting[i]:
            entry_ref.append(last_entry)
        else:
            entry_ref.append(np.nan)

    out["entry_ref"] = entry_ref
    out["gain_pct"] = np.where(
        pd.notna(out["entry_ref"]) & (out["entry_ref"] > 0),
        (out["close"] / out["entry_ref"] - 1.0) * 100.0,
        np.nan,
    )

    return out


def latest_signal_summary(df: pd.DataFrame, p: TQParams) -> Dict:
    result = {
        "state": "NO DATA",
        "signal": "NO DATA",
        "since_bars": None,
        "tq": None,
        "zone": "n/a",
        "close": None,
        "entry": None,
        "gain_pct": None,
        "bar_time": None,
        "bars": len(df),
        "is_fresh_buy": False,
        "is_fresh_sell": False,
    }

    calc = calculate_tq(df, p)

    if calc.empty or "TQ" not in calc.columns:
        return result

    valid = calc.dropna(subset=["TQ"])
    if valid.empty:
        result["state"] = "WARMING"
        result["signal"] = "WARMING"
        return result

    last = valid.iloc[-1]
    last_idx = valid.index[-1]

    if bool(last.get("green", False)):
        zone = "green"
    elif bool(last.get("red", False)):
        zone = "red"
    else:
        zone = "neutral"

    in_trade = bool(last.get("in_trade", False))
    entering_now = bool(last.get("entering", False))
    exiting_now = bool(last.get("exiting", False))

    entering_idxs = valid.index[valid.get("entering", False) == True].tolist()
    exiting_idxs = valid.index[valid.get("exiting", False) == True].tolist()

    if in_trade:
        state = "LONG"
        signal = "BUY"

        if entering_idxs:
            last_entry_idx = entering_idxs[-1]
            since_bars = int(last_idx - last_entry_idx)
        else:
            since_bars = None

        entry = float(last.get("entry_ref", np.nan)) if pd.notna(last.get("entry_ref", np.nan)) else None
        gain_pct = float(last.get("gain_pct", np.nan)) if pd.notna(last.get("gain_pct", np.nan)) else None

    else:
        state = "FLAT"
        signal = "SELL"

        if exiting_idxs:
            last_exit_idx = exiting_idxs[-1]
            since_bars = int(last_idx - last_exit_idx)
        else:
            since_bars = None

        entry = None
        gain_pct = None

    result.update({
        "state": state,
        "signal": signal,
        "since_bars": since_bars,
        "tq": float(last["TQ"]) if pd.notna(last["TQ"]) else None,
        "zone": zone,
        "close": float(last["close"]) if pd.notna(last["close"]) else None,
        "entry": entry,
        "gain_pct": gain_pct,
        "bar_time": str(last["bar_time"]),
        "bars": len(df),
        "is_fresh_buy": entering_now,
        "is_fresh_sell": exiting_now,
    })

    return result


# ============================================================
# HTML dashboard - transposed compact UI
# ============================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IBKR TQ Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">

    <style>
        :root {
            --bg: #020617;
            --panel: rgba(15, 23, 42, 0.88);
            --border: rgba(148, 163, 184, 0.18);
            --text: #e5e7eb;
            --muted: #94a3b8;
            --green: #22c55e;
            --red: #ef4444;
            --yellow: #facc15;
            --gray: #64748b;
            --blue: #38bdf8;
        }

        * { box-sizing: border-box; }

        body {
            margin: 0;
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(34, 197, 94, 0.10), transparent 25%),
                linear-gradient(135deg, #020617, #0f172a 50%, #020617);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            min-height: 100vh;
            overflow: auto;
        }

        .wrap {
            min-height: 100vh;
            padding: 8px 10px;
            display: flex;
            flex-direction: column;
            gap: 7px;
        }

        .topbar {
            display: grid;
            grid-template-columns: 1.05fr 2.25fr;
            gap: 7px;
            min-height: 48px;
        }

        .titlebox, .stat, .alertbar, .quick-wrap, .matrix-wrap {
            border: 1px solid var(--border);
            background: rgba(15, 23, 42, 0.72);
        }

        .titlebox {
            border-radius: 13px;
            padding: 7px 10px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .titlebox h1 {
            margin: 0;
            font-size: 17px;
            line-height: 1.05;
            font-weight: 950;
            letter-spacing: -0.04em;
        }

        .subtitle {
            color: var(--muted);
            font-size: 10px;
            margin-top: 3px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .statusgrid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 7px;
        }

        .stat {
            border-radius: 13px;
            padding: 6px 8px;
            min-width: 0;
        }

        .stat .k {
            color: var(--muted);
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            white-space: nowrap;
        }

        .stat .v {
            margin-top: 2px;
            font-size: 14px;
            font-weight: 950;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .connected { color: var(--green); }
        .disconnected { color: var(--red); }

        .alertbar {
            border-color: rgba(34, 197, 94, 0.32);
            border-radius: 13px;
            background:
                radial-gradient(circle at top left, rgba(34, 197, 94, 0.14), transparent 30%),
                rgba(15, 23, 42, 0.78);
            padding: 6px 8px;
            min-height: 60px;
            max-height: 60px;
            display: grid;
            grid-template-columns: 135px 1fr;
            gap: 7px;
            overflow: hidden;
        }

        .alert-title {
            display: flex;
            flex-direction: column;
            justify-content: center;
            border-right: 1px solid rgba(148, 163, 184, 0.16);
            padding-right: 7px;
        }

        .alert-title-main { font-size: 12px; font-weight: 950; }
        .alert-title-sub { color: var(--muted); font-size: 9px; margin-top: 2px; }

        .alert-count {
            margin-top: 3px;
            display: inline-block;
            width: fit-content;
            padding: 2px 6px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.36);
            color: #bbf7d0;
            font-size: 9px;
            font-weight: 950;
        }

        .alert-grid {
            display: flex;
            gap: 6px;
            overflow-x: auto;
            overflow-y: hidden;
            padding-bottom: 2px;
        }

        .alert-card {
            min-width: 164px;
            max-width: 164px;
            border: 1px solid rgba(34, 197, 94, 0.38);
            background: rgba(2, 6, 23, 0.34);
            border-radius: 10px;
            padding: 5px 6px;
        }

        .alert-card-top {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 6px;
            margin-bottom: 3px;
        }

        .alert-symbol {
            font-size: 15px;
            font-weight: 950;
            letter-spacing: -0.04em;
            line-height: 1;
        }

        .alert-tf { color: var(--muted); font-size: 9px; margin-top: 1px; }

        .alert-badge {
            color: #bbf7d0;
            background: rgba(34, 197, 94, 0.16);
            border: 1px solid rgba(34, 197, 94, 0.34);
            font-size: 8.5px;
            font-weight: 950;
            padding: 2px 5px;
            border-radius: 999px;
            white-space: nowrap;
        }

        .alert-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 2px 5px;
        }

        .mini-label {
            color: var(--muted);
            font-size: 8px;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }

        .mini-value {
            color: var(--text);
            font-size: 9.5px;
            font-weight: 850;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .no-alerts {
            width: 100%;
            border: 1px dashed rgba(148, 163, 184, 0.24);
            border-radius: 10px;
            color: var(--muted);
            font-size: 11px;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 45px;
        }

        .toolbar {
            display: grid;
            grid-template-columns: 220px 122px 1fr;
            gap: 8px;
            align-items: center;
            min-height: 28px;
        }

        .search {
            height: 28px;
            width: 100%;
            background: rgba(15, 23, 42, 0.80);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 5px 9px;
            border-radius: 10px;
            outline: none;
            font-size: 11.5px;
        }

        .alert-toggle {
            height: 28px;
            border: 1px solid rgba(56, 189, 248, 0.35);
            background: rgba(56, 189, 248, 0.10);
            color: #bae6fd;
            border-radius: 10px;
            font-size: 10.5px;
            font-weight: 900;
            cursor: pointer;
            white-space: nowrap;
        }

        .alert-toggle.on {
            border-color: rgba(34, 197, 94, 0.45);
            background: rgba(34, 197, 94, 0.14);
            color: #bbf7d0;
        }

        .alert-toggle.blocked {
            border-color: rgba(239, 68, 68, 0.40);
            background: rgba(239, 68, 68, 0.12);
            color: #fecaca;
        }

        .legend {
            color: var(--muted);
            font-size: 10px;
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            align-items: center;
        }

        .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; margin-right: 4px; }
        .dot.green { background: var(--green); }
        .dot.red { background: var(--red); }
        .dot.yellow { background: var(--yellow); }

        /* =====================================================
           DIV 1: compact quick-read matrix
           ===================================================== */

        .quick-wrap {
            border-radius: 15px;
            padding: 8px;
            background: rgba(15, 23, 42, 0.68);
        }

        .section-title-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }

        .section-title {
            font-size: 13px;
            font-weight: 950;
            letter-spacing: -0.03em;
        }

        .section-subtitle {
            color: var(--muted);
            font-size: 9.5px;
        }

        .quick-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 5px;
            table-layout: fixed;
        }

        .quick-table th {
            color: var(--muted);
            text-align: center;
            font-size: 9px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0 4px 1px;
        }

        .quick-table th:first-child { width: 60px; text-align: center; }

        .quick-symbol {
            font-size: 11px;
            font-weight: 950;
            letter-spacing: -0.03em;
            white-space: nowrap;
            text-align: center;
        }

        .quick-tf {
            font-size: 14px;
            font-weight: 950;
            letter-spacing: -0.04em;
            white-space: nowrap;
            text-align: center;
            color: #f8fafc;
        }

        .quick-cell { padding: 0 4px; }

        .quick-pill {
            min-height: 34px;
            border-radius: 11px;
            border: 1px solid rgba(148, 163, 184, 0.14);
            display: grid;
            place-items: center;
            padding: 3px 4px;
            text-align: center;
            background: rgba(2, 6, 23, 0.28);
        }

        .quick-pill .qtop {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5px;
            line-height: 1;
        }

        .quick-pill .qs {
            font-size: 12px;
            line-height: 1;
            font-weight: 950;
            letter-spacing: -0.02em;
        }

        .quick-zone {
            min-width: 17px;
            height: 17px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            font-size: 8.5px;
            font-weight: 950;
            border: 1px solid rgba(148, 163, 184, 0.24);
            color: #cbd5e1;
            background: rgba(148, 163, 184, 0.10);
        }

        .quick-zone.green {
            color: #bbf7d0;
            border-color: rgba(34, 197, 94, 0.36);
            background: rgba(34, 197, 94, 0.14);
        }

        .quick-zone.red {
            color: #fecaca;
            border-color: rgba(239, 68, 68, 0.34);
            background: rgba(239, 68, 68, 0.12);
        }

        .quick-zone.neutral {
            color: #fef08a;
            border-color: rgba(250, 204, 21, 0.34);
            background: rgba(250, 204, 21, 0.10);
        }

        .quick-pill .qm {
            margin-top: 2px;
            color: rgba(226, 232, 240, 0.78);
            font-size: 8.5px;
            line-height: 1;
            font-weight: 800;
            white-space: nowrap;
        }

        .quick-pill.buy {
            color: #dcfce7;
            border-color: rgba(34, 197, 94, 0.45);
            background: linear-gradient(180deg, rgba(34, 197, 94, 0.30), rgba(34, 197, 94, 0.11));
            box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.12);
        }

        .quick-pill.sell {
            color: #fee2e2;
            border-color: rgba(239, 68, 68, 0.45);
            background: linear-gradient(180deg, rgba(239, 68, 68, 0.28), rgba(239, 68, 68, 0.10));
            box-shadow: inset 0 0 0 1px rgba(239, 68, 68, 0.10);
        }

        .quick-pill.neutral {
            color: #e2e8f0;
            border-color: rgba(148, 163, 184, 0.30);
            background: linear-gradient(180deg, rgba(100, 116, 139, 0.28), rgba(100, 116, 139, 0.10));
        }

        .quick-pill.warming, .quick-pill.nodata {
            color: #fef08a;
            border-color: rgba(250, 204, 21, 0.32);
            background: rgba(250, 204, 21, 0.08);
        }

        .quick-pill.fresh-ring {
            box-shadow: 0 0 0 1px rgba(34,197,94,0.70), 0 0 18px rgba(34,197,94,0.16);
        }

        /* =====================================================
           DIV 2: current detailed transposed matrix
           ===================================================== */

        .matrix-wrap {
            border-radius: 15px;
            background: rgba(15, 23, 42, 0.62);
            overflow: hidden;
        }

        .detail-table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed;
        }

        .detail-table thead th {
            height: 52px;
            background: rgba(2, 6, 23, 0.96);
            color: #cbd5e1;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            padding: 4px 5px;
            text-align: center;
            border-bottom: 1px solid var(--border);
            border-right: 1px solid rgba(148, 163, 184, 0.10);
        }

        .detail-table thead th:first-child { width: 74px; }

        .detail-table tbody td {
            border-bottom: 1px solid rgba(148, 163, 184, 0.10);
            border-right: 1px solid rgba(148, 163, 184, 0.08);
            padding: 4px;
            vertical-align: middle;
        }

        .detail-table tbody tr:last-child td { border-bottom: none; }

        .tf-cell {
            font-size: 16px;
            font-weight: 950;
            letter-spacing: -0.04em;
            background: rgba(15, 23, 42, 0.92);
            color: #f8fafc;
            text-align: center;
        }

        .symbol-head {
            font-size: 12px;
            font-weight: 950;
            letter-spacing: -0.03em;
            color: #f8fafc;
            display: grid;
            gap: 1px;
            line-height: 1.05;
        }

        .symbol-name { font-size: 13px; font-weight: 950; }
        .wall-line {
            font-size: 9px;
            color: var(--muted);
            font-weight: 700;
            letter-spacing: 0;
            white-space: nowrap;
            text-transform: none;
        }
        .call-wall { color: #bbf7d0; }
        .put-wall { color: #fecaca; }
        .wall-line span { color: var(--muted); font-weight: 600; }
        .wall-error .symbol-name::after { content: " ⚠"; color: var(--red); }

        .cell-card {
            height: 100%;
            min-height: 70px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: rgba(2, 6, 23, 0.24);
            padding: 4px 5px;
            display: grid;
            grid-template-rows: auto 1fr;
            gap: 2px;
        }

        .cell-card.buy { border-color: rgba(34, 197, 94, 0.44); background: rgba(34, 197, 94, 0.055); }
        .cell-card.sell { border-color: rgba(239, 68, 68, 0.30); background: rgba(239, 68, 68, 0.035); }

        .cell-head { display: flex; justify-content: space-between; align-items: center; gap: 4px; }

        .signal { font-size: 12px; line-height: 1; font-weight: 950; letter-spacing: -0.03em; }
        .signal.buy { color: var(--green); }
        .signal.sell { color: var(--red); }
        .signal.warming { color: var(--yellow); }
        .signal.nodata { color: var(--gray); }

        .zone {
            font-size: 8.5px;
            line-height: 1;
            padding: 2px 5px;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #cbd5e1;
            max-width: 50px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .zone.green { color: #bbf7d0; background: rgba(34, 197, 94, 0.11); border-color: rgba(34, 197, 94, 0.30); }
        .zone.red { color: #fecaca; background: rgba(239, 68, 68, 0.10); border-color: rgba(239, 68, 68, 0.28); }
        .zone.neutral { color: #fef08a; background: rgba(250, 204, 21, 0.09); border-color: rgba(250, 204, 21, 0.28); }

        .cell-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 2px 4px;
            align-content: center;
        }

        .kv { min-width: 0; }
        .kv .k { color: var(--muted); font-size: 7.6px; line-height: 1.05; text-transform: uppercase; letter-spacing: 0.06em; }
        .kv .v { color: var(--text); font-size: 9.5px; line-height: 1.15; font-weight: 850; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .gain.pos { color: var(--green); }
        .gain.neg { color: var(--red); }
        .fresh-ring { box-shadow: 0 0 0 1px rgba(34,197,94,0.65), 0 0 20px rgba(34,197,94,0.15); }
        .error-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--red); display: inline-block; margin-left: 4px; }

        @media (max-width: 1300px) {
            .topbar { grid-template-columns: 1fr; }
            .statusgrid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .quick-wrap, .matrix-wrap { overflow-x: auto; }
            .quick-table, .detail-table { min-width: 1120px; }
        }
    </style>
</head>

<body>
    <div class="wrap">

        <div class="topbar">
            <div class="titlebox">
                <h1>IBKR TrendQuality</h1>
                <div class="subtitle">Fast read on top, detailed matrix below</div>
            </div>

            <div class="statusgrid">
                <div class="stat"><div class="k">IBKR</div><div class="v" id="connected">Checking</div></div>
                <div class="stat"><div class="k">Symbols</div><div class="v" id="metricSymbols">-</div></div>
                <div class="stat"><div class="k">BUY Cells</div><div class="v" id="metricBuys">-</div></div>
                <div class="stat"><div class="k">Fresh BUYs</div><div class="v" id="metricFreshBuys">-</div></div>
                <div class="stat"><div class="k">Bars Updated</div><div class="v" id="lastUpdate">-</div></div>
                <div class="stat"><div class="k">Walls Updated</div><div class="v" id="lastWallUpdate">-</div></div>
            </div>
        </div>

        <div class="alertbar">
            <div class="alert-title">
                <div class="alert-title-main">Fresh BUY Alerts</div>
                <div class="alert-title-sub">Newest entry bars only</div>
                <div class="alert-count" id="newBuyCount">0 new</div>
            </div>
            <div class="alert-grid" id="newBuyGrid">
                <div class="no-alerts">No fresh BUY signals right now.</div>
            </div>
        </div>

        <div class="toolbar">
            <input class="search" id="searchBox" placeholder="Filter symbols...">
            <button class="alert-toggle" id="enableAlertsBtn" type="button">Enable Alerts</button>
            <div class="legend">
                <span><span class="dot green"></span>BUY</span>
                <span><span class="dot yellow"></span>Neutral</span>
                <span><span class="dot red"></span>SELL</span>
                <span>Order: 1m → 5m → 30m → 1h → 1d</span>
            </div>
        </div>

        <!-- DIV 1: compact quick read -->
        <div class="quick-wrap">
            <div class="section-title-row">
                <div>
                    <div class="section-title">Compact Multi-Timeframe Power</div>
                    <div class="section-subtitle">Transposed quick read: rows = timeframes, columns = symbols. Small text shows TQ / since-bars.</div>
                </div>
            </div>
            <table class="quick-table">
                <thead>
                    <tr id="quickHeaderRow"><th>TF</th></tr>
                </thead>
                <tbody id="quickBody"></tbody>
            </table>
        </div>

        <!-- DIV 2: current detailed view -->
        <div class="matrix-wrap">
            <div class="section-title-row" style="padding: 8px 8px 0 8px; margin-bottom: 4px;">
                <div>
                    <div class="section-title">Detailed Signal Matrix</div>
                    <div class="section-subtitle">Strategy state, entry, gain, TQ, and call/put wall headers.</div>
                </div>
            </div>
            <table class="detail-table">
                <thead>
                    <tr id="headerRow"><th>TF</th></tr>
                </thead>
                <tbody id="dashboardBody"></tbody>
            </table>
        </div>

    </div>

<script>
const tfOrder = ["1m", "5m", "30m", "1h", "1d"];
let latestPayload = null;
let alertsEnabled = false;
let audioCtx = null;
let originalTitle = document.title;
const seenFreshBuyKeys = new Set();

// 0 = only newest bar. 1 or 2 keeps alerts visible slightly longer.
const RECENT_BUY_MAX_BARS = 0;

function fmtNum(x, digits=2, prefix="") {
    if (x === null || x === undefined || Number.isNaN(x)) return "-";
    return prefix + Number(x).toFixed(digits);
}
function fmtPct(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return "-";
    return Number(x).toFixed(1) + "%";
}
function fmtSince(x) { return (x === null || x === undefined) ? "-" : `${x}`; }
function fmtShortTime(x) {
    if (!x) return "-";
    try { return new Date(x).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }
    catch { return x; }
}
function fmtLastUpdate(x) { return x ? (String(x).split(" ")[1] || x) : "-"; }
function signalClass(signal) {
    if (signal === "BUY") return "buy";
    if (signal === "SELL") return "sell";
    if (signal === "WARMING") return "warming";
    return "nodata";
}
function zoneClass(zone) {
    if (zone === "green") return "green";
    if (zone === "red") return "red";
    if (zone === "neutral") return "neutral";
    return "";
}
function shortZone(zone) {
    if (zone === "green") return "G";
    if (zone === "red") return "R";
    if (zone === "neutral") return "N";
    return "-";
}
function tfLabel(tf, timeframes) { return timeframes[tf]?.label || tf; }
function fmtWall(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return "-";
    return Number(x).toFixed(2);
}
function fmtWallDist(wall, spot) {
    if (wall === null || wall === undefined || spot === null || spot === undefined || Number.isNaN(wall) || Number.isNaN(spot) || Number(spot) === 0) return "-";
    const pct = ((Number(wall) / Number(spot)) - 1) * 100;
    return pct.toFixed(1) + "%";
}
function getVisibleSymbols(payload) {
    const filter = document.getElementById("searchBox").value.trim().toUpperCase();
    if (!filter) return payload.symbols;
    return payload.symbols.filter(sym => sym.includes(filter));
}
function buildQuickHeader(payload) {
    const quickHeaderRow = document.getElementById("quickHeaderRow");
    const symbols = getVisibleSymbols(payload);
    quickHeaderRow.innerHTML = "<th>TF</th>";
    for (const sym of symbols) quickHeaderRow.innerHTML += `<th><div class="quick-symbol">${sym}</div></th>`;
}
function quickStatus(cell) {
    if (!cell || (!cell.signal && !cell.zone)) {
        return {text: "NO DATA", cls: "nodata", zone: "-", zoneCls: ""};
    }

    if (cell.signal === "WARMING") {
        return {
            text: "WARM",
            cls: "warming",
            zone: shortZone(cell.zone),
            zoneCls: zoneClass(cell.zone)
        };
    }

    // Match the detailed matrix: primary state is BUY/SELL, while the small
    // badge shows the current TQ zone: G, N, or R. This prevents a flat/SELL
    // trade in neutral chop from being mislabeled as simply NEUTRAL.
    if (cell.signal === "BUY") {
        return {
            text: "BUY",
            cls: "buy",
            zone: shortZone(cell.zone),
            zoneCls: zoneClass(cell.zone)
        };
    }

    if (cell.signal === "SELL") {
        return {
            text: "SELL",
            cls: "sell",
            zone: shortZone(cell.zone),
            zoneCls: zoneClass(cell.zone)
        };
    }

    if (cell.zone === "green") {
        return {text: "BUY", cls: "buy", zone: "G", zoneCls: "green"};
    }
    if (cell.zone === "red") {
        return {text: "SELL", cls: "sell", zone: "R", zoneCls: "red"};
    }
    if (cell.zone === "neutral") {
        return {text: "NEUTRAL", cls: "neutral", zone: "N", zoneCls: "neutral"};
    }

    return {text: cell.signal || "NO DATA", cls: signalClass(cell.signal), zone: "-", zoneCls: ""};
}
function buildQuickView(payload) {
    buildQuickHeader(payload);
    const symbols = getVisibleSymbols(payload);
    const body = document.getElementById("quickBody");
    body.innerHTML = "";

    for (const tf of tfOrder) {
        let row = `<tr><td class="quick-tf">${tf}</td>`;
        for (const sym of symbols) {
            const cell = payload.data[sym]?.[tf] || {};
            const status = quickStatus(cell);
            const isFresh = cell.signal === "BUY" && cell.since_bars !== null && cell.since_bars <= RECENT_BUY_MAX_BARS;
            const freshClass = isFresh ? " fresh-ring" : "";
            const meta = cell.tq !== null && cell.tq !== undefined ? `TQ ${fmtNum(cell.tq, 1)} · ${fmtSince(cell.since_bars)}` : "-";
            row += `<td class="quick-cell"><div class="quick-pill ${status.cls}${freshClass}"><div class="qtop"><div class="qs">${status.text}</div><div class="quick-zone ${status.zoneCls}">${status.zone}</div></div><div class="qm">${meta}</div></div></td>`;
        }
        row += "</tr>";
        body.innerHTML += row;
    }
}
function buildHeader(payload) {
    const headerRow = document.getElementById("headerRow");
    const symbols = getVisibleSymbols(payload);
    headerRow.innerHTML = "<th>TF</th>";

    for (const sym of symbols) {
        const wall = payload.walls?.[sym] || {};
        const wallError = wall.error ? " wall-error" : "";
        headerRow.innerHTML += `
            <th title="${wall.error ? wall.error : ''}">
                <div class="symbol-head${wallError}">
                    <div class="symbol-name">${sym}</div>
                    <div class="wall-line">Spot ${fmtWall(wall.spot)}</div>
                    <div class="wall-line call-wall">C ${fmtWall(wall.call_wall_oi)} <span>${fmtWallDist(wall.call_wall_oi, wall.spot)}</span></div>
                    <div class="wall-line put-wall">P ${fmtWall(wall.put_wall_oi)} <span>${fmtWallDist(wall.put_wall_oi, wall.spot)}</span></div>
                </div>
            </th>`;
    }
}
function getFreshBuys(payload) {
    const freshBuys = [];
    for (const sym of payload.symbols) {
        for (const tf of tfOrder) {
            const cell = payload.data[sym]?.[tf];
            if (!cell) continue;
            const isRecentBuy = cell.signal === "BUY" && cell.since_bars !== null && cell.since_bars <= RECENT_BUY_MAX_BARS;
            if (isRecentBuy) {
                freshBuys.push({
                    symbol: sym,
                    timeframe: tf,
                    label: tfLabel(tf, payload.timeframes),
                    close: cell.close,
                    tq: cell.tq,
                    zone: cell.zone,
                    bar_time: cell.bar_time,
                    entry: cell.entry,
                    gain_pct: cell.gain_pct,
                    since_bars: cell.since_bars
                });
            }
        }
    }
    return freshBuys;
}
function freshBuyKey(buy) {
    return `${buy.symbol}|${buy.timeframe}|${buy.bar_time}|${fmtNum(buy.entry, 2)}`;
}

function ensureAudio() {
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === "suspended") audioCtx.resume();
    } catch (err) {
        console.warn("Audio unavailable", err);
    }
}

function playAlertSound() {
    try {
        ensureAudio();
        if (!audioCtx) return;

        const now = audioCtx.currentTime;
        const gain = audioCtx.createGain();
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.55);
        gain.connect(audioCtx.destination);

        const osc1 = audioCtx.createOscillator();
        osc1.type = "sine";
        osc1.frequency.setValueAtTime(880, now);
        osc1.connect(gain);
        osc1.start(now);
        osc1.stop(now + 0.18);

        const osc2 = audioCtx.createOscillator();
        osc2.type = "sine";
        osc2.frequency.setValueAtTime(1175, now + 0.20);
        osc2.connect(gain);
        osc2.start(now + 0.20);
        osc2.stop(now + 0.55);
    } catch (err) {
        console.warn("Alert sound failed", err);
    }
}

function flashPageTitle(message) {
    let flips = 0;
    const maxFlips = 20;
    const timer = setInterval(() => {
        document.title = (flips % 2 === 0) ? message : originalTitle;
        flips += 1;
        if (flips >= maxFlips) {
            clearInterval(timer);
            document.title = originalTitle;
        }
    }, 800);
}

function notifyFreshBuy(buy) {
    const title = `TQ BUY: ${buy.symbol} ${buy.timeframe}`;
    const body = `Entry ${fmtNum(buy.entry, 2, "$ ").replace("$ ", "$")} | Close ${fmtNum(buy.close, 2, "$ ").replace("$ ", "$")} | TQ ${fmtNum(buy.tq, 2)} | ${fmtShortTime(buy.bar_time)}`;

    playAlertSound();
    flashPageTitle(`🚨 ${buy.symbol} ${buy.timeframe} BUY`);

    if ("Notification" in window && Notification.permission === "granted") {
        const n = new Notification(title, {
            body: body,
            tag: freshBuyKey(buy),
            requireInteraction: false,
            silent: true
        });
        n.onclick = () => {
            window.focus();
            n.close();
        };
    }
}

function processFreshBuyNotifications(payload) {
    const freshBuys = getFreshBuys(payload);

    for (const buy of freshBuys) {
        const key = freshBuyKey(buy);
        if (seenFreshBuyKeys.has(key)) continue;

        seenFreshBuyKeys.add(key);
        if (alertsEnabled) notifyFreshBuy(buy);
    }
}

async function enableAlerts() {
    const btn = document.getElementById("enableAlertsBtn");
    ensureAudio();

    if (!("Notification" in window)) {
        alertsEnabled = true;
        btn.textContent = "Sound On";
        btn.className = "alert-toggle on";
        playAlertSound();
        return;
    }

    let permission = Notification.permission;
    if (permission === "default") {
        permission = await Notification.requestPermission();
    }

    if (permission === "granted") {
        alertsEnabled = true;
        btn.textContent = "Alerts On";
        btn.className = "alert-toggle on";
        playAlertSound();
    } else {
        alertsEnabled = true;
        btn.textContent = "Sound Only";
        btn.className = "alert-toggle blocked";
        playAlertSound();
    }
}

function buildNewBuyStrip(payload) {
    const grid = document.getElementById("newBuyGrid");
    const countEl = document.getElementById("newBuyCount");
    const freshBuys = getFreshBuys(payload);
    countEl.textContent = `${freshBuys.length} new`;

    if (freshBuys.length === 0) {
        grid.innerHTML = `<div class="no-alerts">No fresh BUY signals right now.</div>`;
        return;
    }
    grid.innerHTML = "";
    for (const buy of freshBuys) {
        grid.innerHTML += `
            <div class="alert-card">
                <div class="alert-card-top">
                    <div><div class="alert-symbol">${buy.symbol}</div><div class="alert-tf">${buy.label}</div></div>
                    <div class="alert-badge">BUY</div>
                </div>
                <div class="alert-stats">
                    <div><div class="mini-label">Entry</div><div class="mini-value">${fmtNum(buy.entry, 2, "$")}</div></div>
                    <div><div class="mini-label">Close</div><div class="mini-value">${fmtNum(buy.close, 2, "$")}</div></div>
                    <div><div class="mini-label">TQ</div><div class="mini-value">${fmtNum(buy.tq, 2)}</div></div>
                    <div><div class="mini-label">Time</div><div class="mini-value">${fmtShortTime(buy.bar_time)}</div></div>
                </div>
            </div>`;
    }
}
function buildCell(cell) {
    const sig = cell.signal || "NO DATA";
    const sigClass = signalClass(sig);
    const zone = cell.zone || "n/a";
    const zClass = zoneClass(zone);
    let gainClass = "";
    if (cell.gain_pct !== null && cell.gain_pct !== undefined) gainClass = cell.gain_pct >= 0 ? "pos" : "neg";

    const isFresh = cell.signal === "BUY" && cell.since_bars !== null && cell.since_bars <= RECENT_BUY_MAX_BARS;
    const freshClass = isFresh ? "fresh-ring" : "";
    const errDot = cell.error ? `<span class="error-dot" title="${cell.error}"></span>` : "";

    return `
        <div class="cell-card ${sigClass} ${freshClass}">
            <div class="cell-head"><div class="signal ${sigClass}">${sig}${errDot}</div><div class="zone ${zClass}">${shortZone(zone)}</div></div>
            <div class="cell-grid">
                <div class="kv"><div class="k">Since</div><div class="v">${fmtSince(cell.since_bars)}</div></div>
                <div class="kv"><div class="k">TQ</div><div class="v">${fmtNum(cell.tq, 1)}</div></div>
                <div class="kv"><div class="k">Close</div><div class="v">${fmtNum(cell.close, 2)}</div></div>
                <div class="kv"><div class="k">Entry</div><div class="v">${fmtNum(cell.entry, 2)}</div></div>
                <div class="kv"><div class="k">Gain</div><div class="v gain ${gainClass}">${fmtPct(cell.gain_pct)}</div></div>
                <div class="kv"><div class="k">Time</div><div class="v">${fmtShortTime(cell.bar_time)}</div></div>
            </div>
        </div>`;
}
function buildDetailedView(payload) {
    buildHeader(payload);
    const symbols = getVisibleSymbols(payload);
    const body = document.getElementById("dashboardBody");
    body.innerHTML = "";

    for (const tf of tfOrder) {
        let row = `<tr><td class="tf-cell">${tf}</td>`;
        for (const sym of symbols) {
            const cell = payload.data[sym]?.[tf] || {};
            row += `<td>${buildCell(cell)}</td>`;
        }
        row += "</tr>";
        body.innerHTML += row;
    }
}
function render(payload) {
    latestPayload = payload;
    buildNewBuyStrip(payload);
    processFreshBuyNotifications(payload);
    buildQuickView(payload);
    buildDetailedView(payload);

    const connectedEl = document.getElementById("connected");
    connectedEl.textContent = payload.connected ? "Connected" : "Disconnected";
    connectedEl.className = payload.connected ? "v connected" : "v disconnected";

    document.getElementById("metricSymbols").textContent = payload.symbols.length;
    document.getElementById("lastUpdate").textContent = fmtLastUpdate(payload.last_update);
    document.getElementById("lastWallUpdate").textContent = fmtLastUpdate(payload.last_wall_update);

    let buyCount = 0;
    for (const sym of payload.symbols) {
        for (const tf of tfOrder) {
            const cell = payload.data[sym]?.[tf];
            if (!cell) continue;
            if (cell.signal === "BUY") buyCount++;
        }
    }
    const freshBuys = getFreshBuys(payload);
    document.getElementById("metricBuys").textContent = buyCount;
    document.getElementById("metricFreshBuys").textContent = freshBuys.length;
}
async function fetchStatus() {
    try {
        const res = await fetch("/api/status");
        const payload = await res.json();
        render(payload);
    } catch (err) { console.error(err); }
}
document.getElementById("searchBox").addEventListener("input", () => { if (latestPayload) render(latestPayload); });
document.getElementById("enableAlertsBtn").addEventListener("click", enableAlerts);
fetchStatus();
setInterval(fetchStatus, 5000);
</script>
</body>
</html>
"""


# ============================================================
# API payload
# ============================================================

def build_status_payload(conn: sqlite3.Connection, symbols: List[str], params: TQParams) -> Dict:
    data = {}

    for symbol in symbols:
        data[symbol] = {}
        for tf in TIMEFRAMES.keys():
            df = load_bars(conn, symbol, tf)
            summary = latest_signal_summary(df, params)

            with STATE_LOCK:
                err = APP_STATE["errors"].get(f"{symbol}:{tf}", "")
            summary["error"] = err
            data[symbol][tf] = summary

    with STATE_LOCK:
        walls = dict(APP_STATE["walls"])
        errors = dict(APP_STATE["errors"])
        payload_state = {
            "connected": APP_STATE["connected"],
            "is_updating": APP_STATE["is_updating"],
            "last_update": APP_STATE["last_update"],
            "last_cycle_seconds": APP_STATE["last_cycle_seconds"],
            "last_wall_update": APP_STATE["last_wall_update"],
        }

    return {
        "symbols": symbols,
        "timeframes": {
            tf: {
                "label": cfg["label"],
                "ib_bar_size": cfg["ib_bar_size"],
            }
            for tf, cfg in TIMEFRAMES.items()
        },
        "data": data,
        "walls": walls,
        "errors": errors,
        **payload_state,
    }


# ============================================================
# Background updater
# ============================================================

def updater_loop(
    conn: sqlite3.Connection,
    symbols: List[str],
    host: str,
    port: int,
    client_id: int,
    what_to_show: str,
    use_rth: bool,
    refresh_seconds: int,
    request_pause: float,
    delayed_market_data: bool,
    wall_refresh_minutes: int,
    wall_expiries: int,
    wall_band: float,
    wall_mode: str,
    wall_batch: int,
    wall_wait: float,
):
    asyncio.set_event_loop(asyncio.new_event_loop())
    ib = IB()
    last_wall_fetch_ts = 0.0

    while True:
        cycle_start = time.time()
        with STATE_LOCK:
            APP_STATE["is_updating"] = True

        try:
            if not ib.isConnected():
                try:
                    ib.disconnect()
                except Exception:
                    pass

                ib.connect(host, port, clientId=client_id, timeout=20)
                ib.reqMarketDataType(3 if delayed_market_data else 1)

            with STATE_LOCK:
                APP_STATE["connected"] = ib.isConnected()

            for symbol in symbols:
                for tf in TIMEFRAMES.keys():
                    _, err = update_symbol_timeframe(
                        ib=ib,
                        conn=conn,
                        symbol=symbol,
                        timeframe=tf,
                        what_to_show=what_to_show,
                        use_rth=use_rth,
                        request_pause=request_pause,
                    )

                    key = f"{symbol}:{tf}"
                    with STATE_LOCK:
                        APP_STATE["errors"][key] = err or ""

            now_ts = time.time()
            if wall_refresh_minutes > 0:
                should_update_walls = (
                    last_wall_fetch_ts == 0 or
                    now_ts - last_wall_fetch_ts >= wall_refresh_minutes * 60
                )

                if should_update_walls:
                    for wall_symbol in symbols:
                        wall_result = fetch_call_put_walls(
                            ib=ib,
                            symbol=wall_symbol,
                            sectype="STK",
                            exchange="SMART",
                            currency="USD",
                            expiries=wall_expiries,
                            band=wall_band,
                            mode=wall_mode,
                            batch=wall_batch,
                            wait=wall_wait,
                        )
                        with STATE_LOCK:
                            APP_STATE["walls"][wall_symbol] = wall_result

                    with STATE_LOCK:
                        APP_STATE["last_wall_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    last_wall_fetch_ts = now_ts

            with STATE_LOCK:
                APP_STATE["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        except Exception as e:
            with STATE_LOCK:
                APP_STATE["connected"] = False
                APP_STATE["errors"]["GLOBAL"] = str(e)
            try:
                ib.disconnect()
            except Exception:
                pass

        with STATE_LOCK:
            APP_STATE["last_cycle_seconds"] = time.time() - cycle_start
            APP_STATE["is_updating"] = False
            cycle_seconds = APP_STATE["last_cycle_seconds"] or 0

        sleep_for = max(1, refresh_seconds - cycle_seconds)
        time.sleep(sleep_for)


# ============================================================
# Flask app
# ============================================================

def create_app(conn: sqlite3.Connection, symbols: List[str], params: TQParams) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(HTML)

    @app.route("/api/status")
    def api_status():
        return jsonify(build_status_payload(conn, symbols, params))

    return app


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="IBKR SQLite TrendQuality updater + compact transposed HTML dashboard + call/put walls"
    )

    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--db", default="ibkr_tq_bars.sqlite")

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--client-id", type=int, default=17)
    parser.add_argument("--paper", action="store_true")

    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8050)

    parser.add_argument("--what-to-show", default="TRADES")
    parser.add_argument("--use-rth", action="store_true")
    parser.add_argument("--refresh-seconds", type=int, default=60)
    parser.add_argument("--request-pause", type=float, default=0.25)
    parser.add_argument("--delayed", action="store_true", help="Use delayed market data type 3 for market data / option walls.")

    parser.add_argument("--fast-length", type=int, default=20)
    parser.add_argument("--slow-length", type=int, default=50)
    parser.add_argument("--trend-length", type=int, default=4)
    parser.add_argument("--noise-type", choices=["linear", "squared"], default="linear")
    parser.add_argument("--noise-length", type=int, default=250)
    parser.add_argument("--correction-factor", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=3.0)

    parser.add_argument(
        "--exit-mode",
        choices=["OnTurnRed", "OnLeaveGreen", "OnZeroCross"],
        default="OnZeroCross",
    )

    parser.add_argument("--use-max-loss-stop", action="store_true")
    parser.add_argument("--max-loss-pct", type=float, default=12.0)

    parser.add_argument(
        "--wall-refresh-minutes",
        type=int,
        default=30,
        help="Refresh call/put walls every N minutes. Use 0 to disable.",
    )
    parser.add_argument("--wall-expiries", type=int, default=1)
    parser.add_argument("--wall-band", type=float, default=0.15)
    parser.add_argument("--wall-mode", choices=["oi", "gamma", "both"], default="oi")
    parser.add_argument("--wall-batch", type=int, default=40)
    parser.add_argument("--wall-wait", type=float, default=8.0)

    return parser.parse_args()


def main():
    args = parse_args()

    symbols = [s.upper().strip() for s in args.symbols if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")

    port = args.port
    if port is None:
        port = 7497 if args.paper else 7496

    params = TQParams(
        fast_length=args.fast_length,
        slow_length=args.slow_length,
        trend_length=args.trend_length,
        noise_type=args.noise_type,
        noise_length=args.noise_length,
        correction_factor=args.correction_factor,
        threshold_value=args.threshold,
        exit_mode=args.exit_mode,
        use_max_loss_stop=args.use_max_loss_stop,
        max_loss_pct=args.max_loss_pct,
    )

    conn = init_db(args.db)

    worker = threading.Thread(
        target=updater_loop,
        daemon=True,
        args=(
            conn,
            symbols,
            args.host,
            port,
            args.client_id,
            args.what_to_show,
            args.use_rth,
            args.refresh_seconds,
            args.request_pause,
            args.delayed,
            args.wall_refresh_minutes,
            args.wall_expiries,
            args.wall_band,
            args.wall_mode,
            args.wall_batch,
            args.wall_wait,
        ),
    )
    worker.start()

    app = create_app(conn, symbols, params)

    print()
    print("====================================================")
    print(" IBKR TrendQuality HTML Dashboard + Option Walls")
    print("====================================================")
    print(f"Symbols:        {symbols}")
    print(f"SQLite DB:      {args.db}")
    print(f"IBKR:           {args.host}:{port} clientId={args.client_id}")
    print(f"Use RTH only:   {args.use_rth}")
    print(f"WhatToShow:     {args.what_to_show}")
    print(f"Delayed data:   {args.delayed}")
    print(f"Refresh:        {args.refresh_seconds}s")
    print(f"Wall refresh:   {args.wall_refresh_minutes} min")
    print(f"Dashboard URL:  http://{args.web_host}:{args.web_port}")
    print("====================================================")
    print()

    app.run(
        host=args.web_host,
        port=args.web_port,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
