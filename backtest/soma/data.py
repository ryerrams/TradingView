"""Bar data for the SOMA backtest.

- NQ: 1-minute IBKR bars cached by MBot (read-only) in ~/MBot/backtesting/data/ib_cache,
  one segment per contract, resampled to the test timeframe.
- Crypto: Binance spot klines from the public market-data mirror
  (data-api.binance.vision; api.binance.com is geo-blocked in the US), cached as CSV.
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

MBOT_CACHE = Path.home() / "MBot" / "backtesting" / "data" / "ib_cache"
CACHE = Path(__file__).resolve().parent.parent / "data"
BINANCE = "https://data-api.binance.vision/api/v3"
NQ_TICK = 0.25


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "soma-backtest"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


# ----------------------------------------------------------------------------- NQ
def nq_segments(rule: str) -> list[tuple[str, pd.DataFrame]]:
    """One continuous OHLC frame per NQ contract, resampled to `rule` (e.g. '5min')."""
    by_contract: dict[str, list[Path]] = {}
    for f in sorted(MBOT_CACHE.glob("NQ_*_1min_*.csv")):
        by_contract.setdefault(f.name.split("_")[1], []).append(f)
    out = []
    for contract, files in sorted(by_contract.items(), key=lambda kv: kv[1][0].name.split("_")[3]):
        raw = pd.concat(pd.read_csv(f) for f in files)
        if raw.empty:                             # e.g. NQU4 files are header-only
            continue
        raw["datetime"] = pd.to_datetime(raw["datetime"], utc=True, format="ISO8601")  # naive = UTC
        raw = raw.drop_duplicates("datetime").set_index("datetime").sort_index()
        bars = raw.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna(subset=["open"])
        out.append((contract, bars))
    return out


# ----------------------------------------------------------------------------- crypto
def binance_tick(symbol: str) -> float:
    info = _get(f"{BINANCE}/exchangeInfo?symbol={symbol}")
    for flt in info["symbols"][0]["filters"]:
        if flt["filterType"] == "PRICE_FILTER":
            return float(flt["tickSize"])
    raise ValueError(f"no tick size for {symbol}")


def binance_klines(symbol: str, interval: str, start: str) -> pd.DataFrame:
    """All closed klines from `start` (YYYY-MM-DD) to now, cached and topped up."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"binance_{symbol}_{interval}.csv"
    have = pd.read_csv(path, parse_dates=["time"]) if path.exists() else None
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    if have is not None and len(have):
        start_ms = int(pd.Timestamp(have["time"].iloc[-1]).timestamp() * 1000)
    rows = []
    while True:
        batch = _get(f"{BINANCE}/klines?symbol={symbol}&interval={interval}&startTime={start_ms}&limit=1000")
        if not batch:
            break
        rows += batch
        if len(batch) < 1000:
            break
        start_ms = batch[-1][0] + 1
        time.sleep(0.1)
    new = pd.DataFrame([r[:6] for r in rows], columns=["time", "open", "high", "low", "close", "volume"])
    new["time"] = pd.to_datetime(new["time"], unit="ms", utc=True)
    new[["open", "high", "low", "close", "volume"]] = new[["open", "high", "low", "close", "volume"]].astype(float)
    if have is not None:
        have["time"] = pd.to_datetime(have["time"], utc=True)
        new = pd.concat([have, new])
    new = new.drop_duplicates("time", keep="last").sort_values("time")
    new.to_csv(path, index=False)
    df = new.set_index("time")
    # drop the still-forming last bar
    step = pd.Timedelta(interval.replace("m", "min")) if interval.endswith("m") else pd.Timedelta(interval)
    if len(df) and df.index[-1] + step > pd.Timestamp.now(tz="UTC"):
        df = df.iloc[:-1]
    return df


TF_SECONDS = {"5min": 300, "15min": 900, "30min": 1800, "1h": 3600, "4h": 14400, "1d": 86400}
