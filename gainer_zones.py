"""
gainer_zones.py
================================================================================
Pull today's top % gainers from IBKR, filter them down to something tradeable,
then measure each survivor against the Sibbet Demand Index supply/demand model
on 5-minute bars.

Two stages:

  1. SCAN   IBKR's TOP_PERC_GAIN scanner does the heavy lifting server-side --
            % change, price, volume and MARKET CAP floors are all scanner-side
            filters, so we never pull bars for names that cannot qualify.

  2. MEASURE  For each survivor, fetch 5m bars and compute:
              - the Demand Index (DMI) and the live demand zone, if any
              - session VWAP and whether price is above it
              - position in the day's range
              - retracement from the high of day

OUTPUT IS DESCRIPTIVE STATE, NOT A RECOMMENDATION. The columns say where price
is relative to structure. What to do about that is your call.

REQUIREMENTS
  pip install ib_async pandas numpy      (ib_insync also works)

USAGE (run with TWS/Gateway up):
  python gainer_zones.py
  python gainer_zones.py --min-pct 20 --min-mcap 500 --max-symbols 15
  python gainer_zones.py --in-zone-only --csv gainers.csv
  python gainer_zones.py --delayed          # no live data subscription

NOTES
  * The IBKR scanner caps results at 50 rows and needs market-data permissions
    for the location you scan. Without them reqScannerData returns empty.
  * marketCapAbove1e6 is denominated in MILLIONS -- '300' means $300M.
  * Market cap is enforced by the scanner but not re-fetched for display, since
    reading it back requires a Reuters fundamentals subscription.
  * Historical-data pacing is ~60 requests / 10 min. --sleep spaces the bar
    requests out; raising --max-symbols much past 30 will start throttling.
  * TWS logs a benign "API scanner subscription cancelled" (error 162) when the
    one-shot scan tears down. It is filtered out below; the data still arrives.
================================================================================
"""

import argparse
import logging
import math
import re
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
# ib_async is the maintained fork of ib_insync; the API is the same.
try:
    from ib_async import IB, ScannerSubscription, Stock, TagValue, util
except ImportError:
    from ib_insync import IB, ScannerSubscription, Stock, TagValue, util


# ============================================================
# Demand Index engine  (port of the Pine/TOS logic)
# ============================================================

def demand_index(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Attach DMI plus the running demand/supply zone bounds to a bar frame.

    Mirrors SD_Zones.pine exactly: volume is split into buying vs selling
    pressure by the direction of the weighted close, smoothed recursively,
    then ratioed into a normalised oscillator.
    """
    h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

    wc = (h + l + 2 * c) * 0.25
    wc_rate = (wc - wc.shift(1)) / np.minimum(wc, wc.shift(1))

    # NB: the rolling means are NaN over the warm-up window. Left unguarded that
    # NaN reaches the recursive smoother below and poisons every later bar, so
    # every intermediate is scrubbed to 0 -- matching Pine's nz() behaviour.
    rng_avg = (h.rolling(2).max() - l.rolling(2).min()).rolling(n).mean().to_numpy()
    safe_rng = np.where((rng_avg == 0) | np.isnan(rng_avg), 1.0, rng_avg)
    volatility = np.nan_to_num(
        3 * wc.to_numpy() / safe_rng * np.abs(np.nan_to_num(wc_rate.to_numpy()))
    )
    volatility = np.where(np.isnan(rng_avg) | (rng_avg == 0), 0.0, volatility)

    vol_avg = v.rolling(n).mean().to_numpy()
    safe_vol = np.where((vol_avg == 0) | np.isnan(vol_avg), 1.0, vol_avg)
    volume_ratio = np.nan_to_num(v.to_numpy() / safe_vol)
    volume_ratio = np.where(np.isnan(vol_avg) | (vol_avg == 0), 0.0, volume_ratio)
    vol_per_range = np.nan_to_num(volume_ratio / np.exp(np.minimum(88.0, volatility)))

    rate = np.nan_to_num(wc_rate.to_numpy())
    buy_p = np.where(rate > 0, volume_ratio, vol_per_range)
    sell_p = np.where(rate > 0, vol_per_range, volume_ratio)

    size = len(df)
    buy_pres = np.zeros(size)
    sell_pres = np.zeros(size)
    dmi = np.zeros(size)

    for i in range(size):
        if i == 0:
            buy_pres[i] = 0.0
            sell_pres[i] = 0.0
            buy_raw = sell_raw = 0.0
        else:
            buy_raw = (buy_pres[i - 1] * (n - 1) + buy_p[i]) / n
            sell_raw = (sell_pres[i - 1] * (n - 1) + sell_p[i]) / n
            buy_pres[i] = buy_raw
            sell_pres[i] = sell_raw

        if (sell_raw - buy_raw) > 0:
            di = -(buy_pres[i] / sell_pres[i] if sell_pres[i] != 0 else 1.0)
        else:
            di = sell_pres[i] / buy_pres[i] if buy_pres[i] != 0 else 1.0
        dmi[i] = (-1 - di) if di < 0 else (1 - di)

    out = df.copy()
    out["dmi"] = dmi
    return out


def live_demand_zone(df: pd.DataFrame, n: int = 5, thr: float = 0.35, start: int = 1):
    """Return (low, high, bar_index, mitigated) of the most recent demand zone.

    Same lifecycle as the Pine version: a cross below -thr stamps the zone, it
    ratchets lower while DMI < -0.2, and it dies on a close beneath it.

    `start` bounds the scan. The DMI itself is always computed over the full
    frame so it is warm, but restricting the scan to today's bars stops a
    prior-session zone being reported as current structure -- on a stock that
    gapped, yesterday's zone can sit 15% away and is not a level in play.
    """
    dmi = df["dmi"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()

    z_lo = z_hi = None
    z_bar = None
    dead = False

    for i in range(max(1, start), len(df)):
        crossed = dmi[i] < -thr <= dmi[i - 1]
        if crossed:
            z_lo, z_hi, z_bar, dead = low[i], high[i], i, False
        elif z_lo is not None and not dead and dmi[i] < -0.2 and low[i] < z_lo:
            z_lo, z_hi, z_bar = low[i], high[i], i
        if z_lo is not None and not dead and close[i] < z_lo:
            dead = True

    return z_lo, z_hi, z_bar, dead


# ============================================================
# IBKR
# ============================================================

class _DropScannerCancelNoise(logging.Filter):
    """TWS reports the one-shot scanner teardown as error 162. Harmless."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "scanner subscription cancelled" not in record.getMessage().lower()


def _silence_scanner_noise() -> None:
    """Filters do not inherit through log propagation, so attach ours to the
    handlers that actually emit -- including logging.lastResort, which is what
    prints when no basicConfig has been called."""
    f = _DropScannerCancelNoise()
    logging.lastResort.addFilter(f)
    for h in logging.root.handlers:
        h.addFilter(f)
    for name in ("ib_async", "ib_async.wrapper", "ib_insync", "ib_insync.wrapper"):
        logging.getLogger(name).addFilter(f)


def connect_ib(host: str, port: int, client_id: int) -> IB:
    _silence_scanner_noise()
    ib = IB()
    ib.connect(host, port, clientId=client_id, timeout=20)
    return ib


def scan_gainers(ib: IB, args) -> list:
    """Top % gainers, filtered server-side. Returns a list of symbols."""
    sub = ScannerSubscription(
        instrument="STK",
        locationCode=args.location,
        scanCode="TOP_PERC_GAIN",
        numberOfRows=50,
        stockTypeFilter="CORP" if args.corp_only else "ALL",
    )
    filters = [
        TagValue("changePercAbove", str(args.min_pct)),
        TagValue("priceAbove", str(args.min_price)),
        TagValue("volumeAbove", str(args.min_volume)),
        TagValue("marketCapAbove1e6", str(args.min_mcap)),   # millions
    ]

    rows = ib.reqScannerData(sub, [], filters)
    if not rows:
        print("Scanner returned nothing. Check market-data permissions for "
              f"{args.location}, or loosen the filters.", file=sys.stderr)
        return []

    symbols = []
    for r in rows:
        sym = r.contractDetails.contract.symbol
        if args.drop_derivatives and re.fullmatch(r"[A-Z]{4}[WUR]", sym):
            continue                      # Nasdaq warrant / unit / right convention
        symbols.append(sym)
    return symbols


def fetch_bars(ib: IB, symbol: str, delayed: bool) -> pd.DataFrame:
    contract = Stock(symbol, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception:
        return pd.DataFrame()

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="2 D",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=2,
        keepUpToDate=False,
    )
    return util.df(bars) if bars else pd.DataFrame()


# ============================================================
# Per-symbol measurement
# ============================================================

def measure(symbol: str, bars: pd.DataFrame, args) -> Optional[dict]:
    if bars.empty or len(bars) < args.n * 4:
        return None

    bars = bars.reset_index(drop=True)
    bars["date"] = pd.to_datetime(bars["date"])
    session = bars["date"].dt.date
    today = session.iloc[-1]
    tdf = bars[session == today]
    if len(tdf) < 3:
        return None

    last = float(tdf["close"].iloc[-1])
    hod = float(tdf["high"].max())
    lod = float(tdf["low"].min())

    tp = (tdf["high"] + tdf["low"] + tdf["close"]) / 3
    vwap = float((tp * tdf["volume"]).cumsum().iloc[-1] / tdf["volume"].cumsum().iloc[-1])

    # NB: retracement off the high is exactly 1 - rng_pos, so it is not tracked
    # separately -- carrying both was double-counting one measurement.
    rng_pos = (last - lod) / (hod - lod) if hod > lod else float("nan")

    scored = demand_index(bars, n=args.n)
    first_today = int(np.argmax((session == today).to_numpy()))
    z_lo, z_hi, z_bar, dead = live_demand_zone(
        scored, n=args.n, thr=args.threshold,
        start=first_today if args.today_only else 1,
    )

    zone_session = None
    if z_bar is not None:
        zone_session = "today" if bars["date"].iloc[z_bar].date() == today else "prior"

    in_zone = False
    dist = float("nan")
    if z_lo is not None and not dead:
        in_zone = z_lo <= last <= z_hi
        dist = (last - z_hi) / last * 100.0        # % above the proximal edge

    # ib_async returns actual share volume here, not round lots -- verified
    # against known session totals (RZLV, CRML) rather than assumed.
    dollar_vol = float(tdf["volume"].sum()) * last

    return {
        "symbol": symbol,
        "last": last,
        "vwap_pct": (last - vwap) / vwap * 100.0,
        "above_vwap": last > vwap,
        "rng_pos": rng_pos,
        "dollar_vol_m": dollar_vol / 1e6,
        "dmi": float(scored["dmi"].iloc[-1]),
        "zone_lo": z_lo,
        "zone_hi": z_hi,
        "zone_dead": dead,
        "zone_session": zone_session,
        "in_zone": in_zone,
        "dist_to_zone": dist,
    }


def criteria_hits(row: dict, args) -> list:
    """Which structural conditions this name currently satisfies.

    RANGE is a band, not a floor. The old build tested "above mid-range" AND
    "retraced 10-40%", which are the same axis measured twice -- any name in
    the band tripped both and its condition count was inflated.
    """
    hits = []
    if row["above_vwap"]:
        hits.append("VWAP")
    if not math.isnan(row["rng_pos"]) and args.min_rng <= row["rng_pos"] <= args.max_rng:
        hits.append("RANGE")
    if row["in_zone"]:
        hits.append("IN-ZONE")
    elif not math.isnan(row["dist_to_zone"]) and abs(row["dist_to_zone"]) <= args.max_dist:
        hits.append("NEAR")
    elif row["zone_lo"] is not None and not row["zone_dead"]:
        hits.append("zone-far")
    return hits


def proximity_key(r: dict):
    """Sort by how close price actually is to its zone.

    The old build sorted by condition count, which ranked the most EXTENDED
    names top -- a name could satisfy three conditions while sitting 17% away
    from the only zone it had.
    """
    live = r["zone_lo"] is not None and not r["zone_dead"]
    if not live or math.isnan(r["dist_to_zone"]):
        return (2, 0.0, -r["dollar_vol_m"])
    return (0 if r["in_zone"] else 1, abs(r["dist_to_zone"]), -r["dollar_vol_m"])


# ============================================================
# Output
# ============================================================

def render(rows: list, args) -> None:
    if not rows:
        print("\nNo symbols survived the filters.")
        return

    rows.sort(key=proximity_key)

    hdr = (f"{'SYM':<7}{'LAST':>9}{'vsVWAP':>9}{'RNG':>6}"
           f"{'$VOL(M)':>10}{'DMI':>7}{'ZONE':>19}{'FROM':>7}{'DIST':>8}  CONDITIONS MET")
    print("\n" + hdr)
    print("-" * len(hdr))

    for r in rows:
        zone = "-"
        if r["zone_lo"] is not None:
            tag = " (spent)" if r["zone_dead"] else ""
            zone = f"{r['zone_lo']:.2f}-{r['zone_hi']:.2f}{tag}"
        dist = "-" if math.isnan(r["dist_to_zone"]) else f"{r['dist_to_zone']:+.1f}%"
        rng = "-" if math.isnan(r["rng_pos"]) else f"{r['rng_pos']:.2f}"
        sess = r["zone_session"] or "-"

        print(f"{r['symbol']:<7}{r['last']:>9.2f}{r['vwap_pct']:>+8.1f}%{rng:>6}"
              f"{r['dollar_vol_m']:>10.1f}{r['dmi']:>7.2f}{zone:>19}{sess:>7}{dist:>8}  "
              f"{' '.join(r['hits']) if r['hits'] else '--'}")

    print("\nRNG  = position in today's range (0 = at low, 1 = at high)")
    print("FROM = which session stamped the zone. 'prior' means it is not today's structure")
    print("DIST = distance from the demand zone's upper edge; negative means price is at or below it")
    print("Sorted by proximity to the zone -- in-zone first, then nearest.")
    print("\nThis is a description of where price sits relative to structure.")
    print("It is not a recommendation, and it says nothing about what happens next.\n")


def main():
    ap = argparse.ArgumentParser(description="IBKR top-gainer demand-zone screener")

    # scan filters
    ap.add_argument("--min-pct", type=float, default=15.0, help="minimum %% change (default 15)")
    ap.add_argument("--min-price", type=float, default=1.0, help="minimum share price (default 1)")
    ap.add_argument("--min-volume", type=int, default=1_000_000, help="minimum share volume")
    ap.add_argument("--min-mcap", type=float, default=300.0,
                    help="minimum market cap in MILLIONS (default 300 = $300M)")
    ap.add_argument("--location", default="STK.US.MAJOR")
    ap.add_argument("--corp-only", action="store_true", default=True,
                    help="exclude ETFs and ADRs (default on -- drops leveraged ETFs)")
    ap.add_argument("--keep-etfs", dest="corp_only", action="store_false")
    ap.add_argument("--drop-derivatives", action="store_true", default=True,
                    help="drop 5-letter symbols ending W/U/R (warrants, units, rights; default on)")
    ap.add_argument("--keep-derivatives", dest="drop_derivatives", action="store_false")

    # structure thresholds
    ap.add_argument("--n", type=int, default=5, help="Demand Index lookback (default 5)")
    ap.add_argument("--threshold", type=float, default=0.35, help="DI zone threshold (default 0.35)")
    ap.add_argument("--min-rng", type=float, default=0.60,
                    help="lower edge of the range band (default 0.60)")
    ap.add_argument("--max-rng", type=float, default=0.92,
                    help="upper edge of the range band (default 0.92 -- above this is extended)")
    ap.add_argument("--max-dist", type=float, default=3.0,
                    help="%% from the zone edge that still counts as NEAR (default 3)")
    ap.add_argument("--today-only", action="store_true",
                    help="ignore zones stamped in a prior session")
    ap.add_argument("--in-zone-only", action="store_true", help="only print names inside a live zone")

    # plumbing
    ap.add_argument("--max-symbols", type=int, default=25, help="cap bar requests (pacing)")
    ap.add_argument("--sleep", type=float, default=1.2, help="seconds between bar requests")
    ap.add_argument("--delayed", action="store_true", help="use delayed market data")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--clientid", type=int, default=41)
    ap.add_argument("--csv", help="also write results to this path")

    args = ap.parse_args()

    ib = connect_ib(args.host, args.port, args.clientid)
    if args.delayed:
        ib.reqMarketDataType(3)

    try:
        print(f"Scanning: >{args.min_pct}% | >${args.min_price} | "
              f">{args.min_volume:,} shares | >${args.min_mcap:.0f}M cap")
        symbols = scan_gainers(ib, args)
        if not symbols:
            return
        print(f"{len(symbols)} passed the scan: {', '.join(symbols)}")

        symbols = symbols[: args.max_symbols]
        if len(symbols) == args.max_symbols:
            print(f"(measuring the first {args.max_symbols} -- raise --max-symbols to widen)")

        rows = []
        for i, sym in enumerate(symbols):
            print(f"  [{i+1}/{len(symbols)}] {sym}", end="\r", flush=True)
            row = measure(sym, fetch_bars(ib, sym, args.delayed), args)
            if row:
                row["hits"] = criteria_hits(row, args)
                if not args.in_zone_only or row["in_zone"]:
                    rows.append(row)
            time.sleep(args.sleep)
        print(" " * 40, end="\r")

        render(rows, args)

        if args.csv and rows:
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"Written to {args.csv}")

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
