#!/usr/bin/env python3
"""
Pull one CME-style trading session from Interactive Brokers and write it to CSV.

A session runs 17:00 America/Chicago on the prior day to 16:00 the next day,
which is 23 hours of tape. IB caps how much intraday history it returns per
request, so this pulls in one-day chunks backward from the session close and
stitches them together.

    python ib_fetch_session.py --symbol MNQ --sec-type FUT --exchange CME \
        --expiry 202609 --date 2026-08-28 --out session.csv

TWS or IB Gateway must be running with the API enabled. Default port 7497 is
paper TWS; use 7496 for live TWS, 4002 / 4001 for Gateway paper / live.

Requires: pip install ib_async pandas    (falls back to ib_insync if present)
"""
import argparse
import datetime as dt
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd

try:
    from ib_async import IB, Future, Stock, Contract
except ImportError:                                    # older installs
    try:
        from ib_insync import IB, Future, Stock, Contract
    except ImportError:
        sys.exit("Install the API client first:  pip install ib_async")

CT = ZoneInfo("America/Chicago")


def session_window(session_date: dt.date, start_hour: int, end_hour: int):
    """17:00 the day before the session date, to 16:00 on it."""
    end = dt.datetime.combine(session_date, dt.time(end_hour, 0), tzinfo=CT)
    start = dt.datetime.combine(session_date - dt.timedelta(days=1),
                                dt.time(start_hour, 0), tzinfo=CT)
    return start, end


def build_contract(args):
    if args.sec_type == "FUT":
        if not args.expiry:
            sys.exit("--expiry is required for futures, e.g. --expiry 202609")
        return Future(symbol=args.symbol, lastTradeDateOrContractMonth=args.expiry,
                      exchange=args.exchange, currency=args.currency)
    if args.sec_type == "STK":
        return Stock(args.symbol, args.exchange or "SMART", args.currency)
    c = Contract(secType=args.sec_type, symbol=args.symbol,
                 exchange=args.exchange, currency=args.currency)
    if args.expiry:
        c.lastTradeDateOrContractMonth = args.expiry
    return c


def fetch(args):
    start, end = session_window(dt.date.fromisoformat(args.date),
                                args.session_start_hour, args.session_end_hour)
    print(f"session window: {start:%Y-%m-%d %H:%M %Z}  ->  {end:%Y-%m-%d %H:%M %Z}")

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.client_id, timeout=15)
    contract = build_contract(args)
    ib.qualifyContracts(contract)
    print(f"contract: {contract.localSymbol or contract.symbol} "
          f"({contract.secType} on {contract.exchange})")

    frames, cursor = [], end
    for i in range(args.chunks):
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=cursor,
            durationStr=args.chunk_duration,
            barSizeSetting=args.bar_size,
            whatToShow=args.what_to_show,
            useRTH=False,
            formatDate=2,          # epoch seconds, timezone-unambiguous
        )
        if not bars:
            print(f"  chunk {i + 1}: empty, stopping")
            break
        df = pd.DataFrame([{
            "time": b.date, "open": b.open, "high": b.high,
            "low": b.low, "close": b.close, "volume": b.volume,
        } for b in bars])
        print(f"  chunk {i + 1}: {len(df)} bars ending {cursor:%Y-%m-%d %H:%M}")
        frames.append(df)
        cursor = cursor - dt.timedelta(days=1)
        if i < args.chunks - 1:
            time.sleep(args.pacing)     # IB pacing: stay well under the limits

    ib.disconnect()
    if not frames:
        sys.exit("no data returned. Check the contract, the date, and market data permissions.")

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(CT)
    df = (df.drop_duplicates(subset="time")
            .sort_values("time")
            .reset_index(drop=True))

    if args.keep_history:
        inside = df[df["time"] < end].reset_index(drop=True)
        print(f"keeping {len(inside) - len(df[(df['time'] >= start) & (df['time'] < end)])} "
              f"pre-session bars for ATR warmup")
    else:
        inside = df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)
    if inside.empty:
        sys.exit("no bars fell inside the session window. Wrong date, or a holiday.")

    got = inside["time"].iloc[-1] - inside["time"].iloc[0]
    print(f"kept {len(inside)} bars spanning {got}, "
          f"{inside['time'].iloc[0]:%H:%M} to {inside['time'].iloc[-1]:%H:%M} CT")
    print(f"range {inside['low'].min():.2f} to {inside['high'].max():.2f}, "
          f"total volume {inside['volume'].sum():,.0f}")

    inside.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", required=True)
    p.add_argument("--sec-type", default="FUT", choices=["FUT", "STK", "IND", "CASH"])
    p.add_argument("--exchange", default="CME")
    p.add_argument("--currency", default="USD")
    p.add_argument("--expiry", help="futures contract month, e.g. 202609")
    p.add_argument("--date", required=True, help="session CLOSE date, YYYY-MM-DD")
    p.add_argument("--bar-size", default="1 min",
                   help="'1 min' gives the best price resolution for the profile")
    p.add_argument("--what-to-show", default="TRADES")
    p.add_argument("--chunks", type=int, default=2,
                   help="one-day pulls stitched backward from the close")
    p.add_argument("--chunk-duration", default="1 D")
    p.add_argument("--pacing", type=float, default=11.0,
                   help="seconds between requests, IB throttles below ~10s")
    p.add_argument("--session-start-hour", type=int, default=17)
    p.add_argument("--session-end-hour", type=int, default=16)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7497)
    p.add_argument("--client-id", type=int, default=17)
    p.add_argument("--keep-history", action="store_true",
                   help="keep bars from before the session in the csv. vp_replay uses "
                        "them to warm up ATR and does not replay them.")
    p.add_argument("--out", default="session.csv")
    fetch(p.parse_args())


if __name__ == "__main__":
    main()
