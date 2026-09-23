"""
call_wall.py
================================================================================
Find the CALL WALL (and PUT WALL) for a symbol from IBKR option data.

  call wall = strike with the largest CALL open interest (or OI*gamma) at/above spot
  put wall  = strike with the largest PUT  open interest (or OI*gamma) at/below spot

Open interest is not a default field: it arrives via generic tick 101.
Gamma arrives via the option's modelGreeks. Both are pulled per strike for the
nearest N expirations, within a +/- band around spot, in pacing-friendly batches.

USAGE (run on your machine with TWS/Gateway up):
  python call_wall.py SPY
  python call_wall.py NVDA --expiries 2 --band 0.20 --mode both --out nvda_walls.csv
  python call_wall.py SPX  --sectype IND --delayed
  python call_wall.py AAPL --port 7497 --clientid 33

NOTES
  * Needs OPRA option market-data permission for live OI; --delayed uses delayed
    data (reqMarketDataType 3), which still carries OI for many names.
  * OI updates ~once per day (exchange-reported), so this is an end-of-day-ish
    snapshot, not real-time.
  * IBKR gives TOTAL OI per strike, not dealer-side positioning, so the wall is
    an approximation of where hedging pressure clusters, not signed dealer gamma.
================================================================================
"""

import argparse
import math
from collections import defaultdict


def is_num(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def fnum(x, default=0.0):
    return float(x) if is_num(x) else default


def main():
    ap = argparse.ArgumentParser(description="IBKR call-wall / put-wall finder")
    ap.add_argument("symbol")
    ap.add_argument("--sectype", default="STK", choices=["STK", "IND"])
    ap.add_argument("--exchange", default="SMART")
    ap.add_argument("--currency", default="USD")
    ap.add_argument("--expiries", type=int, default=1,
                    help="number of nearest expirations to include (default 1)")
    ap.add_argument("--band", type=float, default=0.15,
                    help="strike band +/- around spot as a fraction (default 0.15 = +/-15%%)")
    ap.add_argument("--mode", default="oi", choices=["oi", "gamma", "both"],
                    help="oi = raw open interest; gamma = OI*gamma*100; both = report each")
    ap.add_argument("--delayed", action="store_true",
                    help="use delayed market data (reqMarketDataType 3) if no live OPRA sub")
    ap.add_argument("--port", type=int, default=7497)
    ap.add_argument("--clientid", type=int, default=33)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--batch", type=int, default=40, help="market-data lines per batch")
    ap.add_argument("--wait", type=float, default=8.0, help="seconds to let each batch populate")
    ap.add_argument("--out", help="write the per-strike profile to this CSV")
    args = ap.parse_args()

    from ib_async import IB, Stock, Index, Option, util

    ib = IB()
    ib.connect(args.host, args.port, clientId=args.clientid)
    ib.reqMarketDataType(3 if args.delayed else 1)

    try:
        und = (Stock(args.symbol, args.exchange, args.currency) if args.sectype == "STK"
               else Index(args.symbol, args.exchange, args.currency))
        ib.qualifyContracts(und)
        if not und.conId:
            raise RuntimeError(f"Could not qualify {args.symbol} ({args.sectype}/{args.exchange}).")

        # ---- spot ----
        [snap] = ib.reqTickers(und)
        spot = snap.marketPrice()
        if not is_num(spot) or spot <= 0:
            spot = fnum(snap.close) or fnum(snap.last)
        if not spot or spot <= 0:
            raise RuntimeError("Could not get a spot price for the underlying.")
        print(f"\n{args.symbol}  spot ~ {spot:.2f}")

        # ---- option chain ----
        chains = ib.reqSecDefOptParams(und.symbol, "", und.secType, und.conId)
        if not chains:
            raise RuntimeError("No option chain returned (no options or no permission).")
        chain = next((c for c in chains if c.exchange == "SMART"),
                     max(chains, key=lambda c: len(c.strikes)))
        expirations = sorted(e for e in chain.expirations if e)[:max(1, args.expiries)]
        lo, hi = spot * (1 - args.band), spot * (1 + args.band)
        strikes = sorted(k for k in chain.strikes if lo <= k <= hi)
        if not strikes:
            raise RuntimeError("No strikes inside the band; widen --band.")
        print(f"expirations: {', '.join(expirations)}")
        print(f"strikes in band [{lo:.2f}, {hi:.2f}]: {len(strikes)}  "
              f"-> {len(strikes) * len(expirations) * 2} option lines\n")

        # ---- build + qualify contracts ----
        opts = []
        for exp in expirations:
            for k in strikes:
                for right in ("C", "P"):
                    opts.append(Option(args.symbol, exp, k, right, args.exchange,
                                       multiplier="100", currency=args.currency,
                                       tradingClass=chain.tradingClass))
        ib.qualifyContracts(*opts)
        opts = [o for o in opts if o.conId]
        if not opts:
            raise RuntimeError("No option contracts qualified.")

        # ---- pull OI + gamma in pacing-friendly batches ----
        call_oi = defaultdict(float)
        put_oi = defaultdict(float)
        call_gw = defaultdict(float)   # OI * gamma * 100
        put_gw = defaultdict(float)
        seen_gamma = False

        for i in range(0, len(opts), args.batch):
            chunk = opts[i:i + args.batch]
            tks = [ib.reqMktData(o, genericTickList="100,101,104,106", snapshot=False)
                   for o in chunk]
            ib.sleep(args.wait)
            for o, tk in zip(chunk, tks):
                oi = fnum(tk.callOpenInterest if o.right == "C" else tk.putOpenInterest)
                g = (tk.modelGreeks.gamma if (tk.modelGreeks and is_num(tk.modelGreeks.gamma))
                     else float("nan"))
                gw = oi * g * 100 if is_num(g) else 0.0
                if is_num(g):
                    seen_gamma = True
                if o.right == "C":
                    call_oi[o.strike] += oi
                    call_gw[o.strike] += gw
                else:
                    put_oi[o.strike] += oi
                    put_gw[o.strike] += gw
            for o in chunk:
                ib.cancelMktData(o)
            print(f"  pulled {min(i + args.batch, len(opts))}/{len(opts)} lines")
    finally:
        ib.disconnect()

    # ---- walls ----
    def wall(profile, side):
        # call wall at/above spot, put wall at/below spot; fall back to overall max
        sub = {k: v for k, v in profile.items() if (k >= spot if side == "call" else k <= spot) and v > 0}
        pool = sub or {k: v for k, v in profile.items() if v > 0}
        return max(pool, key=pool.get) if pool else None

    if not any(call_oi.values()) and not any(put_oi.values()):
        print("\nNo open interest came back. Likely no OPRA permission, a thin name, "
              "or the market is closed with no cached OI. Try --delayed.")
        return
    if args.mode in ("gamma", "both") and not seen_gamma:
        print("\n[warn] no gamma populated (model greeks empty) — gamma walls will be blank.")

    cw_oi, pw_oi = wall(call_oi, "call"), wall(put_oi, "put")
    cw_g, pw_g = wall(call_gw, "call"), wall(put_gw, "put")

    print("\n" + "=" * 60)
    if args.mode in ("oi", "both"):
        print(f"  CALL WALL (max call OI):    {cw_oi:.2f}   OI {int(call_oi.get(cw_oi, 0)):,}"
              if cw_oi else "  CALL WALL (OI): n/a")
        print(f"  PUT  WALL (max put OI):     {pw_oi:.2f}   OI {int(put_oi.get(pw_oi, 0)):,}"
              if pw_oi else "  PUT  WALL (OI): n/a")
    if args.mode in ("gamma", "both") and seen_gamma:
        print(f"  CALL WALL (max call gamma): {cw_g:.2f}"
              if cw_g else "  CALL WALL (gamma): n/a")
        print(f"  PUT  WALL (max put gamma):  {pw_g:.2f}"
              if pw_g else "  PUT  WALL (gamma): n/a")
    print("=" * 60)

    # ---- per-strike profile ----
    all_strikes = sorted(set(call_oi) | set(put_oi))
    print(f"\n{'strike':>9} {'callOI':>10} {'putOI':>10} {'netGamma':>12}   marker")
    for k in all_strikes:
        net_g = call_gw.get(k, 0) - put_gw.get(k, 0)   # rough: + call gamma, - put gamma
        mark = ""
        if k == cw_oi:
            mark += " <CALL WALL"
        if k == pw_oi:
            mark += " <PUT WALL"
        if abs(k - spot) <= (all_strikes[1] - all_strikes[0]) / 2 if len(all_strikes) > 1 else False:
            mark += " <~spot"
        print(f"{k:>9.2f} {int(call_oi.get(k, 0)):>10,} {int(put_oi.get(k, 0)):>10,} "
              f"{net_g:>12,.0f}  {mark}")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["strike", "call_oi", "put_oi", "call_gamma_wt", "put_gamma_wt", "net_gamma_wt"])
            for k in all_strikes:
                w.writerow([k, int(call_oi.get(k, 0)), int(put_oi.get(k, 0)),
                            round(call_gw.get(k, 0), 2), round(put_gw.get(k, 0), 2),
                            round(call_gw.get(k, 0) - put_gw.get(k, 0), 2)])
        print(f"\nprofile written -> {args.out}")


if __name__ == "__main__":
    main()