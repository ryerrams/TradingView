"""
mover_zones.py
================================================================================
Post-close companion to gainer_zones.py.

Pull the biggest AFTER-HOURS movers from IBKR -- both sides, gainers and losers
-- then measure each one against the structure the REGULAR session left behind.

Two stages, same shape as the intraday screener:

  1. SCAN     TOP_AFTER_HOURS_PERC_GAIN / _LOSE do the work server-side.
              afterHoursChangePerc, price, regular-session volume and market cap
              are all scanner-side filters, so we never pull bars for a name that
              cannot qualify.

  2. MEASURE  For each survivor, fetch extended-hours 5m bars and compute:
              - the after-hours move off the regular close, and how much of the
                extreme it is still holding
              - after-hours volume, in dollars and as a share of the day's volume
              - where the after-hours price sits against regular-session VWAP,
                the day's range, and the trailing 5-session range
              - the side-appropriate Sibbet supply/demand zone -- computed on
                REGULAR-SESSION bars only -- and the distance to it

WHY THE ZONE COMES FROM THE REGULAR SESSION
  The Demand Index is a volume-normalised recursive smoother over a 5-bar window.
  After the close there are at most 48 bars, most of them near-empty, and the
  first one carries the closing auction plus the earnings gap in a single print
  (observed live: INTU 16:00 bar, high 368.00 / low 301.80). Feeding that to the
  DI produces a number, not a measurement. So structure is read off RTH, where
  the volume the index divides by is real, and the after-hours price is measured
  against it. That is also how the level would actually be traded the next day.

OUTPUT IS DESCRIPTIVE STATE, NOT A RECOMMENDATION. The columns say where price
is relative to structure. What to do about that is your call.

REQUIREMENTS
  pip install ib_async pandas numpy      (ib_insync also works)

USAGE (run with TWS/Gateway up, any time after 16:00 ET):
  python mover_zones.py
  python mover_zones.py --side up --min-ah-pct 5
  python mover_zones.py --min-ah-dollar-vol 1.0 --sort liq
  python mover_zones.py --scan day            # regular-session movers instead
  python mover_zones.py --csv movers.csv

NOTES
  * Historical after-hours bars lag the tape by roughly 10-15 minutes. The header
    prints the timestamp of the newest bar seen so the staleness is never hidden;
    running at 16:05 ET will show you almost nothing.
  * The after-hours scan codes go stale overnight. Once the session is well and
    truly over, --scan day is the honest fallback.
  * The regular close is taken as the last continuous print before 16:00, not
    the closing auction. Every percentage here is measured off that one
    reference, consistently.
  * Post-close volume starts at 16:01, with zero-volume padding bars and any
    late-reported closing cross removed, so the auction is never counted as
    after-hours participation. This costs a second (1-minute) bar request per
    symbol; --fast skips it and accepts the contamination.
  * marketCapAbove1e6 is denominated in MILLIONS -- '300' means $300M.
  * Historical-data pacing is ~60 requests / 10 min and this makes TWO requests
    per symbol, so the default --max-symbols 25 sits just under the ceiling.
    Widen it with --fast, or raise --sleep.
================================================================================
"""

import argparse
import math
import re
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    from ib_async import IB, ScannerSubscription, Stock, TagValue, util
except ImportError:
    from ib_insync import IB, ScannerSubscription, Stock, TagValue, util

# One DI implementation lives in the repo, in the intraday screener. Import it
# rather than fork it -- a second copy would drift from the Pine source.
from gainer_zones import demand_index, connect_ib

ET = "America/New_York"
RTH_OPEN = pd.Timestamp("09:30").time()
RTH_CLOSE = pd.Timestamp("16:00").time()
AH_END = pd.Timestamp("20:00").time()
# IB reports some names' closing cross a minute or two late. Window for it.
LATE_CROSS = pd.Timestamp("16:02").time()


# ============================================================
# Zones -- both sides
# ============================================================

def live_zone(df: pd.DataFrame, side: str, thr: float = 0.35, start: int = 1):
    """Most recent live zone on the given side: (lo, hi, bar_index, mitigated).

    Mirrors SD_Zones.pine for both sides, including its asymmetry -- the demand
    ratchet runs while DMI < -0.2, the supply ratchet while DMI > 0.5. That gap
    is in the TOS original and is deliberately preserved; supply zones are meant
    to be the rarer stamp.

    `start` bounds the scan so a prior session's zone is not reported as today's
    structure. The DI itself is always computed over the whole frame so it is
    warm by the time the scan window opens.
    """
    dmi = df["dmi"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    close = df["close"].to_numpy()

    z_lo = z_hi = None
    z_bar = None
    dead = False

    for i in range(max(1, start), len(df)):
        if side == "demand":
            crossed = dmi[i] < -thr <= dmi[i - 1]
            if crossed:
                z_lo, z_hi, z_bar, dead = low[i], high[i], i, False
            elif z_lo is not None and not dead and dmi[i] < -0.2 and low[i] < z_lo:
                z_lo, z_hi, z_bar = low[i], high[i], i
            if z_lo is not None and not dead and close[i] < z_lo:
                dead = True
        else:
            crossed = dmi[i] > thr >= dmi[i - 1]
            if crossed:
                z_lo, z_hi, z_bar, dead = low[i], high[i], i, False
            elif z_hi is not None and not dead and dmi[i] > 0.5 and high[i] > z_hi:
                z_lo, z_hi, z_bar = low[i], high[i], i
            if z_hi is not None and not dead and close[i] > z_hi:
                dead = True

    return z_lo, z_hi, z_bar, dead


# ============================================================
# IBKR
# ============================================================

SCAN_CODES = {
    ("ah", "up"): ("TOP_AFTER_HOURS_PERC_GAIN", "afterHoursChangePercAbove"),
    ("ah", "down"): ("TOP_AFTER_HOURS_PERC_LOSE", "afterHoursChangePercBelow"),
    ("day", "up"): ("TOP_PERC_GAIN", "changePercAbove"),
    ("day", "down"): ("TOP_PERC_LOSE", "changePercBelow"),
}


def scan_movers(ib: IB, direction: str, args) -> list:
    """One side of the scan. Returns symbols, scanner rank order preserved."""
    code, pct_tag = SCAN_CODES[(args.scan, direction)]
    pct = args.min_ah_pct if direction == "up" else -args.min_ah_pct

    sub = ScannerSubscription(
        instrument="STK",
        locationCode=args.location,
        scanCode=code,
        numberOfRows=50,
        stockTypeFilter="CORP" if args.corp_only else "ALL",
    )
    filters = [
        TagValue(pct_tag, str(pct)),
        TagValue("priceAbove", str(args.min_price)),
        TagValue("volumeAbove", str(args.min_volume)),        # regular-session shares
        TagValue("marketCapAbove1e6", str(args.min_mcap)),    # millions
    ]

    rows = ib.reqScannerData(sub, [], filters)
    if not rows:
        return []

    symbols = []
    for r in rows:
        sym = r.contractDetails.contract.symbol
        if args.drop_derivatives and re.fullmatch(r"[A-Z]{4}[WUR]", sym):
            continue                      # Nasdaq warrant / unit / right convention
        symbols.append(sym)
    return symbols


def fetch_bars(ib: IB, symbol: str, days: int) -> pd.DataFrame:
    """5m bars including extended hours, indexed in Eastern time.

    useRTH=False is the whole point: with it on, the after-hours session -- the
    thing being screened -- simply is not in the response.
    """
    contract = Stock(symbol, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception:
        return pd.DataFrame()

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr=f"{days} D",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()

    df = util.df(bars).reset_index(drop=True)
    # formatDate=2 is epoch-based, so this is a real conversion, not a relabel.
    df["et"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(ET)
    return df


def fetch_minute_bars(ib: IB, symbol: str) -> pd.DataFrame:
    """Today's 1m bars, extended hours included.

    This exists for one reason: the closing auction. On a 5m frame the cross
    lands inside the 16:00 bar together with the first minutes of post-close
    trade, and there is no way to separate them -- so every name's after-hours
    volume carries its auction. That is not a rounding error. Observed live:
    PRIM printed 78.6% of its regular-session volume "after hours" on a move of
    +0.0%, which is a closing cross, not participation. At 1m resolution the
    auction is confined to the 16:00 bar and everything from 16:01 is clean.
    """
    contract = Stock(symbol, "SMART", "USD")
    try:
        ib.qualifyContracts(contract)
    except Exception:
        return pd.DataFrame()

    bars = ib.reqHistoricalData(
        contract,
        endDateTime="",
        durationStr="1 D",
        barSizeSetting="1 min",
        whatToShow="TRADES",
        useRTH=False,
        formatDate=2,
        keepUpToDate=False,
    )
    if not bars:
        return pd.DataFrame()

    df = util.df(bars).reset_index(drop=True)
    df["et"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(ET)
    return df


# ============================================================
# Per-symbol measurement
# ============================================================

def measure(symbol: str, bars: pd.DataFrame, minutes: pd.DataFrame,
            scan_side: str, args) -> Optional[dict]:
    if bars.empty or len(bars) < 20:
        return None

    et = bars["et"]
    day = et.dt.date
    tod = et.dt.time
    today = day.iloc[-1]

    is_today = day == today
    rth = bars[is_today & (tod >= RTH_OPEN) & (tod < RTH_CLOSE)]

    # Post-close window, auction excluded: 16:01 onward on the minute frame. If
    # the minute request came back empty we fall back to the 5m frame from 16:00,
    # which does include the cross -- flagged as 'ah_src' so the CSV never
    # silently mixes the two definitions.
    if len(rth) < 10:
        return None

    # Reference close: the last continuous print before the bell. The minute
    # frame puts that at 15:59 instead of 15:55, so use it when we have it.
    # Checked against IBKR's own useRTH daily close: exact on PRIM and NCNO,
    # 5bp on INTU. Every percentage in the row is measured off this one number.
    rth_close = float(rth["close"].iloc[-1])
    if not minutes.empty:
        pre = minutes[(minutes["et"].dt.date == today) & (minutes["et"].dt.time < RTH_CLOSE)]
        if not pre.empty:
            rth_close = float(pre["close"].iloc[-1])

    ah_src = "1m"
    ah = pd.DataFrame()
    have_minutes = not minutes.empty
    if have_minutes:
        m_tod = minutes["et"].dt.time
        m_today = minutes["et"].dt.date == today
        ah = minutes[m_today & (m_tod > RTH_CLOSE) & (m_tod <= AH_END)].copy()

        # IB pads quiet post-close minutes with zero-volume bars that carry the
        # last price forward. Left in, they set ah_high/ah_low from a price
        # nobody traded -- NCNO carried a 21.22 print across four empty minutes
        # and it became the post-close extreme.
        ah = ah[ah["volume"] > 0]

        # And some names' closing cross is reported a minute or two late, as one
        # flat print at the closing price: PRIM, 387,414 shares at 16:01, with
        # open == high == low == close == 74.80. That is a regular-session event.
        # Counted as post-close volume it made PRIM the most liquid name on the
        # board on a move of +0.0%.
        late_cross = ((ah["et"].dt.time <= LATE_CROSS)
                      & (ah["high"] == ah["low"])
                      & (np.isclose(ah["close"], rth_close)))
        ah = ah[~late_cross]

        # An empty window here is a FINDING, not a gap in the data: the name had
        # a closing cross and nothing since. Falling through to the 5m frame
        # would hand the auction back as after-hours participation.
        if ah.empty:
            return None
    else:
        ah_src = "5m"
        ah = bars[is_today & (tod >= RTH_CLOSE) & (tod <= AH_END)]

    # A name halted through the close, or one the scanner surfaced before any
    # post-close print landed, has nothing to measure against.
    if ah.empty:
        return None
    rth_hod = float(rth["high"].max())
    rth_lod = float(rth["low"].min())
    rth_vol = float(rth["volume"].sum())

    tp = (rth["high"] + rth["low"] + rth["close"]) / 3
    rth_vwap = float((tp * rth["volume"]).sum() / rth["volume"].sum())

    ah_last = float(ah["close"].iloc[-1])
    ah_high = float(ah["high"].max())
    ah_low = float(ah["low"].min())
    ah_vol = float(ah["volume"].sum())
    ah_pct = (ah_last - rth_close) / rth_close * 100.0

    # The scanner's direction is authoritative, not the sign of ah_pct. Bars lag
    # the tape by ~15 minutes, so a freshly-scanned mover can still read flat or
    # even opposite here -- deriving side from the bar made PRIM, a scanned
    # decliner, print as an advancer. Direction is why the name is on the list;
    # the bars say how far it has got, and disagreement is reported, not hidden.
    side = scan_side
    extreme = ah_high if side == "up" else ah_low

    # HOLD: the fraction of the move to the after-hours extreme that is still
    # standing. 1.00 = sitting on the extreme, 0.00 = fully round-tripped back to
    # the close, negative = reversed clean through it. This is the one number
    # that separates a repricing from a spike somebody already sold into.
    span = extreme - rth_close
    if (span <= 1e-9) if side == "up" else (span >= -1e-9):
        # Nothing traded in the scanned direction yet, so there is no move to
        # hold a fraction of. A ratio here would divide by noise and read as
        # conviction; NaN is the honest answer.
        hold = float("nan")
    else:
        hold = (ah_last - rth_close) / span

    # Position in the regular session's range. Deliberately NOT clipped: >1 means
    # the after-hours print is above everything the day session traded, <0 below
    # it, and that overshoot is the informative part.
    rng_pos = (ah_last - rth_lod) / (rth_hod - rth_lod) if rth_hod > rth_lod else float("nan")

    # Trailing 5-session context, regular hours only -- an after-hours pop that
    # merely reclaims last Tuesday's price is a different animal to one that
    # clears the whole window.
    prior = bars[(tod >= RTH_OPEN) & (tod < RTH_CLOSE)]
    w_hi, w_lo = float(prior["high"].max()), float(prior["low"].min())
    pos5 = (ah_last - w_lo) / (w_hi - w_lo) if w_hi > w_lo else float("nan")

    # Structure from the regular session only (see module docstring). Demand is
    # what an up-mover would pull back into; supply is what a down-mover would
    # bounce into. Warm the DI on the full frame, scan only today's RTH bars.
    zone_side = "demand" if side == "up" else "supply"
    scored = demand_index(bars, n=args.n)
    rth_mask = (is_today & (tod >= RTH_OPEN) & (tod < RTH_CLOSE)).to_numpy()
    first_rth_today = int(np.argmax(rth_mask))
    # Truncate at the bell before scanning. The DI is warm across the whole
    # frame, but leaving the post-close bars in let them STAMP zones -- a gap
    # bar would mint a "regular session" level that the regular session never
    # traded, which is the one thing this split is supposed to prevent.
    last_rth_today = int(len(rth_mask) - 1 - np.argmax(rth_mask[::-1]))
    z_lo, z_hi, z_bar, dead = live_zone(
        scored.iloc[: last_rth_today + 1], zone_side,
        thr=args.threshold, start=first_rth_today,
    )

    in_zone = False
    dist = float("nan")
    if z_lo is not None and not dead:
        in_zone = z_lo <= ah_last <= z_hi
        # Signed distance to the edge price would approach from: the top of a
        # demand zone below, the bottom of a supply zone above.
        edge = z_hi if zone_side == "demand" else z_lo
        dist = (ah_last - edge) / ah_last * 100.0

    return {
        "symbol": symbol,
        "side": side,
        "rth_close": rth_close,
        "ah_last": ah_last,
        "ah_pct": ah_pct,
        "hold": hold,
        "ah_vol": ah_vol,
        "ah_dollar_vol_m": ah_vol * ah_last / 1e6,
        "ah_vol_pct_rth": ah_vol / rth_vol * 100.0 if rth_vol else float("nan"),
        "rth_dollar_vol_m": rth_vol * rth_close / 1e6,
        "vwap_pct": (ah_last - rth_vwap) / rth_vwap * 100.0,
        "rng_pos": rng_pos,
        "pos_5d": pos5,
        "zone_side": zone_side,
        "zone_lo": z_lo,
        "zone_hi": z_hi,
        "zone_dead": dead,
        "in_zone": in_zone,
        "dist_to_zone": dist,
        "ah_bars": len(ah),
        "ah_src": ah_src,
        "last_bar": ah["et"].iloc[-1],
    }


def criteria_hits(row: dict, args) -> list:
    """Which structural conditions this name currently satisfies.

    Every test is written in the direction of the mover, so an up-mover and a
    down-mover earn the same tag for the mirror-image condition. A shared tag
    that meant opposite things on the two sides would make the column unreadable.
    """
    up = row["side"] == "up"
    hits = []

    # LIQ first: it is a gate on whether the other tags are worth reading at all.
    # A 30% after-hours move on 4,000 shares is a quote, not a repricing.
    if (row["ah_dollar_vol_m"] >= args.min_ah_dollar_vol
            and row["ah_vol_pct_rth"] >= args.min_ah_vol_pct):
        hits.append("LIQ")

    if (row["vwap_pct"] > 0) == up:
        hits.append("VWAP")

    if not math.isnan(row["hold"]) and row["hold"] >= args.min_hold:
        hits.append("HOLD")

    if not math.isnan(row["rng_pos"]) and ((row["rng_pos"] > 1.0) if up else (row["rng_pos"] < 0.0)):
        hits.append("BREAK")

    # The bars disagree with the scan by more than noise. Either the move has
    # already turned, or it landed after the newest bar -- the lag banner above
    # the table is what tells the two apart.
    if (row["ah_pct"] < -0.25) if up else (row["ah_pct"] > 0.25):
        hits.append("REVERSED")

    if row["in_zone"]:
        hits.append("IN-ZONE")
    elif not math.isnan(row["dist_to_zone"]) and abs(row["dist_to_zone"]) <= args.max_dist:
        hits.append("NEAR")
    elif row["zone_lo"] is not None and not row["zone_dead"]:
        hits.append("zone-far")

    return hits


# ============================================================
# Sorting
# ============================================================

def sort_key(mode: str):
    def by_move(r):
        # Liquidity-qualified names first regardless of mode: an unqualified
        # name topping the list on move size is exactly the trap this screener
        # exists to avoid.
        return (0 if "LIQ" in r["hits"] else 1, -abs(r["ah_pct"]))

    def by_liq(r):
        return (-r["ah_dollar_vol_m"],)

    def by_zone(r):
        live = r["zone_lo"] is not None and not r["zone_dead"]
        if not live or math.isnan(r["dist_to_zone"]):
            return (2, 0.0, -r["ah_dollar_vol_m"])
        return (0 if r["in_zone"] else 1, abs(r["dist_to_zone"]), -r["ah_dollar_vol_m"])

    return {"move": by_move, "liq": by_liq, "zone": by_zone}[mode]


# ============================================================
# Output
# ============================================================

def render(rows: list, args, skipped: int, illiquid: int) -> None:
    if not rows:
        # Two very different reasons to see nothing, and conflating them sends
        # you to the wrong knob: no measurable data vs measured and gated out.
        why = []
        if skipped:
            why.append(f"{skipped} had no post-close prints to measure")
        if illiquid:
            why.append(f"{illiquid} were measured but failed LIQ (--liq-only)")
        print("\nNo symbols survived the filters."
              + (" " + "; ".join(why) + "." if why else ""))
        return

    rows.sort(key=sort_key(args.sort))

    newest = max(r["last_bar"] for r in rows)
    now = pd.Timestamp.now(tz=ET)
    lag = (now - newest).total_seconds() / 60.0
    print(f"\nNewest bar: {newest:%H:%M} ET  ({lag:.0f} min behind {now:%H:%M})")
    if lag > 10:
        # The scanner ranked these off the live tape; every column below is read
        # off bars that trail it. A name can show a small AH% here purely because
        # its move landed after the newest bar -- that is lag, not a fade.
        print(f"  Ranked on live quotes, measured on {lag:.0f}-minute-old bars. "
              f"Re-run later for names whose AH% looks too small to have been scanned.")

    hdr = (f"{'SYM':<7}{'SIDE':>5}{'CLOSE':>9}{'AH':>9}{'AH%':>8}{'HOLD':>6}"
           f"{'AH$M':>7}{'AH/RTH':>8}{'vsVWAP':>8}{'RNG':>6}{'5D':>6}"
           f"{'ZONE':>21}{'DIST':>8}  CONDITIONS MET")
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        zone = "-"
        if r["zone_lo"] is not None:
            tag = " (spent)" if r["zone_dead"] else ""
            zone = f"{r['zone_lo']:.2f}-{r['zone_hi']:.2f}{tag}"
        dist = "-" if math.isnan(r["dist_to_zone"]) else f"{r['dist_to_zone']:+.1f}%"
        # A reversal off a near-zero excursion divides by noise and can run to
        # three digits. REVERSED already carries the meaning; the number just
        # needs to stay inside its column. Raw value is preserved in the CSV.
        if math.isnan(r["hold"]):
            hold = "-"
        elif abs(r["hold"]) > 99:
            hold = "99+" if r["hold"] > 0 else "-99+"
        else:
            hold = f"{r['hold']:.2f}"
        rng = "-" if math.isnan(r["rng_pos"]) else f"{r['rng_pos']:.2f}"
        p5 = "-" if math.isnan(r["pos_5d"]) else f"{r['pos_5d']:.2f}"

        print(f"{r['symbol']:<7}{r['side']:>5}{r['rth_close']:>9.2f}{r['ah_last']:>9.2f}"
              f"{r['ah_pct']:>+7.1f}%{hold:>7}{r['ah_dollar_vol_m']:>7.1f}"
              f"{r['ah_vol_pct_rth']:>7.1f}%{r['vwap_pct']:>+7.1f}%{rng:>6}{p5:>6}"
              f"{zone:>21}{dist:>8}  {' '.join(r['hits']) if r['hits'] else '--'}")

    if skipped:
        print(f"\n{skipped} scanner hit(s) dropped: no post-close prints beyond the "
              f"closing auction, or a regular session too thin to measure against.")
    if illiquid:
        print(f"{illiquid} more measured cleanly but failed LIQ and were hidden by --liq-only.")

    print("\nAH%    = move off the regular-session close (last pre-16:00 bar)")
    print("HOLD   = share of the move to the post-close extreme still standing")
    print("         1.00 = at the extreme, 0.00 = round-tripped, negative = reversed through the close")
    print("AH$M   = post-close dollar volume, closing auction excluded (16:01 onward)")
    print("AH/RTH = those post-close shares as a % of the regular session's")
    print("RNG    = position in the day's range; >1 is above the whole session, <0 below it")
    print("5D     = position in the trailing 5-session regular-hours range")
    print("ZONE   = the side-appropriate S/D zone, computed on REGULAR-SESSION bars only")
    print("         (up-movers get demand below, down-movers supply above)")
    print("DIST   = distance to the edge price would approach from; sign is raw, not directional")
    print("SIDE   = the direction the scanner ranked it in; REVERSED means the bars disagree")
    print("\nConditions are written in the direction of the move, so both sides read the same.")
    print("LIQ is a gate on the rest: without it the other tags describe a handful of prints.")
    print("It is an accumulation over a four-hour session -- a 16:15 run will fail names")
    print("that clear it comfortably by 18:00. Re-run rather than loosening the threshold.")
    print("\nThis is a description of where price sits relative to structure.")
    print("It is not a recommendation, and it says nothing about what happens next.\n")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="IBKR after-hours mover supply/demand screener")

    # scan filters
    ap.add_argument("--scan", choices=["ah", "day"], default="ah",
                    help="'ah' = post-close movers (default), 'day' = regular-session movers")
    ap.add_argument("--side", choices=["up", "down", "both"], default="both")
    ap.add_argument("--min-ah-pct", type=float, default=3.0,
                    help="minimum %% move, magnitude, applied to both sides (default 3)")
    ap.add_argument("--min-price", type=float, default=2.0, help="minimum share price (default 2)")
    ap.add_argument("--min-volume", type=int, default=500_000,
                    help="minimum REGULAR-session share volume (default 500k)")
    ap.add_argument("--min-mcap", type=float, default=300.0,
                    help="minimum market cap in MILLIONS (default 300 = $300M)")
    ap.add_argument("--location", default="STK.US.MAJOR")
    ap.add_argument("--corp-only", action="store_true", default=True,
                    help="exclude ETFs and ADRs (default on)")
    ap.add_argument("--keep-etfs", dest="corp_only", action="store_false")
    ap.add_argument("--drop-derivatives", action="store_true", default=True,
                    help="drop 5-letter symbols ending W/U/R (default on)")
    ap.add_argument("--keep-derivatives", dest="drop_derivatives", action="store_false")

    # structure thresholds
    ap.add_argument("--n", type=int, default=5, help="Demand Index lookback (default 5)")
    ap.add_argument("--threshold", type=float, default=0.35, help="DI zone threshold (default 0.35)")
    ap.add_argument("--min-hold", type=float, default=0.60,
                    help="HOLD needed to earn the tag (default 0.60)")
    ap.add_argument("--min-ah-dollar-vol", type=float, default=0.25,
                    help="post-close dollar volume in MILLIONS for LIQ (default 0.25)")
    ap.add_argument("--min-ah-vol-pct", type=float, default=0.5,
                    help="post-close shares as %% of regular volume for LIQ (default 0.5)")
    ap.add_argument("--max-dist", type=float, default=3.0,
                    help="%% from the zone edge that still counts as NEAR (default 3)")
    ap.add_argument("--liq-only", action="store_true", help="only print names that clear LIQ")
    ap.add_argument("--sort", choices=["move", "liq", "zone"], default="move")

    # plumbing
    ap.add_argument("--days", type=int, default=5, help="days of bars for context (default 5)")
    ap.add_argument("--fast", action="store_true",
                    help="skip the 1m pass: halves IBKR requests, but post-close "
                         "volume then includes the closing auction")
    ap.add_argument("--max-symbols", type=int, default=25, help="cap bar requests (pacing)")
    ap.add_argument("--sleep", type=float, default=1.2, help="seconds between symbols")
    ap.add_argument("--delayed", action="store_true", help="use delayed market data")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7496, help="7496 live TWS, 7497 paper")
    ap.add_argument("--clientid", type=int, default=42)
    ap.add_argument("--csv", help="also write results to this path")

    args = ap.parse_args()

    ib = connect_ib(args.host, args.port, args.clientid)
    if args.delayed:
        ib.reqMarketDataType(3)

    try:
        label = "post-close" if args.scan == "ah" else "regular-session"
        print(f"Scanning {label} movers: >{args.min_ah_pct}% | >${args.min_price} | "
              f">{args.min_volume:,} day shares | >${args.min_mcap:.0f}M cap")

        sides = ["up", "down"] if args.side == "both" else [args.side]
        symbols, seen = [], set()
        for d in sides:
            hit = scan_movers(ib, d, args)
            print(f"  {d:<4}: {len(hit):>2} hits" + (f" -- {', '.join(hit)}" if hit else ""))
            for s in hit:
                if s not in seen:
                    seen.add(s)
                    symbols.append((s, d))

        if not symbols:
            print("\nScanner returned nothing. After 20:00 ET the after-hours codes go "
                  "stale -- try --scan day, or loosen --min-ah-pct.", file=sys.stderr)
            return

        symbols = symbols[: args.max_symbols]
        if len(symbols) == args.max_symbols:
            print(f"(measuring the first {args.max_symbols} -- raise --max-symbols to widen)")

        rows, skipped, illiquid = [], 0, 0
        for i, (sym, scan_side) in enumerate(symbols):
            print(f"  [{i+1}/{len(symbols)}] {sym}", end="\r", flush=True)
            minutes = pd.DataFrame() if args.fast else fetch_minute_bars(ib, sym)
            row = measure(sym, fetch_bars(ib, sym, args.days), minutes, scan_side, args)
            if row is None:
                skipped += 1
            else:
                row["hits"] = criteria_hits(row, args)
                if not args.liq_only or "LIQ" in row["hits"]:
                    rows.append(row)
                else:
                    illiquid += 1
            time.sleep(args.sleep)
        print(" " * 40, end="\r")

        render(rows, args, skipped, illiquid)

        if args.csv and rows:
            pd.DataFrame(rows).to_csv(args.csv, index=False)
            print(f"Written to {args.csv}")

    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
