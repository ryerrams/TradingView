"""Run the SOMA backtest across NQ (MBot IBKR cache) and a crypto basket (Binance).

    python3 -m backtest.soma.run                 # full run
    python3 -m backtest.soma.run --quick         # NQ 30m + crypto 4h only

Writes backtest/results/{trades.csv, continuation.csv, summary.csv, summary.md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import data, engine, sim

RESULTS = Path(__file__).resolve().parent.parent / "results"
CRYPTO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "NEARUSDT",
          "AVAXUSDT", "LINKUSDT", "SUIUSDT", "PUMPUSDT"]
CRYPTO_TF = {"1h": "2024-01-01", "4h": "2022-01-01", "1d": "2019-01-01"}
NQ_TF = ["5min", "15min", "30min"]
WARMUP_BARS = 250
CRYPTO_COST = 0.001          # 0.10% round trip (0.05% per side)
NQ_COST_PTS = 0.5            # 2 ticks round trip (slippage + commission)
MODES = {"TrendRun": engine.Mode(scalp=False), "Scalp": engine.Mode(scalp=True)}


def one(df, market, tf, symbol, tick, cost, frames, cont_frames):
    tf_sec = data.TF_SECONDS[tf]
    intraday = tf_sec < engine.TF_D
    resets = [True, False] if (market == "crypto" and intraday) else [True]
    for mname, mode in MODES.items():
        sig = engine.run_engine(df, market, tf_sec, tick, mode)
        attrs = dict(sig.attrs)
        sig = sig[sig.bar >= WARMUP_BARS].copy()
        sig.attrs.update(attrs)
        for entry_mode in ("next_open", "limit"):
            for reset in resets:
                tr = sim.simulate(df, sig, intraday, cost, entry_mode=entry_mode, session_reset=reset)
                if len(tr):
                    tr.insert(0, "session_reset", reset)
                    tr.insert(0, "entry_mode", entry_mode)
                    tr.insert(0, "mode", mname)
                    tr.insert(0, "tf", tf)
                    tr.insert(0, "symbol", symbol)
                    tr.insert(0, "market", market)
                    frames.append(tr)
        if mname == "TrendRun":            # continuation logic is mode-independent
            cs = sim.continuation_stats(df, sig)
            if len(cs):
                cs.insert(0, "tf", tf)
                cs.insert(0, "symbol", symbol)
                cs.insert(0, "market", market)
                cont_frames.append(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    nq_tfs = ["30min"] if args.quick else NQ_TF
    crypto_tfs = {"4h": CRYPTO_TF["4h"]} if args.quick else CRYPTO_TF

    frames, cont = [], []
    for tf in nq_tfs:
        for contract, df in data.nq_segments(tf):
            print(f"NQ {contract} {tf}: {len(df)} bars", flush=True)
            one(df, "cme", tf, f"NQ:{contract}", data.NQ_TICK, lambda px: NQ_COST_PTS, frames, cont)
    for sym in CRYPTO:
        tick = data.binance_tick(sym)
        for tf, start in crypto_tfs.items():
            df = data.binance_klines(sym, tf, start)
            print(f"{sym} {tf}: {len(df)} bars since {df.index[0].date()}", flush=True)
            one(df, "crypto", tf, sym, tick, lambda px: px * CRYPTO_COST, frames, cont)

    RESULTS.mkdir(parents=True, exist_ok=True)
    trades = pd.concat(frames, ignore_index=True)
    trades.to_csv(RESULTS / "trades.csv", index=False)
    conts = pd.concat(cont, ignore_index=True) if cont else pd.DataFrame()
    conts.to_csv(RESULTS / "continuation.csv", index=False)

    trades["asset"] = trades["symbol"].where(trades.market == "crypto", "NQ")
    rows = []
    grp = ["tf", "mode", "engine", "entry_mode", "session_reset"]
    for keys, g in trades.groupby(["market"] + grp):
        rows.append(dict(zip(["market"] + grp, keys), **sim.summarize(g)))
    for keys, g in trades.groupby(["asset"] + grp):
        rows.append(dict(zip(["asset"] + grp, keys), **sim.summarize(g)))
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS / "summary.csv", index=False)
    print(summary[summary.market.notna()].drop(columns=["asset"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    if len(conts):
        cst = conts.groupby(["market", "tf"]).result.value_counts(normalize=True).unstack().round(2)
        print("\nContinuation: +1 ATR before -1 ATR within 20 bars\n", cst.to_string())


if __name__ == "__main__":
    main()
