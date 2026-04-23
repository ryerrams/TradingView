#!/usr/bin/env python3
"""
Signa API Test Suite
====================
Comprehensive test script for the Signa trading API (Early Adopter / Founding tier).
Covers all 20+ endpoints across Member, Pro, and Founding tiers.

Usage:
    export SIGNA_API_KEY="cmts_your_key_here"
    python signa_api_test.py

    # Run a single group:
    python signa_api_test.py --group signal

    # Run with a specific symbol:
    python signa_api_test.py --symbol NVDA
"""

import os
import sys
import json
import time
import uuid
import argparse
import requests
from datetime import date, timedelta
from typing import Any, Optional, List

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
BASE_URL   = "https://app.getsigna.ai"
API_V1     = f"{BASE_URL}/api/v1"
API_KEY    = os.environ.get("SIGNA_API_KEY", "")
TIMEOUT    = 20          # seconds per request
DEFAULT_SYM = "AAPL"    # change to your preferred test symbol

ANSI = {
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "bold":   "\033[1m",
    "reset":  "\033[0m",
}

def c(color: str, text: str) -> str:
    return f"{ANSI[color]}{text}{ANSI['reset']}"

# ──────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────
def auth_headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}

def get(endpoint: str, params: Optional[dict] = None, base: str = API_V1) -> tuple:
    url = f"{base}{endpoint}"
    try:
        r = requests.get(url, headers=auth_headers(), params=params, timeout=TIMEOUT)
        return r.status_code, _safe_json(r)
    except requests.exceptions.Timeout:
        return 0, {"error": "timeout"}
    except requests.exceptions.ConnectionError as e:
        return 0, {"error": str(e)}

def post(endpoint: str, body: dict, base: str = API_V1) -> tuple:
    url = f"{base}{endpoint}"
    try:
        r = requests.post(url, headers={**auth_headers(), "Content-Type": "application/json"},
                          json=body, timeout=TIMEOUT)
        return r.status_code, _safe_json(r)
    except requests.exceptions.Timeout:
        return 0, {"error": "timeout"}
    except requests.exceptions.ConnectionError as e:
        return 0, {"error": str(e)}

def _safe_json(r: requests.Response) -> Any:
    try:
        return r.json()
    except Exception:
        return {"raw": r.text[:500]}


# ──────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────
results: List[dict] = []

def run_test(name: str, tier: str, status_code: int, body: Any,
             expect_keys: Optional[List[str]] = None) -> bool:
    ok = status_code in (200, 201)
    if ok and expect_keys:
        # Walk nested keys like "data.direction"
        for key_path in expect_keys:
            parts = key_path.split(".")
            node = body
            for p in parts:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    ok = False
                    break

    icon = c("green", "✓ PASS") if ok else c("red", "✗ FAIL")
    tier_badge = c("cyan", f"[{tier}]")
    print(f"  {icon}  {tier_badge}  {name}  (HTTP {status_code})")

    if not ok:
        snippet = json.dumps(body, indent=2)[:400] if isinstance(body, dict) else str(body)[:400]
        print(c("yellow", f"       Response: {snippet}"))

    results.append({"name": name, "tier": tier, "ok": ok,
                    "status": status_code, "body": body})
    return ok


def skip_test(name: str, tier: str, reason: str):
    """Mark a test as skipped (requires external setup like Alpaca or Unusual Whales)."""
    print(f"  {c('yellow', '⚠ SKIP')}  {c('cyan', f'[{tier}]')}  {name}")
    print(c("yellow", f"       Reason: {reason}"))
    results.append({"name": name, "tier": tier, "ok": None,
                    "status": 0, "body": {"skipped": reason}})


# ──────────────────────────────────────────────
# ── GROUP 1: Health & Quotes (Member)
# ──────────────────────────────────────────────
def test_health(sym: str):
    print(c("bold", "\n── Health & Quotes (Member) ─────────────────────"))

    code, body = get("/health")
    run_test("GET /api/v1/health", "Member", code, body)

    code, body = get(f"/quote/{sym}")
    run_test(f"GET /api/v1/quote/{sym}", "Member", code, body,
             expect_keys=["symbol", "price"])


# ──────────────────────────────────────────────
# ── GROUP 2: Signals (Pro)
# ──────────────────────────────────────────────
def test_signals(sym: str):
    print(c("bold", "\n── Signal Engine (Pro) ──────────────────────────"))

    code, body = get("/signal", params={"sym": sym})
    run_test(f"GET /api/v1/signal?sym={sym}", "Pro", code, body,
             expect_keys=["ok", "data"])

    code, body = get("/analysis", params={"sym": sym})
    run_test(f"GET /api/v1/analysis?sym={sym}", "Pro", code, body)

    code, body = get("/signal-index")
    run_test("GET /api/v1/signal-index", "Pro", code, body)

    code, body = get(f"/history/{sym}")
    run_test(f"GET /api/v1/history/{sym}", "Pro", code, body)

    # Try /api/v1 first, fall back to /api
    code, body = get("/enhanced-signal", params={"sym": sym})
    if code == 404:
        code, body = get("/enhanced-signal", params={"sym": sym}, base=f"{BASE_URL}/api")
    run_test(f"GET /enhanced-signal?sym={sym}", "Pro", code, body)

    code, body = get("/agents", params={"sym": sym})
    run_test(f"GET /api/v1/agents?sym={sym}", "Pro", code, body)


# ──────────────────────────────────────────────
# ── GROUP 3: Scanner & NLQ (Pro)
# ──────────────────────────────────────────────
def test_scanner():
    print(c("bold", "\n── Scanner & Natural Language (Pro) ─────────────"))

    symbols = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL",
               "META", "BTC/USD", "ETH/USD", "SPY"]
    code, body = post("/scan", {"symbols": symbols})
    run_test(f"POST /api/v1/scan ({len(symbols)} symbols)", "Pro", code, body)

    code, body = post("/interpret", {"query": "best bullish setups in tech right now"})
    run_test("POST /api/v1/interpret (NLQ)", "Pro", code, body)


# ──────────────────────────────────────────────
# ── GROUP 4: Options Flow (Pro)
# ──────────────────────────────────────────────
def test_options_flow(sym: str):
    print(c("bold", "\n── Options Flow & Dark Pool (Pro) ───────────────"))
    NOTE = "Requires Unusual Whales connection in Signa app settings"

    code, body = get(f"/options-flow/{sym}", base=f"{BASE_URL}/api")
    if code == 401:
        skip_test(f"GET /api/options-flow/{sym}", "Pro", NOTE)
    else:
        run_test(f"GET /api/options-flow/{sym}", "Pro", code, body)

    code, body = get("/options-flow/dark-pool", base=f"{BASE_URL}/api")
    if code == 401:
        skip_test("GET /api/options-flow/dark-pool", "Pro", NOTE)
    else:
        run_test("GET /api/options-flow/dark-pool", "Pro", code, body)

    code, body = get("/options-flow/congress", base=f"{BASE_URL}/api")
    if code == 401:
        skip_test("GET /api/options-flow/congress", "Pro", NOTE)
    else:
        run_test("GET /api/options-flow/congress", "Pro", code, body)

    code, body = get("/options-flow/tide", base=f"{BASE_URL}/api")
    if code == 401:
        skip_test("GET /api/options-flow/tide", "Pro", NOTE)
    else:
        run_test("GET /api/options-flow/tide", "Pro", code, body)


# ──────────────────────────────────────────────
# ── GROUP 5: Edge Detection & Prediction Markets (Pro)
# ──────────────────────────────────────────────
def test_edge(sym: str):
    print(c("bold", "\n── Edge Detection & Prediction Markets (Pro) ────"))

    code, body = get("/edge-detection", params={"sym": sym}, base=f"{BASE_URL}/api")
    if code == 401:
        skip_test(f"GET /api/edge-detection?sym={sym}", "Pro",
                  "Requires Unusual Whales + prediction market connection")
    else:
        run_test(f"GET /api/edge-detection?sym={sym}", "Pro", code, body)

    code, body = get("/prediction-markets", params={"sym": sym}, base=f"{BASE_URL}/api")
    run_test(f"GET /api/prediction-markets?sym={sym}", "Pro", code, body)


# ──────────────────────────────────────────────
# ── GROUP 6: A2A Agent Routing (Pro)
# ──────────────────────────────────────────────
def test_a2a(sym: str):
    print(c("bold", "\n── A2A Agent Routing (Pro) ──────────────────────"))

    payload = {
        "requestId": str(uuid.uuid4()),
        "fromAgent": "signa-test-suite",
        "task": "signal_analysis",
        "symbols": [sym, "MSFT"],
        "context": {"timeframe": "1d"}
    }
    code, body = post("/a2a", payload)
    run_test("POST /api/v1/a2a (signal_analysis)", "Pro", code, body)


# ──────────────────────────────────────────────
# ── GROUP 7: Broker Integration (Pro)
# ──────────────────────────────────────────────
def test_broker():
    print(c("bold", "\n── Broker Integration (Pro) ─────────────────────"))
    NOTE = "Requires Alpaca account connected in Signa app settings"

    code, body = get("/broker/connections", base=f"{BASE_URL}/api")
    if code == 401:
        skip_test("GET /api/broker/connections", "Pro", NOTE)
    else:
        run_test("GET /api/broker/connections", "Pro", code, body)

    code, body = get("/broker/positions", base=f"{BASE_URL}/api")
    if code == 401:
        skip_test("GET /api/broker/positions", "Pro", NOTE)
    else:
        run_test("GET /api/broker/positions", "Pro", code, body)


# ──────────────────────────────────────────────
# ── GROUP 8: Backtesting (Founding)
# ──────────────────────────────────────────────
def test_backtest(sym: str):
    print(c("bold", "\n── Backtesting Engine (Founding) ────────────────"))

    end   = date.today()
    start = end - timedelta(days=90)
    payload = {
        "symbol": sym,
        "strategy": "momentum",
        "startDate": start.isoformat(),
        "endDate":   end.isoformat(),
        "initialCapital": 10000,
        "simulations": 500
    }
    code, body = post("/backtest", payload)
    run_test(f"POST /api/v1/backtest ({sym})", "Founding", code, body)


# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
def print_summary():
    total   = len(results)
    passed  = sum(1 for r in results if r["ok"] is True)
    skipped = sum(1 for r in results if r["ok"] is None)
    failed  = sum(1 for r in results if r["ok"] is False)

    print(c("bold", "\n══════════════════════════════════════════════════"))
    print(c("bold", "  SIGNA API TEST SUMMARY"))
    print(c("bold", "══════════════════════════════════════════════════"))
    print(f"  Total   : {total}")
    print(f"  {c('green',  'Passed')} : {passed}")
    print(f"  {c('yellow', 'Skipped')}: {skipped}  (require external connections)")
    print(f"  {c('red',    'Failed')} : {failed}")
    print()

    if failed:
        print(c("yellow", "  Failed tests:"))
        for r in results:
            if r["ok"] is False:
                print(c("red", f"    ✗ {r['name']}  (HTTP {r['status']})"))

    print()
    return failed == 0


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Signa API Test Suite")
    parser.add_argument("--symbol", default=DEFAULT_SYM,
                        help=f"Symbol to test (default: {DEFAULT_SYM})")
    parser.add_argument("--group", choices=[
        "health", "signal", "scanner", "options",
        "edge", "a2a", "broker", "backtest", "all"
    ], default="all", help="Which group of tests to run (default: all)")
    args = parser.parse_args()

    if not API_KEY:
        print(c("red", "\n[ERROR] SIGNA_API_KEY environment variable not set."))
        print(c("yellow", "        export SIGNA_API_KEY='cmts_your_key_here'\n"))
        sys.exit(1)

    sym = args.symbol.upper()

    print(c("bold", "══════════════════════════════════════════════════"))
    print(c("bold", "  SIGNA API TEST SUITE"))
    print(c("bold", "══════════════════════════════════════════════════"))
    print(f"  Base URL : {BASE_URL}")
    print(f"  Symbol   : {sym}")
    print(f"  API Key  : {API_KEY[:8]}{'*' * (len(API_KEY) - 8)}")
    print(f"  Group    : {args.group}")

    t0 = time.time()

    g = args.group
    if g in ("health", "all"):  test_health(sym)
    if g in ("signal", "all"):  test_signals(sym)
    if g in ("scanner", "all"): test_scanner()
    if g in ("options", "all"): test_options_flow(sym)
    if g in ("edge", "all"):    test_edge(sym)
    if g in ("a2a", "all"):     test_a2a(sym)
    if g in ("broker", "all"):  test_broker()
    if g in ("backtest", "all"): test_backtest(sym)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    all_passed = print_summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
