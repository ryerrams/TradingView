#!/usr/bin/env python3
"""
Signa API Inspector
-------------------
Fetches the full raw response from any endpoint and prints it + saves to JSON.

Usage:
    export $(cat .env)
    python3 signa_inspect.py                        # signal for AAPL
    python3 signa_inspect.py --sym NVDA             # signal for NVDA
    python3 signa_inspect.py --endpoint analysis    # analysis endpoint
    python3 signa_inspect.py --endpoint agents
    python3 signa_inspect.py --endpoint quote
"""

import os, sys, json, argparse, requests
from datetime import datetime

API_KEY  = os.environ.get("SIGNA_API_KEY", "")
BASE     = "https://app.getsigna.ai"

ENDPOINTS = {
    "signal":          ("GET",  f"{BASE}/api/v1/signal",        "sym"),
    "analysis":        ("GET",  f"{BASE}/api/v1/analysis",      "sym"),
    "quote":           ("GET",  f"{BASE}/api/v1/quote/{{sym}}", None),
    "history":         ("GET",  f"{BASE}/api/v1/history/{{sym}}", None),
    "signal-index":    ("GET",  f"{BASE}/api/v1/signal-index",  None),
    "enhanced-signal": ("GET",  f"{BASE}/api/v1/enhanced-signal","sym"),
    "agents":          ("GET",  f"{BASE}/api/v1/agents",        "sym"),
    "prediction":      ("GET",  f"{BASE}/api/prediction-markets","sym"),
}

def fetch(endpoint, sym):
    method, url_tmpl, param_key = ENDPOINTS[endpoint]
    url = url_tmpl.replace("{sym}", sym)
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params  = {param_key: sym} if param_key else None
    r = requests.get(url, headers=headers, params=params, timeout=20)
    return r.status_code, r.json() if r.headers.get("content-type","").startswith("application/json") else {"raw": r.text}

def pretty(data):
    return json.dumps(data, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym",      default="AAPL")
    parser.add_argument("--endpoint", default="signal", choices=list(ENDPOINTS))
    parser.add_argument("--save",     action="store_true", help="Save to JSON file")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: SIGNA_API_KEY not set"); sys.exit(1)

    sym = args.sym.upper()
    print(f"\n{'='*60}")
    print(f"  Signa API Inspector")
    print(f"  Endpoint : {args.endpoint}")
    print(f"  Symbol   : {sym}")
    print(f"{'='*60}\n")

    code, data = fetch(args.endpoint, sym)
    print(f"HTTP {code}\n")
    print(pretty(data))

    if args.save:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"signa_{args.endpoint}_{sym}_{ts}.json"
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n✓ Saved to {fname}")

if __name__ == "__main__":
    main()
