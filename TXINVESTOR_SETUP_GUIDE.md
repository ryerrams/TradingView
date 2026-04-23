# TXInvestor Dashboard — TradingView Setup Guide

## Overview

This package recreates the TXInvestor dashboard using two custom Pine Script indicators
plus a set of TradingView built-in indicators layered on top.

---

## Files Included

| File | Purpose |
|---|---|
| `txinvestor_main.pine` | Main overlay: WMA30, stage background, signal markers, left-panel dashboard table |
| `txinvestor_rs.pine` | Separate pane: RS Rating line (Mansfield RS, normalized 1–99) |

---

## Step 1 — Add the Custom Pine Script Indicators

### txinvestor_main.pine (Overlay)

1. Open TradingView → Pine Editor (bottom toolbar)
2. Delete any default code, paste the contents of `txinvestor_main.pine`
3. Click **Save** → give it a name (e.g. "TXInvestor Main")
4. Click **Add to chart** — it will appear as an overlay on the price chart

**What you get:**
- Red WMA30 line (Weinstein stage line)
- Blue EMA20, Orange SMA50, Purple SMA200
- Colored stage background (yellow=S1, green=S2, orange=S3, red=S4)
- Triangle markers for stage transitions and B3 breakout signals
- Full left-panel dashboard table with Daily / Weekly columns

### txinvestor_rs.pine (Separate Pane)

1. In Pine Editor, open a new tab
2. Paste the contents of `txinvestor_rs.pine`
3. Click **Save** → name it "TXInvestor RS Rating"
4. Click **Add to chart** — it creates a new pane below the price chart

**What you get:**
- RS Rating line (1–99 scale, green when strong, red when weak)
- Green/red background zones above 80 / below 40
- Label showing current RS Rating value

---

## Step 2 — Add TradingView Built-in Indicators

Layer these on top of the custom scripts for the full TXInvestor experience:

### Required Built-ins (search these in the Indicators panel)

| Built-in Indicator | Purpose in Dashboard | Notes |
|---|---|---|
| **Technical Ratings** | "Tech Rating", "Oscillator Rating", "MA Rating" rows | Change display to "Table" mode in settings |
| **Analyst Ratings** | "Analyst Rating" row | Shows consensus Buy/Hold/Sell |
| **Volume** | Volume bars at bottom of price pane | Color by up/down bar |
| **MACD** | Momentum oscillator sub-panel | Set 12/26/9; shows the bottom momentum bars |

### Optional But Recommended

| Built-in Indicator | Purpose |
|---|---|
| **Average True Range (ATR)** | Cross-reference ATR extension values |
| **Relative Strength Index (RSI)** | Quick oscillator check |
| **Chaikin Money Flow (CMF)** | Cross-reference Accumulation grade |
| **On Balance Volume (OBV)** | Secondary confirmation of accumulation |

---

## Step 3 — Configure the Layout

Match the screenshot layout:

```
┌─────────────────────────────────────────┐
│  [Dashboard Table]  │  Price Candles    │
│  (top_left)         │  + WMA30 overlay  │
│                     │  + EMA20, SMA50   │
│                     │  + Stage BG color │
│                     │  + Signal markers │
├─────────────────────────────────────────┤
│  RS Rating Panel (txinvestor_rs)        │
├─────────────────────────────────────────┤
│  MACD / Volume (built-in)               │
└─────────────────────────────────────────┘
```

### Recommended Chart Settings
- Chart type: **Candles** (default)
- Timeframe: **1D** (daily) for primary analysis; script auto-pulls weekly data
- Scale: Right-side price scale

---

## Step 4 — Understanding Each Metric

### Stage Analysis (Weinstein Method)
| Stage | Condition | Action |
|---|---|---|
| S1 | Price near WMA30, slope flattening after decline | Watch for base |
| S2 | Price > WMA30, WMA30 rising | Buy zone |
| S3 | Price near WMA30, slope flattening after rise | Trim/sell |
| S4 | Price < WMA30, WMA30 declining | Avoid |

### Key Indicators Explained

**ATR Extension (Dist from MA):** How many 14-day ATRs the price is above/below the WMA30.
- < 1.5x = Normal range
- 1.5–3x = Extended (caution on new buys)
- > 3x = Highly extended (very high risk entry)

**PowerTrend:** Fast EMA (10) > Medium EMA (20) > Slow SMA (50), all rising simultaneously.
This indicates strong multi-period upward momentum.

**Vol Ratio:** Current bar volume vs 50-bar average.
- > 1.5x = Above-average (confirming)
- > 2.0x = Strong confirmation
- < 1.0x = Below-average (weak signal)

**ML Score (0–5):** Composite score:
- +1 if Stage 2
- +1 if Volume ratio > 1.2x
- +1 if full MA Alignment
- +1 if RSI > 50
- +1 if Mansfield RS > 0

**B3 Score:** Breakout scoring system:
- +3 if price is within 2% of 52-week high
- +3 if volume > 2x average
- +2 if Stage 2
- +2 if full MA Alignment
- Score >= 7 = B3 Signal triggered

**Accumulation (A/B/C/D):** Based on Chaikin Money Flow over 20 bars:
- A = CMF > 0.05 (strong accumulation)
- B = CMF 0–0.05 (mild accumulation)
- C = CMF -0.05 to 0 (mild distribution)
- D = CMF < -0.05 (strong distribution)

**TI65:** 65-bar relative performance ratio vs benchmark. > 1.0 = outperforming.

**MDT:** Momentum Derivative — acceleration of 20-bar price momentum. > 1.0 = accelerating.

**Mansfield RS:** Relative performance vs SPY over ~1 year.
- Positive = outperforming benchmark
- Negative = underperforming benchmark

---

## Step 5 — Alerts

The main script includes these pre-built alert conditions:
- **S2 Entry** — stock crosses into Stage 2
- **S4 Warning** — stock crosses into Stage 4 (avoid)
- **B3 Signal** — B3 breakout score crosses threshold
- **A+ Setup** — all A+ conditions align simultaneously
- **High-Vol S2** — Stage 2 with volume > 2x and RSI > 60

To create alerts: Right-click chart → Add Alert → select the condition.

---

## What Cannot Be Replicated in Pine Script

| Feature | Reason | Workaround |
|---|---|---|
| **Analyst Rating** | Requires analyst consensus data not available via Pine | Add TradingView's built-in "Analyst Ratings" indicator |
| **IBD RS Rating (exact)** | Requires ranking against 8,000+ stocks simultaneously | The RS script approximates it using historical percentile ranking |
| **Market Signal (top bar)** | Requires market-wide breadth data | Add "Market Breadth" or "% Above MA" indicators separately |
| **VIX display** | Can only plot VIX as overlay/pane, not in main table | Add VIX as a separate symbol in the watchlist or overlay |

---

## Troubleshooting

**"Too many securities" error:** Pine Script allows up to 40 `request.security()` calls.
If you hit this limit, disable the weekly calculations in settings or split into two scripts.

**Table not visible:** Check that "Show Dashboard Table" is enabled in indicator settings.
Also check the table position setting — move to "top_right" if another indicator is overlapping.

**RS Rating shows flat line:** Needs at least `rsLen` bars of history (default 252 = ~1 year).
Switch to a longer timeframe or reduce the RS Lookback in settings.

**Stage colors look wrong:** The slope lookback (default 5 bars) controls sensitivity.
Increase to 10-15 for less noise on daily charts; decrease for more sensitivity.
