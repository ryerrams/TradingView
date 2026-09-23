#!/usr/bin/env python3
"""
Replay a session's volume profile building up over time, as a video.

Volume is bucketed into 30 minute slots. Each slot is colored by the 2 hour
block it belongs to, using the NinjaTrader palette, and shaded from light to
dark across the four 30 minute slots inside that block. So you see the 2 hour
structure at a glance and the 30 minute structure inside it.

    python vp_replay.py --csv session.csv --out replay.mp4
    python vp_replay.py --demo --out demo.mp4          # synthetic session

SPEED
    --frame-minutes  market minutes advanced per frame  (default 5)
    --fps            frames per second of video         (default 20)
    Playback speed is frame-minutes x fps market-minutes per real second.
    Defaults give 100x, so a 23 hour session runs about 14 seconds.

Requires: pip install pandas numpy matplotlib   (plus ffmpeg on PATH for mp4)
"""
import argparse
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

CT = ZoneInfo("America/Chicago")
DELTA_METHOD = ["range"]        # set from argv before build()

# 2 hour blocks, keyed by block index. Block 0 starts at 08:00.
PALETTE = [
    ("08-10", "#00CED1"), ("10-12", "#DAA520"), ("12-14", "#3CB371"),
    ("14-16", "#7CFC00"), ("16-18", "#FF0000"), ("18-20", "#0000FF"),
    ("20-22", "#FF00FF"), ("22-00", "#FF8C00"), ("00-02", "#808080"),
    ("02-04", "#CD853F"), ("04-06", "#008080"), ("06-08", "#800000"),
]
SHADES = (1.00, 0.82, 0.64, 0.48)      # lightness across the four 30m slots


def block_of(hour: int) -> int:
    return ((hour - 8) % 24) // 2


def slot_in_block(hour: int, minute: int) -> int:
    return ((hour - 8) % 2) * 2 + (minute // 30)


def shade(hex_color: str, factor: float):
    r, g, b = to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def value_area(vols, edges, frac):
    """Contiguous band around the POC holding `frac` of the volume. Expands one
    row at a time toward whichever neighbour holds more, which is the standard
    value-area construction. Returns (bottom, top, poc_row)."""
    total = vols.sum()
    if total <= 0:
        return None
    poc = int(vols.argmax())
    lo = hi = poc
    covered = vols[poc]
    n = len(vols)
    while covered < frac * total and (lo > 0 or hi < n - 1):
        below = vols[lo - 1] if lo > 0 else -1.0
        above = vols[hi + 1] if hi < n - 1 else -1.0
        if above >= below:
            hi += 1
            covered += vols[hi]
        else:
            lo -= 1
            covered += vols[lo]
    return edges[lo], edges[hi + 1], poc


def atr_wilder(df, n=14):
    h, l, c = df["high"].to_numpy(float), df["low"].to_numpy(float), df["close"].to_numpy(float)
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr = np.empty_like(tr)
    atr[0] = tr[0]
    a = 1.0 / n
    for i in range(1, len(tr)):                      # Wilder smoothing
        atr[i] = atr[i - 1] + a * (tr[i] - atr[i - 1])
    return atr


def confluence_signals(df, g, rows, args):
    """Port of the Daily/Hourly Tail Confluence study.

    Flags when the lows of the last few hourly profiles cluster within a
    tolerance of the running day low, mirrored with highs against the day high.
    Everything is computed forward bar by bar, so nothing looks ahead. The
    trailing stop and target cascade from the original are deliberately absent.
    """
    atr = atr_wilder(df, 14)
    t = df["time"]
    lo_a, hi_a, cl_a = (df["low"].to_numpy(float), df["high"].to_numpy(float),
                        df["close"].to_numpy(float))
    INF = float("inf")
    run_lo, run_hi = INF, -INF
    hour_lo, hour_hi = INF, -INF
    hL = [np.nan] * 3                                 # last 3 completed hours
    hH = [np.nan] * 3
    prev_hour = None
    totals = np.zeros(rows)
    prev_raw = {"LONG": False, "SHORT": False}
    fired = {"LONG": False, "SHORT": False}
    out, cnt_lo, cnt_hi, tol_s = [], np.zeros(len(df), int), np.zeros(len(df), int), np.zeros(len(df))

    for i in range(len(df)):
        key = (t.iloc[i].date(), t.iloc[i].hour)
        if prev_hour is not None and key != prev_hour:
            hL = [hour_lo] + hL[:2]
            hH = [hour_hi] + hH[:2]
            hour_lo, hour_hi = INF, -INF
        prev_hour = key
        hour_lo, hour_hi = min(hour_lo, lo_a[i]), max(hour_hi, hi_a[i])
        run_lo, run_hi = min(run_lo, lo_a[i]), max(run_hi, hi_a[i])
        totals[g["j_lo"][i]:g["j_hi"][i] + 1] += g["per"][i]

        tol = (args.conf_atr_mult * atr[i] if args.conf_tolerance_mode == "atr"
               else args.conf_tolerance_points)
        tol_s[i] = tol

        nl = int(args.conf_include_live and hour_lo < INF and abs(hour_lo - run_lo) <= tol)
        nh = int(args.conf_include_live and hour_hi > -INF and abs(hour_hi - run_hi) <= tol)
        for k in range(3):
            if args.conf_lookback >= k + 2:
                nl += int(not np.isnan(hL[k]) and abs(hL[k] - run_lo) <= tol)
                nh += int(not np.isnan(hH[k]) and abs(hH[k] - run_hi) <= tol)
        cnt_lo[i], cnt_hi[i] = nl, nh

        raw = {"LONG": nl >= args.conf_min_hours and (not args.conf_confirm or cl_a[i] > run_lo),
               "SHORT": nh >= args.conf_min_hours and (not args.conf_confirm or cl_a[i] < run_hi)}
        for sd in ("LONG", "SHORT"):
            edge = raw[sd] and not prev_raw[sd]
            if edge and not (args.conf_once and fired[sd]):
                fired[sd] = True
                poc = g["centers"][int(totals.argmax())] if totals.max() > 0 else np.nan
                out.append((t.iloc[i], float(cl_a[i]), sd,
                            nl if sd == "LONG" else nh, float(poc)))
        prev_raw = raw
    return out, cnt_lo, cnt_hi, tol_s


def thin_tail(vols, row, poc_v, thin_frac, tail_rows, upward):
    """Is `row` sitting in a thin pocket at the running extreme?

    Thin means the row holds less than `thin_frac` of the POC row. The extreme
    is the highest or lowest row that has traded so far, so this only ever looks
    at volume already accumulated."""
    traded = np.flatnonzero(vols > 0)
    if traded.size == 0:
        return False
    edge = traded.max() if upward else traded.min()
    near = row >= edge - tail_rows if upward else row <= edge + tail_rows
    return bool(near and vols[row] < thin_frac * poc_v)


def demo_session(seed=7):
    """Synthetic 23 hour session so the renderer can be exercised without IB."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-08-27 17:00", tz=CT)
    idx = pd.date_range(start, periods=23 * 60, freq="1min")
    price = 29600.0
    rows = []
    for t in idx:
        # thin overnight, heavy on the 08:30 CT cash open
        h = t.hour + t.minute / 60
        busy = 4.0 if 8.4 <= h < 10.0 else (1.8 if 7 <= h < 8.4 else 1.0)
        drift = rng.normal(0, 1.6) * busy ** 0.4
        o = price
        c = o + drift
        hi = max(o, c) + abs(rng.normal(0, 0.9))
        lo = min(o, c) - abs(rng.normal(0, 0.9))
        rows.append({"time": t, "open": o, "high": hi, "low": lo, "close": c,
                     "volume": max(1.0, rng.normal(300 * busy, 90 * busy))})
        price = c
    return pd.DataFrame(rows)


def load(args):
    if args.demo:
        return demo_session()
    df = pd.read_csv(args.csv)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(CT)
    need = {"time", "open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"csv is missing columns: {sorted(missing)}")
    return df.sort_values("time").reset_index(drop=True)


def split_delta(df, method):
    """Per-bar buy and sell volume.

    'columns'   real data, if the csv carries buy_volume / sell_volume
    'range'     volume weighted by where the bar closed in its own range.
                A close on the high counts fully as buying. The usual proxy.
    'direction' whole bar counts as buying if it closed up, selling if down.
    """
    vol = df["volume"].to_numpy(float)
    if method == "columns":
        return df["buy_volume"].to_numpy(float), df["sell_volume"].to_numpy(float)
    if method == "direction":
        up = (df["close"] >= df["open"]).to_numpy()
        return np.where(up, vol, 0.0), np.where(up, 0.0, vol)
    hi, lo = df["high"].to_numpy(float), df["low"].to_numpy(float)
    cl, op = df["close"].to_numpy(float), df["open"].to_numpy(float)
    span = hi - lo
    frac = np.where(span > 0, (cl - lo) / np.where(span > 0, span, 1.0),
                    np.where(cl >= op, 1.0, 0.0))
    return vol * frac, vol * (1.0 - frac)


def build(df, rows):
    """Bin every bar into (price row, 30 minute slot). Returns the pieces the
    animation needs. Bins are fixed up front from the whole session so the
    y axis never shifts mid-replay."""
    lo, hi = df["low"].min(), df["high"].max()
    edges = np.linspace(lo, hi, rows + 1)
    height = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2

    t0 = df["time"].iloc[0]
    slot = ((df["time"] - t0).dt.total_seconds() // 1800).astype(int).to_numpy()
    n_slots = int(slot.max()) + 1

    colors, slot_block = [], []
    for s in range(n_slots):
        t = t0 + pd.Timedelta(minutes=30 * s)
        b = block_of(t.hour)
        slot_block.append(b)
        colors.append(shade(PALETTE[b][1], SHADES[slot_in_block(t.hour, t.minute)]))
    slot_block = np.array(slot_block)

    # spread each bar's volume evenly over the rows its range covers
    j_lo = np.clip(np.searchsorted(edges, df["low"].to_numpy(), "right") - 1, 0, rows - 1)
    j_hi = np.clip(np.searchsorted(edges, df["high"].to_numpy(), "right") - 1, 0, rows - 1)
    spanned = (j_hi - j_lo + 1)
    per = df["volume"].to_numpy() / spanned
    buy, sell = split_delta(df, DELTA_METHOD[0])
    per_buy, per_sell = buy / spanned, sell / spanned

    return dict(edges=edges, height=height, centers=centers, lo=lo, hi=hi,
                slot=slot, n_slots=n_slots, colors=colors, slot_block=slot_block,
                j_lo=j_lo, j_hi=j_hi, per=per, t0=t0,
                per_buy=per_buy, per_sell=per_sell)


def final_matrix(df, g, rows):
    m = np.zeros((rows, g["n_slots"]))
    for i in range(len(df)):
        m[g["j_lo"][i]:g["j_hi"][i] + 1, g["slot"][i]] += g["per"][i]
    return m


def pick_writer(args):
    """mp4 needs an ffmpeg binary. Look on PATH, then fall back to the one
    imageio-ffmpeg bundles, then to an animated gif, rather than failing after
    the whole render has already been computed."""
    if args.out.endswith(".gif"):
        return animation.PillowWriter(fps=args.fps)

    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"ffmpeg not on PATH, using the imageio-ffmpeg binary")
        except Exception:
            pass
    if exe is None:
        args.out = args.out.rsplit(".", 1)[0] + ".gif"
        print("no ffmpeg found. Install it with:  sudo apt install -y ffmpeg\n"
              "  (or: pip install imageio-ffmpeg)\n"
              f"falling back to {args.out}")
        return animation.PillowWriter(fps=args.fps)

    matplotlib.rcParams["animation.ffmpeg_path"] = exe
    return animation.FFMpegWriter(fps=args.fps, bitrate=args.bitrate, codec="libx264",
                                  extra_args=["-pix_fmt", "yuv420p"])


def render(args):
    df = load(args)
    has_cols = {"buy_volume", "sell_volume"}.issubset(df.columns)
    if args.delta_method == "auto":
        DELTA_METHOD[0] = "columns" if has_cols else "range"
    else:
        DELTA_METHOD[0] = args.delta_method
    if DELTA_METHOD[0] == "columns" and not has_cols:
        sys.exit("--delta-method columns needs buy_volume and sell_volume in the csv")
    print(f"delta from: {DELTA_METHOD[0]}"
          + ("" if DELTA_METHOD[0] == "columns" else "  (estimated, not true tape delta)"))
    rows = args.rows
    g = build(df, rows)
    total = final_matrix(df, g, rows)
    x_max = total.sum(axis=1).max() * 1.06
    fb = np.zeros(rows); fs = np.zeros(rows)
    for i in range(len(df)):
        fb[g["j_lo"][i]:g["j_hi"][i] + 1] += g["per_buy"][i]
        fs[g["j_lo"][i]:g["j_hi"][i] + 1] += g["per_sell"][i]
    d_max = max(np.abs(fb - fs).max() * 1.10, 1.0)

    # step by wall-clock minutes, whatever the bar size actually is
    bar_min = float(pd.Series(df["time"]).diff().dt.total_seconds().median() / 60) or 1.0
    bars_per_frame = max(1, int(round(args.frame_minutes / bar_min)))
    n_frames = int(np.ceil(len(df) / bars_per_frame)) + args.hold_frames
    speed = args.frame_minutes * args.fps
    print(f"{len(df)} bars of {bar_min:g} min, {rows} price rows, "
          f"{g['n_slots']} half-hour slots, {bars_per_frame} bars/frame")
    print(f"{n_frames} frames at {args.fps} fps -> {n_frames / args.fps:.1f}s of video, "
          f"{speed}x realtime")

    plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
                         "text.color": "#c9d1d9", "axes.labelcolor": "#c9d1d9",
                         "xtick.color": "#8b949e", "ytick.color": "#8b949e",
                         "axes.edgecolor": "#30363d", "font.size": 10})
    fig, (axp, axd, axv) = plt.subplots(
        1, 3, figsize=(args.width / 100, args.height / 100), dpi=100,
        gridspec_kw={"width_ratios": [2.4, 0.6, 1.1], "wspace": 0.03}, sharey=True)

    t = df["time"]
    if args.ghost:
        axp.plot(t, df["close"], color="#21262d", lw=0.8, zorder=1)
    revealed, = axp.plot([], [], color="#e6edf3", lw=1.0, zorder=3)
    dot, = axp.plot([], [], "o", color="#f0b429", ms=5, zorder=4)
    axp.set_xlim(t.iloc[0], t.iloc[-1])
    axp.set_ylim(g["lo"] - g["height"], g["hi"] + g["height"])
    axp.grid(alpha=0.12, lw=0.5)
    axp.set_title("price", loc="left", color="#8b949e")

    seg = PolyCollection([], linewidths=0)
    axv.add_collection(seg)
    poc = axv.axhline(np.nan, color="#f0e442", lw=1.2, ls="--", alpha=0.9, zorder=5)
    axv.set_xlim(0, x_max)
    axv.grid(alpha=0.12, lw=0.5, axis="x")
    axv.set_title("volume profile, stacked oldest first", loc="left", color="#8b949e")
    axv.tick_params(labelleft=False)

    # delta profile, sitting between price and the volume profile
    dbars = axd.barh(g["centers"], np.zeros(rows), height=g["height"] * 0.88,
                     align="center", linewidth=0)
    axd.axvline(0, color="#30363d", lw=0.8, zorder=1)
    axd.set_xlim(-d_max, d_max)
    axd.grid(alpha=0.12, lw=0.5, axis="x")
    axd.set_title("delta", loc="left", color="#8b949e")
    axd.tick_params(labelleft=False, labelsize=8)
    axd.set_xticks([-d_max * 0.6, 0, d_max * 0.6])

    # band marking where the current 2 hour block's volume actually sits
    band = {}
    for key in ("top", "bot"):
        band[f"p_{key}"] = axp.axhline(np.nan, lw=1.3, alpha=0.95, zorder=6)
        band[f"v_{key}"] = axv.axhline(np.nan, lw=1.3, alpha=0.95, zorder=6)
        band[f"d_{key}"] = axd.axhline(np.nan, lw=1.3, alpha=0.95, zorder=6)
        band[f"t_{key}"] = axv.text(0, 0, "", fontsize=9, ha="right", family="monospace",
                                    zorder=7, visible=False)
    ghosts = []
    if args.ghost_blocks:
        for _ in range(12):
            ghosts.append((axp.axhline(np.nan, lw=0.7, ls=":", alpha=0.35, zorder=2),
                           axp.axhline(np.nan, lw=0.7, ls=":", alpha=0.35, zorder=2)))
    done_blocks = {}

    clock = fig.text(0.012, 0.965, "", fontsize=13, color="#e6edf3", family="monospace")
    stats = fig.text(0.012, 0.935, "", fontsize=10, color="#8b949e", family="monospace")
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in PALETTE],
               loc="lower center", ncol=12, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.10)

    # overnight is everything before the RTH open. Fully in the past by the time
    # any signal is evaluated, so using it is not lookahead.
    rth_h, rth_m = int(args.rth_start[:2]), int(args.rth_start[2:])
    rth_open = None
    for ts in t:
        if (ts.hour, ts.minute) >= (rth_h, rth_m) and ts.date() != g["t0"].date():
            rth_open = ts
            break
    if rth_open is None:
        rth_open = t.iloc[-1]
    on = df[t < rth_open]
    on_hi, on_lo = (on["high"].max(), on["low"].min()) if len(on) else (np.nan, np.nan)
    on_range = max(on_hi - on_lo, 1e-9)
    on_vols = np.zeros(rows)
    for i in range(len(on)):
        on_vols[g["j_lo"][i]:g["j_hi"][i] + 1] += g["per"][i]
    on_va = value_area(on_vols, g["edges"], args.block_va)
    print(f"RTH open {rth_open:%a %H:%M}, overnight range {on_lo:.2f} to {on_hi:.2f}"
          + (f", value area {on_va[0]:.2f} to {on_va[1]:.2f}" if on_va else ""))

    if on_va and args.signals:
        for y, lab in ((on_va[1], "ON VAH"), (on_va[0], "ON VAL")):
            axp.axhline(y, color="#7d8590", lw=0.8, ls="--", alpha=0.6, zorder=2)
            axp.text(t.iloc[0], y, f" {lab}", color="#7d8590", fontsize=8,
                     va="bottom", family="monospace")

    conf, cnt_lo, cnt_hi, tol_s = ([], None, None, None)
    if args.signals and args.signal_mode in ("confluence", "both"):
        conf, cnt_lo, cnt_hi, tol_s = confluence_signals(df, g, rows, args)
        mode = ("ATR x %.2f" % args.conf_atr_mult if args.conf_tolerance_mode == "atr"
                else "%.2f pts" % args.conf_tolerance_points)
        print(f"confluence: tolerance {mode}, need {args.conf_min_hours} of "
              f"{args.conf_lookback} hours -> {len(conf)} signal(s)")
        for c_ts, c_px, c_sd, c_n, c_poc in conf:
            print(f"  {c_sd:5s} confluence {c_ts:%a %H:%M} @ {c_px:,.2f}  {c_n} hourly "
                  f"tails near the day {'low' if c_sd == 'LONG' else 'high'}, "
                  f"POC {c_poc:,.2f}")
    conf_left = list(conf)

    win = max(1, int(round(args.velocity_window / bar_min)))
    fired = []                       # (time, price, side)
    marks = []

    acc = np.zeros((rows, g["n_slots"]))
    accb = np.zeros(rows)
    accs = np.zeros(rows)
    state = {"i": 0}
    GREEN, RED = "#26a69a", "#ef5350"

    def draw(frame):
        end = min(len(df), (frame + 1) * bars_per_frame)
        for i in range(state["i"], end):
            lo_i, hi_i = g["j_lo"][i], g["j_hi"][i] + 1
            acc[lo_i:hi_i, g["slot"][i]] += g["per"][i]
            accb[lo_i:hi_i] += g["per_buy"][i]
            accs[lo_i:hi_i] += g["per_sell"][i]
        state["i"] = end
        cut = max(0, end - 1)

        verts, cols = [], []
        active = np.flatnonzero(acc.sum(axis=0) > 0)
        for r in range(rows):
            x = 0.0
            y0, y1 = g["edges"][r], g["edges"][r + 1] - g["height"] * 0.12
            for s in active:
                wgt = acc[r, s]
                if wgt > 0:
                    verts.append([(x, y0), (x + wgt, y0), (x + wgt, y1), (x, y1)])
                    cols.append(g["colors"][s])
                    x += wgt
        seg.set_verts(verts)
        seg.set_facecolors(cols)

        revealed.set_data(t.iloc[:end], df["close"].iloc[:end])
        dot.set_data([t.iloc[cut]], [df["close"].iloc[cut]])

        delta = accb - accs
        for rect, d in zip(dbars, delta):
            rect.set_width(d)
            rect.set_color(GREEN if d >= 0 else RED)

        totals = acc.sum(axis=1)
        if totals.max() > 0:
            poc.set_ydata([g["centers"][int(totals.argmax())]] * 2)

        now = t.iloc[cut]
        elapsed = (now - g["t0"]).total_seconds() / 3600

        # current 2 hour block: isolate its slots, take its value area
        cur = block_of(now.hour)
        name, base = PALETTE[cur]
        va = None
        if args.block_band:
            mask = g["slot_block"] == cur
            va = value_area(acc[:, mask].sum(axis=1), g["edges"], args.block_va)
        if va is not None:
            bot, top, _ = va
            done_blocks[cur] = (bot, top)
            for key, y in (("top", top), ("bot", bot)):
                band[f"p_{key}"].set_ydata([y, y])
                band[f"v_{key}"].set_ydata([y, y])
                band[f"d_{key}"].set_ydata([y, y])
                band[f"d_{key}"].set_color(base)
                band[f"p_{key}"].set_color(base)
                band[f"v_{key}"].set_color(base)
                txt = band[f"t_{key}"]
                txt.set_position((x_max * 0.98, y))
                txt.set_text(f"{name} {key} {y:,.2f}")
                txt.set_color(base)
                txt.set_va("bottom" if key == "top" else "top")
                txt.set_visible(True)
            if args.ghost_blocks:
                for i, (b, (gb, gt)) in enumerate(sorted(done_blocks.items())):
                    if b == cur or i >= len(ghosts):
                        continue
                    ghosts[i][0].set_ydata([gt, gt])
                    ghosts[i][1].set_ydata([gb, gb])
                    ghosts[i][0].set_color(PALETTE[b][1])
                    ghosts[i][1].set_color(PALETTE[b][1])
        clock.set_text(f"{now:%a %H:%M} CT   +{elapsed:4.1f}h into session")
        while conf_left and conf_left[0][0] <= now:
            ts, px_c, sd, nh, poc_c = conf_left.pop(0)
            col = GREEN if sd == "LONG" else RED
            dy = -34 if sd == "LONG" else 34
            marks.append(axp.annotate(
                f"{sd}  confluence {nh}h\n{px_c:,.2f}  POC {poc_c:,.2f}", xy=(ts, px_c),
                xytext=(-8, dy), textcoords="offset points", fontsize=8,
                color="#0d1117", family="monospace", ha="right", zorder=9,
                bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none", alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.2)))

        if (args.signals and args.signal_mode in ("thin", "both")
                and on_va is not None and now >= rth_open and len(fired) < args.max_signals):
            px = float(df["close"].iloc[cut])
            row = int(np.clip(np.searchsorted(g["edges"], px, "right") - 1, 0, rows - 1))
            poc_v = totals.max()
            back = max(0, cut - win)
            moved = px - float(df["close"].iloc[back])
            fast = abs(moved) / on_range >= args.velocity
            recent = [f for f in fired
                      if (now - f[0]).total_seconds() / 60 < args.cooldown]
            traded = np.flatnonzero(totals > 0)
            side = kind = None
            why = ""
            miss = None

            def cooled(k):
                return not [f for f in fired if f[2] == k
                            and (now - f[0]).total_seconds() / 60 < args.cooldown]

            if poc_v > 0 and traded.size:
                hi_edge, lo_edge = int(traded.max()), int(traded.min())
                ext = args.delta_extension * on_range

                # rule 1: thin tail. Price ran into a vacuum at the extreme.
                if fast and cooled("thin tail"):
                    if (moved > 0 and px > on_va[1]
                            and thin_tail(totals, row, poc_v, args.thin_frac, args.tail_rows, True)
                            and (not args.delta_confluence or delta[row] < 0)):
                        side, kind = "SHORT", "thin tail"
                        why = f"row vol {totals[row] / poc_v:.0%} of POC"
                    elif (moved < 0 and px < on_va[0]
                          and thin_tail(totals, row, poc_v, args.thin_frac, args.tail_rows, False)
                          and (not args.delta_confluence or delta[row] > 0)):
                        side, kind = "LONG", "thin tail"
                        why = f"row vol {totals[row] / poc_v:.0%} of POC"

                # rule 2: delta flip. Price is extended and the tape at the
                # extreme has turned against it. No speed test: absorption
                # usually shows up once price has stopped going anywhere.
                if side is None and args.delta_signals and cooled("delta flip"):
                    ref = float(np.median(totals[traded]))
                    built = (totals[row] >= args.thin_frac * ref) if args.require_built else True
                    zt = slice(max(0, hi_edge - args.tail_rows), hi_edge + 1)
                    zb = slice(lo_edge, min(rows, lo_edge + args.tail_rows + 1))
                    near_hi = row >= hi_edge - 2 * args.tail_rows
                    near_lo = row <= lo_edge + 2 * args.tail_rows
                    dt, vt = delta[zt].sum(), max(totals[zt].sum(), 1.0)
                    db, vb = delta[zb].sum(), max(totals[zb].sum(), 1.0)
                    up_ext, dn_ext = px > on_va[1] + ext, px < on_va[0] - ext
                    if built and near_hi and up_ext and dt < 0 and abs(dt) / vt >= args.delta_ratio:
                        side, kind = "SHORT", "delta flip"
                        why = f"zone delta {dt:+,.0f} ({abs(dt) / vt:.0%} of its volume)"
                    elif built and near_lo and dn_ext and db > 0 and abs(db) / vb >= args.delta_ratio:
                        side, kind = "LONG", "delta flip"
                        why = f"zone delta {db:+,.0f} ({abs(db) / vb:.0%} of its volume)"
                    elif args.explain and (near_hi or near_lo):
                        at_hi = near_hi
                        d, v, e = (dt, vt, up_ext) if at_hi else (db, vb, dn_ext)
                        want = "negative" if at_hi else "positive"
                        if not e:
                            miss = (f"not extended enough ({px:,.2f} vs "
                                    f"{on_va[1] + ext:,.2f})" if at_hi else
                                    f"not extended enough ({px:,.2f} vs {on_va[0] - ext:,.2f})")
                        elif not built:
                            miss = f"row still thin ({totals[row] / ref:.0%} of median row)"
                        elif (d >= 0) if at_hi else (d <= 0):
                            miss = f"zone delta {d:+,.0f}, wanted {want}"
                        elif abs(d) / v < args.delta_ratio:
                            miss = (f"zone delta {abs(d) / v:.0%} of its volume, "
                                    f"under --delta-ratio {args.delta_ratio:.0%}")
            if args.explain and miss and side is None:
                state.setdefault("last_miss", None)
                if miss.split("(")[0] != (state["last_miss"] or "").split("(")[0]:
                    print(f"    [explain] {now:%H:%M} no delta flip: {miss}")
                    state["last_miss"] = miss
            if side:
                fired.append((now, px, side))
                col = RED if side == "SHORT" else GREEN
                dy = 34 if side == "SHORT" else -34
                marks.append(axp.annotate(
                    f"{side}  {kind}\n{px:,.2f}", xy=(now, px),
                    xytext=(-8, dy), textcoords="offset points", fontsize=8,
                    color="#0d1117", family="monospace", ha="right", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.35", fc=col, ec="none", alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2)))
                print(f"  {side:5s} {kind:10s} {now:%a %H:%M} @ {px:,.2f}  {why}, "
                      f"moved {moved:+.2f} in {args.velocity_window}m "
                      f"({abs(moved) / on_range:.0%} of ON range)")

        net = delta.sum()
        stats.set_text(f"cumulative volume {totals.sum():>12,.0f}    "
                       f"POC {g['centers'][int(totals.argmax())]:.2f}    "
                       f"last {df['close'].iloc[cut]:.2f}    "
                       f"session delta {net:+,.0f}"
                       + (f"    near-lo {cnt_lo[cut]}/{args.conf_lookback}  "
                          f"near-hi {cnt_hi[cut]}/{args.conf_lookback}  "
                          f"tol {tol_s[cut]:.2f}" if cnt_lo is not None else ""))
        stats.set_color(GREEN if net >= 0 else RED)
        return seg, revealed, dot, poc, clock, stats

    anim = animation.FuncAnimation(fig, draw, frames=n_frames, blit=False, interval=1000 / args.fps)

    writer = pick_writer(args)
    anim.save(args.out, writer=writer, dpi=100)
    print(f"wrote {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="output of ib_fetch_session.py")
    src.add_argument("--demo", action="store_true", help="synthetic session instead")
    p.add_argument("--out", default="replay.mp4", help=".mp4 needs ffmpeg, .gif does not")
    p.add_argument("--rows", type=int, default=60, help="price rows in the profile")
    p.add_argument("--frame-minutes", type=int, default=5,
                   help="market minutes advanced per frame")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--signal-mode", default="confluence",
                   choices=["thin", "confluence", "both"],
                   help="'thin' is the thin-tail plus delta-flip rules, 'confluence' "
                        "is the hourly-tail port of the thinkScript study")
    p.add_argument("--conf-tolerance-mode", default="atr", choices=["atr", "points"])
    p.add_argument("--conf-atr-mult", type=float, default=0.5)
    p.add_argument("--conf-tolerance-points", type=float, default=5.0)
    p.add_argument("--conf-lookback", type=int, default=3,
                   help="hours checked, including the current one (1-4)")
    p.add_argument("--conf-min-hours", type=int, default=2,
                   help="how many of those must be near the day's extreme")
    p.add_argument("--conf-no-confirm", dest="conf_confirm", action="store_false",
                   help="drop the requirement that price has reclaimed the level")
    p.add_argument("--conf-no-live", dest="conf_include_live", action="store_false",
                   help="exclude the still-forming current hour")
    p.add_argument("--conf-repeat", dest="conf_once", action="store_false",
                   help="allow more than one signal per side per session")
    p.add_argument("--no-signals", dest="signals", action="store_false",
                   help="hide the long/short thin-tail bubbles")
    p.add_argument("--rth-start", default="0830", help="RTH open in chart time, HHMM")
    p.add_argument("--thin-frac", type=float, default=0.35,
                   help="a row is thin below this fraction of the POC row")
    p.add_argument("--tail-rows", type=int, default=3,
                   help="how close to the running extreme the row must be")
    p.add_argument("--velocity", type=float, default=0.35,
                   help="move over the window, as a fraction of the overnight range")
    p.add_argument("--velocity-window", type=int, default=30, help="minutes")
    p.add_argument("--cooldown", type=int, default=45,
                   help="minutes before another signal can fire")
    p.add_argument("--max-signals", type=int, default=8)
    p.add_argument("--require-built", action="store_true",
                   help="delta-flip also requires the row to have stopped thinning, "
                        "measured against the median traded row")
    p.add_argument("--explain", action="store_true",
                   help="print why a delta-flip near the extreme did not fire")
    p.add_argument("--no-delta-signals", dest="delta_signals", action="store_false",
                   help="drop the delta-flip rule, leaving only thin tails")
    p.add_argument("--delta-extension", type=float, default=0.25,
                   help="delta-flip only fires this far beyond the overnight value "
                        "area, in units of the overnight range")
    p.add_argument("--delta-ratio", type=float, default=0.15,
                   help="delta at the extreme must be at least this fraction of the "
                        "volume sitting there")
    p.add_argument("--delta-confluence", action="store_true",
                   help="also require delta to disagree with the move at that row")
    p.add_argument("--delta-method", default="auto",
                   choices=["auto", "range", "direction", "columns"],
                   help="auto uses buy_volume/sell_volume columns if the csv has "
                        "them, else the range estimator")
    p.add_argument("--block-va", type=float, default=0.70,
                   help="fraction of the current 2h block's volume the band covers")
    p.add_argument("--no-block-band", dest="block_band", action="store_false",
                   help="hide the current-block top and bottom lines")
    p.add_argument("--ghost-blocks", action="store_true",
                   help="leave a faint band behind for each completed 2h block")
    p.add_argument("--no-ghost", dest="ghost", action="store_false",
                   help="hide the faint full-session line, so the future stays hidden")
    p.add_argument("--hold-frames", type=int, default=25,
                   help="frames to hold on the finished profile")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--bitrate", type=int, default=4000)
    render(p.parse_args())


if __name__ == "__main__":
    main()
