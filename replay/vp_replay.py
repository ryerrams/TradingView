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

    colors = []
    for s in range(n_slots):
        t = t0 + pd.Timedelta(minutes=30 * s)
        b = block_of(t.hour)
        colors.append(shade(PALETTE[b][1], SHADES[slot_in_block(t.hour, t.minute)]))

    # spread each bar's volume evenly over the rows its range covers
    j_lo = np.clip(np.searchsorted(edges, df["low"].to_numpy(), "right") - 1, 0, rows - 1)
    j_hi = np.clip(np.searchsorted(edges, df["high"].to_numpy(), "right") - 1, 0, rows - 1)
    per = df["volume"].to_numpy() / (j_hi - j_lo + 1)

    return dict(edges=edges, height=height, centers=centers, lo=lo, hi=hi,
                slot=slot, n_slots=n_slots, colors=colors,
                j_lo=j_lo, j_hi=j_hi, per=per, t0=t0)


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
    rows = args.rows
    g = build(df, rows)
    total = final_matrix(df, g, rows)
    x_max = total.sum(axis=1).max() * 1.06

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
    fig, (axp, axv) = plt.subplots(
        1, 2, figsize=(args.width / 100, args.height / 100), dpi=100,
        gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.02}, sharey=True)

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

    clock = fig.text(0.012, 0.965, "", fontsize=13, color="#e6edf3", family="monospace")
    stats = fig.text(0.012, 0.935, "", fontsize=10, color="#8b949e", family="monospace")
    fig.legend(handles=[Patch(facecolor=c, label=n) for n, c in PALETTE],
               loc="lower center", ncol=12, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.10)

    acc = np.zeros((rows, g["n_slots"]))
    state = {"i": 0}

    def draw(frame):
        end = min(len(df), (frame + 1) * bars_per_frame)
        for i in range(state["i"], end):
            acc[g["j_lo"][i]:g["j_hi"][i] + 1, g["slot"][i]] += g["per"][i]
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

        totals = acc.sum(axis=1)
        if totals.max() > 0:
            poc.set_ydata([g["centers"][int(totals.argmax())]] * 2)

        now = t.iloc[cut]
        elapsed = (now - g["t0"]).total_seconds() / 3600
        clock.set_text(f"{now:%a %H:%M} CT   +{elapsed:4.1f}h into session")
        stats.set_text(f"cumulative volume {totals.sum():>12,.0f}    "
                       f"POC {g['centers'][int(totals.argmax())]:.2f}    "
                       f"last {df['close'].iloc[cut]:.2f}")
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
