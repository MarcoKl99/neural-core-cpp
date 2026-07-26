import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Categorical palette slots 1 (blue) and 2 (orange) from the project's validated default palette -
# fixed order, never cycled.
COLOR_FORWARD = "#2a78d6"
COLOR_BACKWARD = "#eb6834"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

_UNIT_TO_MS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}

# Matches e.g. "Conv2D_Forward" -> base="Conv2D_", direction="Forward"
# and "CNN_FullBackwardPass" -> base="CNN_Full", direction="BackwardPass".
_DIRECTION_RE = re.compile(r"^(?P<base>.*?)(?P<direction>Forward(?:Pass)?|Backward(?:Pass)?)$")


def load_mean_timings(json_path):
    """Group benchmark 'mean' aggregates by base name -> {"Forward": ms, "Backward": ms}."""
    with open(json_path) as f:
        data = json.load(f)

    groups = {}
    for entry in data["benchmarks"]:
        if entry.get("run_type") != "aggregate" or entry.get("aggregate_name") != "mean":
            continue

        match = _DIRECTION_RE.match(entry["run_name"])
        if not match:
            continue

        base = match.group("base").removeprefix("BM_").rstrip("_")
        direction = "Backward" if match.group("direction").startswith("Backward") else "Forward"

        scale = _UNIT_TO_MS[entry["time_unit"]]
        ms = entry["cpu_time"] * scale

        groups.setdefault(base, {})[direction] = ms

    # Keep only bases that have both directions, sorted by Forward time descending.
    complete_bases = [b for b, v in groups.items() if "Forward" in v and "Backward" in v]
    for base in groups:
        if base not in complete_bases:
            print(
                f"warning: skipping '{base}' - missing a Forward or Backward entry", file=sys.stderr
            )
    ordered_bases = sorted(complete_bases, key=lambda b: groups[b]["Forward"], reverse=True)
    return ordered_bases, groups


def format_ms(value):
    if value < 1.0:
        return f"{value * 1000:.0f} µs"
    return f"{value:.2f} ms"


def plot(json_path, output_path):
    bases, groups = load_mean_timings(json_path)
    labels = [b.replace("_", " ") for b in bases]

    forward_vals = [groups[b]["Forward"] for b in bases]
    backward_vals = [groups[b]["Backward"] for b in bases]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    x = range(len(bases))
    width = 0.36
    bars_fwd = ax.bar(
        [i - width / 2 for i in x],
        forward_vals,
        width,
        label="Forward",
        color=COLOR_FORWARD,
        zorder=3,
    )
    bars_bwd = ax.bar(
        [i + width / 2 for i in x],
        backward_vals,
        width,
        label="Backward",
        color=COLOR_BACKWARD,
        zorder=3,
    )

    ax.set_ylabel("Time (ms)", color=INK_SECONDARY, fontsize=11)
    ax.set_title(
        f"Conv Net Benchmark — {Path(json_path).stem}",
        color=INK_PRIMARY,
        fontsize=14,
        fontweight="bold",
        pad=16,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=INK_SECONDARY, fontsize=10)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="x", length=0)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)

    ax.yaxis.grid(True, which="major", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    for bars, values in ((bars_fwd, forward_vals), (bars_bwd, backward_vals)):
        for rect, value in zip(bars, values):
            ax.annotate(
                format_ms(value),
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_SECONDARY,
            )

    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, facecolor=SURFACE)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Path to a Google Benchmark JSON result file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: alongside the input file, same stem)",
    )
    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"error: {args.json_path} does not exist", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or args.json_path.with_suffix(".png")
    plot(args.json_path, output_path)


if __name__ == "__main__":
    main()
