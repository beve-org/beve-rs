#!/usr/bin/env python3

"""Render side-by-side bar charts comparing beve and serde-beve timing CSVs."""

import argparse
import csv
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot beve vs serde-beve timings from CSV")
    parser.add_argument("csv", help="Path to CSV generated via `--csv` flag on compare example")
    parser.add_argument(
        "--output",
        default="comparison.png",
        help="Where to write the plot image (default: comparison.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Image DPI when saving with matplotlib (default: 160)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot in addition to saving it (matplotlib backend only)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "matplotlib", "svg"],
        default="auto",
        help="Plotting backend. `auto` tries matplotlib and falls back to SVG",
    )
    parser.add_argument(
        "--svg-width",
        type=int,
        default=1280,
        help="Canvas width in pixels when rendering SVG (default: 1280)",
    )
    return parser.parse_args()


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "scenario",
            "title",
            "label",
            "beve_us",
            "serde_beve_us",
            "serde_over_beve",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"CSV missing columns: {', '.join(sorted(missing))}")
        return [row for row in reader]


def group_rows(rows: list[dict[str, str]]):
    ordered_keys: OrderedDict[str, dict[str, object]] = OrderedDict()
    for row in rows:
        key = row["scenario"]
        entry = ordered_keys.setdefault(
            key,
            {"title": row["title"], "entries": []},
        )
        entry["entries"].append(
            {
                "label": row["label"],
                "beve": float(row["beve_us"]),
                "serde": float(row["serde_beve_us"]),
            }
        )
    return ordered_keys


def try_import_matplotlib(show: bool):
    try:
        import matplotlib  # type: ignore
        if not show and "MPLBACKEND" not in os.environ:
            matplotlib.use("Agg", force=True)
        elif show and os.environ.get("MPLBACKEND", "").lower() == "agg":
            raise SystemExit("Cannot use --show when MPLBACKEND is set to Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError:
        return None
    return plt


def plot_with_matplotlib(
    rows: list[dict[str, str]],
    output_path: str,
    dpi: int,
    show: bool,
) -> None:
    plt = try_import_matplotlib(show)
    if plt is None:
        raise SystemExit(
            "matplotlib backend requested but matplotlib is not installed. Install with `python -m pip install matplotlib`."
        )

    grouped = group_rows(rows)
    if not grouped:
        raise SystemExit("No data rows to plot")

    scenario_count = len(grouped)
    fig, axes = plt.subplots(
        scenario_count,
        1,
        figsize=(9, max(3.0, 2.4 * scenario_count)),
        squeeze=False,
    )

    for ax, (scenario, payload) in zip(axes.flat, grouped.items()):
        labels = [entry["label"] for entry in payload["entries"]]
        beve_values = [entry["beve"] for entry in payload["entries"]]
        serde_values = [entry["serde"] for entry in payload["entries"]]
        x_positions = range(len(labels))
        width = 0.38

        ax.bar(
            [x - width / 2 for x in x_positions],
            beve_values,
            width=width,
            label="beve",
            color="#3b82f6",
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            serde_values,
            width=width,
            label="serde-beve",
            color="#f97316",
        )

        for idx, (x_pos, beve_val, serde_val) in enumerate(
            zip(x_positions, beve_values, serde_values)
        ):
            if beve_val <= 0.0 or math.isnan(beve_val):
                continue
            ratio = serde_val / beve_val
            y = max(beve_val, serde_val) * 1.05
            ax.text(
                x_pos,
                y,
                f"{ratio:.2f}x",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#374151",
            )

        ax.set_title(f"{payload['title']} ({scenario})")
        ax.set_ylabel("µs / iteration")
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()


def plot_with_svg(rows: list[dict[str, str]], output_path: str, width: int) -> None:
    grouped = group_rows(rows)
    if not grouped:
        raise SystemExit("No data rows to plot")

    max_value = 0.0
    total_height = 40  # top margin
    per_entry_height = 44
    scenario_gap = 36
    for payload in grouped.values():
        entries = payload["entries"]
        if not entries:
            continue
        max_value = max(
            max_value,
            max(max(item["beve"], item["serde"]) for item in entries),
        )
        total_height += len(entries) * per_entry_height + scenario_gap

    if max_value <= 0.0:
        raise SystemExit("All timing values are zero; cannot scale bars")

    width = max(width, 720)
    margin_left = 200
    margin_right = max(260, width // 5)
    margin_bottom = 32
    chart_width = width - margin_left - margin_right
    if chart_width <= 0:
        raise SystemExit(
            f"Canvas width ({width}) too small for margins; increase --svg-width"
        )

    y_cursor = 40
    parts = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='{}' height='{}' viewBox='0 0 {} {}'>".format(
            width, total_height + margin_bottom, width, total_height + margin_bottom
        ),
        "<style>text { font-family: Helvetica, Arial, sans-serif; fill: #1f2937; }</style>",
        "<rect x='0' y='0' width='{}' height='{}' fill='#f9fafb' />".format(
            width, total_height + margin_bottom
        ),
        "<text x='20' y='24' font-size='20' font-weight='bold'>beve vs serde-beve</text>",
    ]

    for scenario, payload in grouped.items():
        entries = payload["entries"]
        if not entries:
            continue

        parts.append(
            f"<text x='20' y='{y_cursor + 6}' font-size='14' font-weight='bold'>{payload['title']} ({scenario})</text>"
        )
        y_cursor += 20

        for idx, entry in enumerate(entries):
            y_base = y_cursor + idx * per_entry_height
            label_y = y_base + 18
            parts.append(
                f"<text x='20' y='{label_y}' font-size='12'>{entry['label']}</text>"
            )

            beve_len = entry["beve"] / max_value * chart_width
            serde_len = entry["serde"] / max_value * chart_width
            bar_height = 12
            beve_y = y_base + 4
            serde_y = beve_y + bar_height + 4

            parts.append(
                "<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{color}' rx='2' ry='2' />".format(
                    x=margin_left,
                    y=beve_y,
                    w=max(beve_len, 0.5),
                    h=bar_height,
                    color="#2563eb",
                )
            )
            parts.append(
                "<rect x='{x}' y='{y}' width='{w}' height='{h}' fill='{color}' rx='2' ry='2' />".format(
                    x=margin_left,
                    y=serde_y,
                    w=max(serde_len, 0.5),
                    h=bar_height,
                    color="#f97316",
                )
            )

            bevelabel = f"{entry['beve']:.2f} µs"
            serdelabel = f"{entry['serde']:.2f} µs"
            ratio = entry["serde"] / entry["beve"] if entry["beve"] > 0 else float("nan")
            ratio_label = f"{ratio:.2f}x" if ratio == ratio else "nan"
            value_x = margin_left + chart_width + 8
            parts.append(
                f"<text x='{value_x}' y='{beve_y + bar_height - 2}' font-size='11'>{bevelabel}</text>"
            )
            parts.append(
                f"<text x='{value_x}' y='{serde_y + bar_height - 2}' font-size='11'>{serdelabel} ({ratio_label})</text>"
            )

        y_cursor += len(entries) * per_entry_height + scenario_gap

    legend_y = total_height + margin_bottom - 10
    parts.append(
        f"<rect x='{margin_left}' y='{legend_y - 14}' width='14' height='14' fill='#2563eb' rx='2' ry='2' />"
    )
    parts.append(
        f"<text x='{margin_left + 20}' y='{legend_y - 2}' font-size='12'>beve</text>"
    )
    parts.append(
        f"<rect x='{margin_left + 70}' y='{legend_y - 14}' width='14' height='14' fill='#f97316' rx='2' ry='2' />"
    )
    parts.append(
        f"<text x='{margin_left + 90}' y='{legend_y - 2}' font-size='12'>serde-beve</text>"
    )

    parts.append("</svg>")

    path = Path(output_path)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    backend = args.backend

    if backend == "matplotlib":
        plot_with_matplotlib(rows, args.output, args.dpi, args.show)
        return

    if backend == "svg":
        plot_with_svg(rows, args.output, args.svg_width)
        return

    # auto
    plt = try_import_matplotlib(show=False)
    if plt is not None:
        plt.close("all")  # ensure clean state before real plotting
        plot_with_matplotlib(rows, args.output, args.dpi, args.show)
        return

    sys.stderr.write("matplotlib not found; falling back to SVG renderer.\n")
    if not args.output.lower().endswith(".svg"):
        sys.stderr.write("Hint: append .svg to --output when using the SVG backend.\n")
    plot_with_svg(rows, args.output, args.svg_width)


if __name__ == "__main__":
    main()
