#!/usr/bin/env python3
"""Plot campaign condensation-sink distributions from contour_CS scan outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap


INPUT_DIR = Path("outputs/smps_acsm_pm25_batch/08_contour_CS")
OUTPUT_DIR = Path("outputs/smps_acsm_pm25_batch/09_cs_distribution")
CS_COLUMN = "CS_H2SO4_9p82_637p8nm_s1"
CANDIDATE_DAYS = ["2024-02-06", "2024-03-06"]
KNOWN_NONSTANDARD_SOURCE = "20240219_0915-20240220_1158.txt"


def load_scan_cs(input_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(input_dir.glob("????-??-??_contour_CS.csv")):
        frame = pd.read_csv(path)
        if CS_COLUMN not in frame.columns:
            continue
        frame["datetime_local"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frame = frame.loc[frame["datetime_local"].notna()].copy()
        frame["date_local"] = frame["datetime_local"].dt.date.astype(str)
        frame["hour_local"] = (
            frame["datetime_local"].dt.hour
            + frame["datetime_local"].dt.minute / 60.0
            + frame["datetime_local"].dt.second / 3600.0
        )
        frame[CS_COLUMN] = pd.to_numeric(frame[CS_COLUMN], errors="coerce")
        frame = frame.loc[np.isfinite(frame[CS_COLUMN]) & (frame[CS_COLUMN] > 0)].copy()
        frames.append(frame[["source_file", "datetime_local", "date_local", "hour_local", CS_COLUMN]])
    if not frames:
        raise ValueError(f"No usable contour_CS CSV files found in {input_dir}.")
    return pd.concat(frames, ignore_index=True)


def daily_summary(scans: pd.DataFrame) -> pd.DataFrame:
    return (
        scans.groupby("date_local", sort=True)
        .agg(
            n_scans=(CS_COLUMN, "size"),
            mean_cs_s1=(CS_COLUMN, "mean"),
            median_cs_s1=(CS_COLUMN, "median"),
            p10_cs_s1=(CS_COLUMN, lambda x: float(np.nanpercentile(x, 10))),
            p90_cs_s1=(CS_COLUMN, lambda x: float(np.nanpercentile(x, 90))),
        )
        .reset_index()
    )


def plot_distribution(scans: pd.DataFrame, daily: pd.DataFrame, output_path: Path) -> None:
    fig, ax_daily = plt.subplots(figsize=(12.8, 5.7))

    candidate_colors = {"2024-02-06": "#d62728", "2024-03-06": "#1f77b4"}
    x = mdates.date2num(scans["datetime_local"])
    scans = scans.copy()
    scans["hour_bin"] = np.floor(scans["hour_local"]).clip(0, 23).astype(int)
    hour_cmap = ListedColormap(plt.get_cmap("hsv", 24)(np.arange(24)))
    hour_norm = BoundaryNorm(np.arange(-0.5, 24.5, 1.0), hour_cmap.N)
    scatter = ax_daily.scatter(
        x,
        scans[CS_COLUMN],
        c=scans["hour_bin"],
        cmap=hour_cmap,
        norm=hour_norm,
        s=5,
        alpha=0.58,
        linewidths=0,
        rasterized=True,
    )
    for day in CANDIDATE_DAYS:
        day_num = mdates.date2num(pd.to_datetime(day))
        ax_daily.axvline(day_num, color=candidate_colors[day], linewidth=1.5, alpha=0.85)
        ax_daily.annotate(
            day,
            (day_num, scans[CS_COLUMN].quantile(0.985)),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=candidate_colors[day],
            rotation=90,
            va="top",
        )

    ax_daily.set_yscale("log")
    ax_daily.set_xlabel("Date")
    ax_daily.set_ylabel(r"H$_2$SO$_4$ condensation sink (s$^{-1}$)")
    ax_daily.set_title("Scan-level condensation sink colored by local hour")
    ax_daily.grid(True, axis="y", color="0.90")
    ax_daily.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_daily.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    cbar = fig.colorbar(scatter, ax=ax_daily, pad=0.012)
    cbar.set_label("Local hour of day")
    cbar.set_ticks(range(0, 24, 2))

    fig.text(
        0.01,
        0.01,
        (
            "CS from all available SMPS bins per timestamp; Dv=0.077 cm2 s-1, alpha=1. "
            f"Nonstandard source present: {KNOWN_NONSTANDARD_SOURCE}."
        ),
        fontsize=7,
        color="0.30",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot campaign condensation-sink distributions from contour_CS scan outputs."
    )
    parser.add_argument("input_dir", nargs="?", type=Path, default=INPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scans = load_scan_cs(args.input_dir)
    daily = daily_summary(scans)
    scans.to_csv(args.output / "cs_scan_distribution_data.csv", index=False)
    daily.to_csv(args.output / "cs_daily_distribution_summary.csv", index=False)
    plot_distribution(scans, daily, args.output / "cs_daily_raw_hour_colored.png")
    print(args.output / "cs_daily_raw_hour_colored.png")
    print(args.output / "cs_daily_distribution_summary.csv")
    print(args.output / "cs_scan_distribution_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
