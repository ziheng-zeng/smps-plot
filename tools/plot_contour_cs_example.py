#!/usr/bin/env python3
"""Single-day SMPS contour with sulfuric-acid condensation sink overlay."""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, time, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


K_B = 1.380649e-23
N_A = 6.02214076e23
M_H2SO4_KG_MOL = 98.079e-3
M_H2SO4_MOLECULE_KG = M_H2SO4_KG_MOL / N_A

DEFAULT_SMPS_WIDE = Path("merged outputs/master_all_processed/smps_master_wide_time_ordered.csv")
DEFAULT_OUTPUT = Path("outputs/smps_acsm_pm25_batch/08_contour_CS")
DEFAULT_DAY = "2024-03-06"
DEFAULT_TEMPERATURE_K = 298.15
H2SO4_DIFFUSION_M2_S = 0.077e-4
ACCOMMODATION_COEFFICIENT = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one daily SMPS contour with CS line overlay.")
    parser.add_argument("--day", default=DEFAULT_DAY, help="Local date as YYYY-MM-DD.")
    parser.add_argument("--all-days", action="store_true", help="Plot every local date in the SMPS wide file.")
    parser.add_argument("--smps-wide", type=Path, default=DEFAULT_SMPS_WIDE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-gap-minutes", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def parse_diameter_columns(columns: list[str]) -> tuple[list[str], np.ndarray]:
    pattern = re.compile(r"^dndlogdp_([0-9.]+)_nm$")
    pairs: list[tuple[str, float]] = []
    for column in columns:
        match = pattern.match(column)
        if match:
            pairs.append((column, float(match.group(1))))
    pairs.sort(key=lambda item: item[1])
    if not pairs:
        raise ValueError("No dndlogdp_<diameter>_nm columns found.")
    return [item[0] for item in pairs], np.array([item[1] for item in pairs], dtype=float)


def log_edges_from_centers(diameters_nm: np.ndarray) -> np.ndarray:
    logs = np.log10(diameters_nm)
    if len(logs) < 2:
        raise ValueError("At least two diameter bins are required.")
    edges = np.empty(len(logs) + 1, dtype=float)
    edges[1:-1] = (logs[:-1] + logs[1:]) / 2.0
    edges[0] = logs[0] - (logs[1] - logs[0]) / 2.0
    edges[-1] = logs[-1] + (logs[-1] - logs[-2]) / 2.0
    return edges


def dlog10_widths(diameters_nm: np.ndarray) -> np.ndarray:
    return np.diff(log_edges_from_centers(diameters_nm))


def day_bounds(day: str) -> tuple[datetime, datetime]:
    start = datetime.combine(pd.to_datetime(day).date(), time.min)
    return start, start + timedelta(days=1)


def finite_positive_range(values: np.ndarray) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0, 10.0
    vmin = max(1.0, float(np.nanpercentile(positive, 5)))
    vmax = float(np.nanpercentile(positive, 99))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def continuous_segments(times: list[datetime], max_gap: timedelta) -> list[tuple[int, int]]:
    if not times:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > max_gap:
            segments.append((start, index))
            start = index
    segments.append((start, len(times)))
    return segments


def time_edges(times: list[datetime], day_start: datetime, day_end: datetime) -> np.ndarray:
    if len(times) == 1:
        first = max(day_start, times[0] - timedelta(minutes=2.5))
        last = min(day_end, times[0] + timedelta(minutes=2.5))
        return np.array([mdates.date2num(first), mdates.date2num(last)])

    edges: list[datetime] = [times[0] - (times[1] - times[0]) / 2]
    edges.extend(times[i] + (times[i + 1] - times[i]) / 2 for i in range(len(times) - 1))
    edges.append(times[-1] + (times[-1] - times[-2]) / 2)
    edges[0] = max(day_start, edges[0])
    edges[-1] = min(day_end, edges[-1])
    return np.array([mdates.date2num(edge) for edge in edges])


def calculate_condensation_sink(
    values_dndlogdp_cm3: np.ndarray,
    diameters_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values_dndlogdp_cm3, dtype=float)
    values = np.where(values < 0, np.nan, values)
    valid = np.isfinite(values)
    valid_fraction = valid.mean(axis=1)
    valid_bins = valid.sum(axis=1)

    widths = dlog10_widths(diameters_nm)
    number_bin_m3 = np.where(valid, values * widths[None, :] * 1.0e6, np.nan)
    dp_m = diameters_nm * 1.0e-9

    lambda_vapor_m = 3.0 * H2SO4_DIFFUSION_M2_S * math.sqrt(
        math.pi * M_H2SO4_MOLECULE_KG / (8.0 * K_B * DEFAULT_TEMPERATURE_K)
    )
    knudsen = 2.0 * lambda_vapor_m / dp_m
    alpha = ACCOMMODATION_COEFFICIENT
    beta = (1.0 + knudsen) / (
        1.0
        + (4.0 / (3.0 * alpha) + 0.377) * knudsen
        + (4.0 / (3.0 * alpha)) * knudsen**2
    )
    bin_cs = 2.0 * math.pi * H2SO4_DIFFUSION_M2_S * beta[None, :] * dp_m[None, :] * number_bin_m3
    return np.nansum(bin_cs, axis=1), valid_fraction, valid_bins


def plot_contour_cs(
    day: str,
    day_df: pd.DataFrame,
    diameter_columns: list[str],
    diameters_nm: np.ndarray,
    out_path: Path,
    max_gap_minutes: float,
    dpi: int,
) -> dict[str, object]:
    day_start, day_end = day_bounds(day)
    day_df = day_df.sort_values("datetime_local")
    times = [ts.to_pydatetime() for ts in day_df["datetime_local"]]
    values = day_df[diameter_columns].to_numpy(dtype=float)
    values = np.where(values < 0, np.nan, values)
    cs_s1, valid_fraction, valid_bins = calculate_condensation_sink(values, diameters_nm)

    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    fig.subplots_adjust(left=0.07, right=0.80, bottom=0.16, top=0.89)
    y_edges = 10**log_edges_from_centers(diameters_nm)
    vmin, vmax = finite_positive_range(values)
    mesh = None
    for start, end in continuous_segments(times, timedelta(minutes=max_gap_minutes)):
        segment_times = times[start:end]
        if not segment_times:
            continue
        mesh = ax.pcolormesh(
            time_edges(segment_times, day_start, day_end),
            y_edges,
            values[start:end, :].T,
            shading="flat",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="turbo",
        )

    ax_cs = ax.twinx()
    ax_cs.spines["right"].set_position(("axes", 1.03))
    line = ax_cs.plot(
        day_df["datetime_local"],
        cs_s1,
        color="#111111",
        linewidth=1.8,
        label=r"CS from valid SMPS bins",
        zorder=10,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=3.4, foreground="white"), pe.Normal()])

    ax.set_title(rf"Kigali SMPS dN/dlogDp with H$_2$SO$_4$ condensation sink - {day}")
    ax.set_xlabel("Local time")
    ax.set_ylabel(r"$D_p$ (nm)")
    ax.set_yscale("log")
    ax.set_ylim(max(8.0, float(np.nanmin(diameters_nm))), min(700.0, float(np.nanmax(diameters_nm))))
    ax.grid(True, axis="x", color="0.82", linewidth=0.6)
    ax.set_xlim(mdates.date2num(day_start), mdates.date2num(day_end))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax_cs.set_ylabel(r"H$_2$SO$_4$ condensation sink (s$^{-1}$)")
    cs_positive = cs_s1[np.isfinite(cs_s1) & (cs_s1 > 0)]
    if cs_positive.size:
        ax_cs.set_ylim(0.0, float(np.nanpercentile(cs_positive, 99)) * 1.18)
    ax_cs.legend(loc="upper right", bbox_to_anchor=(0.995, 0.995), frameon=True, facecolor="white", framealpha=0.78)

    if mesh is not None:
        cax = fig.add_axes([0.91, 0.19, 0.018, 0.62])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(r"$dN/d\log_{10}D_p$ (cm$^{-3}$)")

    fig.text(
        0.01,
        0.01,
        (
            f"Dv=0.077 cm2 s-1, alpha=1, fixed T=298.15 K; "
            f"CS uses all {len(diameters_nm)} diameter columns where valid "
            f"(median valid bins={np.nanmedian(valid_bins):.0f}, median valid fraction={np.nanmedian(valid_fraction):.3f})."
        ),
        fontsize=7,
        color="0.25",
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    cs_out = day_df[["source_file", "sample_number", "datetime"]].copy()
    cs_out["CS_H2SO4_9p82_637p8nm_s1"] = cs_s1
    cs_out["condensation_lifetime_min"] = np.where(cs_s1 > 0, 1.0 / (60.0 * cs_s1), np.nan)
    cs_out["valid_cs_bin_fraction"] = valid_fraction
    cs_out["valid_cs_bins"] = valid_bins
    cs_out["diameter_min_nm"] = float(np.nanmin(diameters_nm))
    cs_out["diameter_max_nm"] = float(np.nanmax(diameters_nm))
    cs_out["diffusion_coefficient_used_cm2_s"] = 0.077
    cs_out["accommodation_coefficient"] = ACCOMMODATION_COEFFICIENT
    cs_out.to_csv(out_path.with_suffix(".csv"), index=False)
    return {
        "date_local": day,
        "n_scans": int(len(day_df)),
        "plot_file": str(out_path),
        "cs_file": str(out_path.with_suffix(".csv")),
        "mean_CS_H2SO4_9p82_637p8nm_s1": float(np.nanmean(cs_s1)),
        "median_CS_H2SO4_9p82_637p8nm_s1": float(np.nanmedian(cs_s1)),
        "min_CS_H2SO4_9p82_637p8nm_s1": float(np.nanmin(cs_s1)),
        "max_CS_H2SO4_9p82_637p8nm_s1": float(np.nanmax(cs_s1)),
        "median_condensation_lifetime_min": float(np.nanmedian(cs_out["condensation_lifetime_min"])),
        "median_valid_cs_bin_fraction": float(np.nanmedian(valid_fraction)),
        "median_valid_cs_bins": float(np.nanmedian(valid_bins)),
        "diameter_columns": int(len(diameters_nm)),
        "diameter_min_nm": float(np.nanmin(diameters_nm)),
        "diameter_max_nm": float(np.nanmax(diameters_nm)),
        "diffusion_coefficient_used_cm2_s": 0.077,
        "accommodation_coefficient": ACCOMMODATION_COEFFICIENT,
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.smps_wide)
    diameter_columns, diameters_nm = parse_diameter_columns(list(df.columns))
    df["datetime_local"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.loc[df["datetime_local"].notna()].copy()
    df["date_local"] = df["datetime_local"].dt.date.astype(str)
    days = sorted(df["date_local"].dropna().unique()) if args.all_days else [args.day]
    records: list[dict[str, object]] = []
    for day in days:
        day_df = df.loc[df["date_local"] == day].copy()
        if day_df.empty:
            continue
        out_path = args.output / f"{day}_contour_CS.png"
        record = plot_contour_cs(
            day,
            day_df,
            diameter_columns,
            diameters_nm,
            out_path,
            args.max_gap_minutes,
            args.dpi,
        )
        records.append(record)
        print(out_path)
    summary = pd.DataFrame(records)
    summary_name = "contour_CS_daily_summary.csv" if args.all_days else f"{args.day}_contour_CS_summary.csv"
    summary_path = args.output / summary_name
    summary.to_csv(summary_path, index=False)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
