#!/usr/bin/env python3
"""Single-day Bigelow SpiderMAGIC contour with H2SO4 condensation sink overlay."""

import argparse
import glob
import math
import os
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

DEFAULT_DAY = "2025-11-26"
DEFAULT_SPIDER_FOLDER = Path(
    r"C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/Spider Data/inverted"
)
DEFAULT_OUTPUT = Path(
    r"D:/Documents/PhD-Research/SMPS Comparison/Figures/Bigelow_banana_daily/CS_overlay"
)
DEFAULT_TEMPERATURE_K = 298.15
H2SO4_DIFFUSION_M2_S = 0.077e-4
ACCOMMODATION_COEFFICIENT = 1.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot one daily Bigelow SMPS contour with H2SO4 condensation sink overlay."
    )
    parser.add_argument("--day", default=DEFAULT_DAY, help="Local date as YYYY-MM-DD.")
    parser.add_argument("--spider-folder", type=Path, default=DEFAULT_SPIDER_FOLDER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--polarity",
        choices=["both", "positive", "negative"],
        default="both",
        help="Ion polarity to plot. SpiderMAGIC convention here: V1 < 0 positive, V1 > 0 negative.",
    )
    parser.add_argument("--max-gap-minutes", type=float, default=10.0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def is_size_bin_column(col):
    try:
        float(str(col))
        return True
    except (TypeError, ValueError):
        return False


def log_edges_from_centers(diameters_nm):
    logs = np.log10(diameters_nm)
    if len(logs) < 2:
        raise ValueError("At least two diameter bins are required.")

    edges = np.empty(len(logs) + 1, dtype=float)
    edges[1:-1] = (logs[:-1] + logs[1:]) / 2.0
    edges[0] = logs[0] - (logs[1] - logs[0]) / 2.0
    edges[-1] = logs[-1] + (logs[-1] - logs[-2]) / 2.0
    return edges


def dlog10_widths(diameters_nm):
    return np.diff(log_edges_from_centers(diameters_nm))


def day_bounds(day):
    start = datetime.combine(pd.to_datetime(day).date(), time.min)
    return start, start + timedelta(days=1)


def continuous_segments(times, max_gap):
    if len(times) == 0:
        return []

    segments = []
    start = 0
    for index in range(1, len(times)):
        if times[index] - times[index - 1] > max_gap:
            segments.append((start, index))
            start = index
    segments.append((start, len(times)))
    return segments


def time_edges(times, day_start, day_end):
    if len(times) == 1:
        first = max(day_start, times[0] - timedelta(minutes=2.5))
        last = min(day_end, times[0] + timedelta(minutes=2.5))
        return np.array([mdates.date2num(first), mdates.date2num(last)])

    edges = [times[0] - (times[1] - times[0]) / 2]
    edges.extend(times[i] + (times[i + 1] - times[i]) / 2 for i in range(len(times) - 1))
    edges.append(times[-1] + (times[-1] - times[-2]) / 2)
    edges[0] = max(day_start, edges[0])
    edges[-1] = min(day_end, edges[-1])
    return np.array([mdates.date2num(edge) for edge in edges])


def calculate_condensation_sink(values_dndlogdp_cm3, diameters_nm):
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


def filter_polarity(df, polarity):
    if polarity == "both":
        return df

    v1 = pd.to_numeric(df["V1 (V)"], errors="coerce")
    if polarity == "positive":
        return df[v1 < 0]
    if polarity == "negative":
        return df[v1 > 0]
    raise ValueError("polarity must be 'both', 'positive', or 'negative'")


def load_bigelow_day(day, spider_folder, polarity):
    date_token = pd.to_datetime(day).strftime("%Y%m%d")
    pattern = str(spider_folder / f"SpiderMAGIC_SN*_N_{date_token}_*.txt")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")

    frames = []
    for file_path in files:
        df = pd.read_csv(file_path, low_memory=False)
        df["source_file"] = os.path.basename(file_path)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True, sort=False)
    df["datetime_local"] = pd.to_datetime(df["Start datetime (PC)"], errors="coerce")
    df = df.loc[df["datetime_local"].notna()].copy()
    df = df.set_index("datetime_local")
    try:
        df.index = df.index.tz_localize("US/Eastern", ambiguous="infer")
    except Exception:
        df.index = df.index.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward")
    df = df[df.index.notna()]
    if "Mode" in df.columns:
        df = df[df["Mode"].astype(str).str.lower().eq("scan")]
    df = filter_polarity(df, polarity)

    start = pd.Timestamp(day, tz="US/Eastern")
    end = start + pd.Timedelta(days=1)
    df = df.loc[(df.index >= start) & (df.index < end)].sort_index()
    if df.empty:
        raise ValueError(f"No scan rows found for {day}.")

    size_cols = [c for c in df.columns if is_size_bin_column(c)]
    diameters_nm = np.array([float(c) for c in size_cols], dtype=float)
    order = np.argsort(diameters_nm)
    size_cols = [size_cols[i] for i in order]
    diameters_nm = diameters_nm[order]

    values = df[size_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return df, diameters_nm, values, files


def plot_contour_with_cs(day, df, diameters_nm, values, out_path, max_gap_minutes, dpi, polarity):
    day_start, day_end = day_bounds(day)
    local_time = df.index.tz_convert("US/Eastern").tz_localize(None)
    plot_df = pd.DataFrame(values, index=local_time, columns=diameters_nm)
    plot_df = plot_df.groupby(level=0).mean().sort_index()

    times = [ts.to_pydatetime() for ts in pd.DatetimeIndex(plot_df.index)]
    values = np.where(plot_df.to_numpy(dtype=float) <= 0, np.nan, plot_df.to_numpy(dtype=float))
    cs_s1, valid_fraction, valid_bins = calculate_condensation_sink(values, diameters_nm)
    cs_smooth = (
        pd.Series(cs_s1, index=pd.DatetimeIndex(plot_df.index))
        .rolling("5min", center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.09, right=0.78, bottom=0.18, top=0.88)

    y_edges = 10**log_edges_from_centers(diameters_nm)
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
            norm=LogNorm(vmin=1e3, vmax=1e5),
            cmap="turbo",
        )

    ax.set_yscale("log")
    ax.set_ylim(float(np.nanmin(diameters_nm)), 400)
    ax.set_ylabel("Dp [nm]")
    ax.set_xlabel("Time (US/Eastern)")
    polarity_label = "both polarity" if polarity == "both" else f"{polarity} ions"
    ax.set_title(f"Bigelow SpiderMAGIC {polarity_label} with H2SO4 CS - {day}")
    ax.set_xlim(day_start, day_end)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    ax_cs = ax.twinx()
    line = ax_cs.plot(
        pd.DatetimeIndex(plot_df.index),
        cs_s1,
        color="#111111",
        linewidth=0.9,
        alpha=0.85,
        label="H2SO4 CS",
        zorder=10,
    )[0]
    line.set_path_effects([pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()])
    ax_cs.set_ylabel("H2SO4 condensation sink [s^-1]")
    cs_positive = cs_s1[np.isfinite(cs_s1) & (cs_s1 > 0)]
    if cs_positive.size:
        ax_cs.set_ylim(0.0, float(np.nanpercentile(cs_positive, 99)) * 1.18)
    ax_cs.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.8)

    if mesh is not None:
        cax = fig.add_axes([0.86, 0.20, 0.025, 0.62])
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label("dN/dlogDp [cm^-3]")

    fig.text(
        0.09,
        0.025,
        (
            f"CS uses {polarity_label} scans; "
            "Dv=0.077 cm^2 s^-1, alpha=1, fixed T=298.15 K; "
            f"median valid bins={np.nanmedian(valid_bins):.0f}/{len(diameters_nm)}."
        ),
        fontsize=7,
        color="0.25",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    cs_out = pd.DataFrame(
        {
            "datetime_local": pd.DatetimeIndex(plot_df.index),
            "polarity": polarity,
            "CS_H2SO4_s1": cs_s1,
            "CS_H2SO4_5min_median_s1": cs_smooth,
            "condensation_lifetime_min": np.where(cs_s1 > 0, 1.0 / (60.0 * cs_s1), np.nan),
            "valid_cs_bin_fraction": valid_fraction,
            "valid_cs_bins": valid_bins,
            "diameter_min_nm": float(np.nanmin(diameters_nm)),
            "diameter_max_nm": float(np.nanmax(diameters_nm)),
            "diffusion_coefficient_used_cm2_s": 0.077,
            "accommodation_coefficient": ACCOMMODATION_COEFFICIENT,
        }
    )
    cs_out.to_csv(out_path.with_suffix(".csv"), index=False)

    return {
        "plot": out_path,
        "csv": out_path.with_suffix(".csv"),
        "polarity": polarity,
        "n_scans": int(len(plot_df)),
        "mean_cs": float(np.nanmean(cs_s1)),
        "median_cs": float(np.nanmedian(cs_s1)),
        "max_cs": float(np.nanmax(cs_s1)),
        "median_smoothed_cs": float(np.nanmedian(cs_smooth)),
    }


def main():
    args = parse_args()
    df, diameters_nm, values, files = load_bigelow_day(args.day, args.spider_folder, args.polarity)
    polarity_suffix = "" if args.polarity == "both" else f"_{args.polarity}"
    out_path = args.output / f"Bigelow_banana_{args.day}{polarity_suffix}_CS_overlay.png"
    result = plot_contour_with_cs(
        args.day,
        df,
        diameters_nm,
        values,
        out_path,
        args.max_gap_minutes,
        args.dpi,
        args.polarity,
    )

    print("Input files:")
    for file_path in files:
        print(f"  {file_path}")
    print(f"Saved plot: {result['plot']}")
    print(f"Saved CS CSV: {result['csv']}")
    print(f"Polarity: {result['polarity']}")
    print(f"Scans: {result['n_scans']}")
    print(f"Mean CS: {result['mean_cs']:.6g} s^-1")
    print(f"Median CS: {result['median_cs']:.6g} s^-1")
    print(f"Max CS: {result['max_cs']:.6g} s^-1")


if __name__ == "__main__":
    main()
