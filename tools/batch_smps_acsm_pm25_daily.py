#!/usr/bin/env python3
"""Batch daily Kigali SMPS, ACSM/BC, and PM2.5 comparison workflow.

This workflow is descriptive. It creates daily inspection figures, daily
availability/metric tables, and summary scatter/rank plots. It does not classify
NPF events or claim mechanistic conclusions.
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, time, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


LOCAL_TZ = "Africa/Kigali"
KNOWN_NONSTANDARD_SOURCE = "20240219_0915-20240220_1158.txt"
CANDIDATE_DAYS = {"2024-02-06", "2024-03-06"}
DAYTIME_WINDOWS = {
    "06_18": (time(6, 0), time(18, 0)),
    "11_12": (time(11, 0), time(12, 0)),
    "10_13": (time(10, 0), time(13, 0)),
    "06_10": (time(6, 0), time(10, 0)),
    "10_18": (time(10, 0), time(18, 0)),
}

DEFAULT_SMPS_WIDE = Path("merged outputs/master_all_processed/smps_master_wide_time_ordered.csv")
DEFAULT_SMPS_METRICS = Path("outputs/psd_basic_analysis/smps_psd_scan_metrics.csv")
DEFAULT_ACSM = Path("other data/PM_component_no_PMF_with_NA.csv")
DEFAULT_PM25 = Path("other data/Kigali_US_Emabassy_PM2.5_T640.csv")
DEFAULT_OUTPUT = Path("outputs/smps_acsm_pm25_batch")

ACSM_COMPONENTS = ["Org (ug/m3)", "NO3 (ug/m3)", "SO4 (ug/m3)", "NH4 (ug/m3)", "Chl (ug/m3)"]
K_B = 1.380649e-23
N_A = 6.02214076e23
M_H2SO4_KG_MOL = 98.079e-3
M_H2SO4_MOLECULE_KG = M_H2SO4_KG_MOL / N_A
DEFAULT_TEMPERATURE_K = 298.15
DEFAULT_PRESSURE_PA = 101325.0
DEFAULT_H2SO4_DIFFUSION_M2_S = 0.8e-5
DEFAULT_ACCOMMODATION_COEFFICIENT = 1.0
CS_MIN_DIAMETER_NM = 9.82
CS_MAX_DIAMETER_NM = 414.2
VARIABLES = {
    "SO4": {"label": "SO4", "units": "ug m$^{-3}$", "color": "#1f77b4"},
    "BC": {"label": "BC", "units": "ug m$^{-3}$", "color": "#252525"},
    "NR_PM1": {"label": "NR-PM1", "units": "ug m$^{-3}$", "color": "#2ca02c"},
    "PM25": {"label": "PM2.5", "units": "ug m$^{-3}$", "color": "#d62728"},
}
CS_TOTAL_COLUMN = "CS_H2SO4_9p82_414nm_s1"
CS_OUTPUT_COLUMNS = [
    CS_TOTAL_COLUMN,
    "condensation_lifetime_min",
    "CS_10_25nm_s1",
    "CS_25_50nm_s1",
    "CS_50_100nm_s1",
    "CS_100_300nm_s1",
    "CS_300_414nm_s1",
    "valid_cs_bin_fraction",
    "temperature_used_K",
    "pressure_used_Pa",
    "diffusion_coefficient_used_m2_s",
]
SMPS_METRICS = ["N_total_cm3", "surface_area_um2_cm3", CS_TOTAL_COLUMN]
TRANSITION_VARIABLES = {
    "SO4": {"label": "SO4", "units": "ug m$^{-3}$"},
    "BC": {"label": "BC", "units": "ug m$^{-3}$"},
    "NR_PM1": {"label": "NR-PM1", "units": "ug m$^{-3}$"},
    "PM25": {"label": "PM2.5", "units": "ug m$^{-3}$"},
    "surface_area_um2_cm3": {"label": "SMPS surface area", "units": "um$^2$ cm$^{-3}$"},
    "N_total_cm3": {"label": "SMPS total number", "units": "cm$^{-3}$"},
    "N_over_surface_area": {"label": "N / surface area", "units": "cm$^{-3}$ / um$^2$ cm$^{-3}$"},
    CS_TOTAL_COLUMN: {"label": "SMPS-range CS", "units": "s$^{-1}$"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-process daily Kigali SMPS with overlapping ACSM/BC and PM2.5."
    )
    parser.add_argument("--smps-wide", type=Path, default=DEFAULT_SMPS_WIDE)
    parser.add_argument("--smps-metrics", type=Path, default=DEFAULT_SMPS_METRICS)
    parser.add_argument("--acsm", type=Path, default=DEFAULT_ACSM)
    parser.add_argument("--pm25", type=Path, default=DEFAULT_PM25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-gap-minutes", type=float, default=10.0)
    parser.add_argument("--common-valid-fraction", type=float, default=0.95)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
        }
    )


def output_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "contour": root / "01_contour_only",
        "lines": root / "02_daily_lineplots",
        "combined": root / "03_contour_plus_lines",
        "scatter": root / "04_summary_scatterplots",
        "tables": root / "05_summary_tables",
        "interesting": root / "06_interestingness_plots",
        "candidate": root / "07_candidate_evolution",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_diameter_columns(columns: list[str]) -> tuple[list[str], np.ndarray]:
    pattern = re.compile(r"^dndlogdp_([0-9.]+)_nm$")
    pairs: list[tuple[str, float]] = []
    for column in columns:
        match = pattern.match(column)
        if match:
            pairs.append((column, float(match.group(1))))
    pairs.sort(key=lambda item: item[1])
    if not pairs:
        raise ValueError("No dndlogdp_<diameter>_nm columns found in SMPS wide file.")
    return [item[0] for item in pairs], np.array([item[1] for item in pairs], dtype=float)


def load_smps_wide(path: Path, common_valid_fraction: float) -> tuple[pd.DataFrame, list[str], np.ndarray, dict[str, object]]:
    df = pd.read_csv(path)
    diameter_columns, diameters = parse_diameter_columns(list(df.columns))
    details: dict[str, object] = {
        "smps_wide": str(path),
        "smps_rows_before_qc": len(df),
        "excluded_source_file": KNOWN_NONSTANDARD_SOURCE,
    }
    if "source_file" in df.columns:
        keep = df["source_file"].astype(str) != KNOWN_NONSTANDARD_SOURCE
        details["excluded_smps_rows"] = int((~keep).sum())
        df = df.loc[keep].copy()
    else:
        details["excluded_smps_rows"] = "source_file unavailable"

    df["datetime_local"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.loc[df["datetime_local"].notna()].copy()
    numeric = df[diameter_columns].apply(pd.to_numeric, errors="coerce").mask(lambda x: x <= 0)
    valid_fraction = numeric.notna().mean(axis=0)
    common_mask = valid_fraction >= common_valid_fraction
    common_columns = [col for col, keep in zip(diameter_columns, common_mask) if keep]
    common_diameters = diameters[common_mask.to_numpy()]
    if not common_columns:
        raise ValueError("No common SMPS diameter bins passed the valid-fraction threshold.")

    for column in common_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column] <= 0, column] = np.nan
    df = df.copy()
    df["date_local"] = df["datetime_local"].dt.date.astype(str)
    details["smps_rows_after_qc"] = len(df)
    details["common_diameter_min_nm"] = float(common_diameters.min())
    details["common_diameter_max_nm"] = float(common_diameters.max())
    details["common_diameter_bins"] = int(len(common_diameters))
    return df, common_columns, common_diameters, details


def dlog10_widths(diameters_nm: np.ndarray) -> np.ndarray:
    logs = np.log10(diameters_nm)
    if len(logs) < 2:
        raise ValueError("At least two diameter bins are required.")
    edges = np.empty(len(logs) + 1, dtype=float)
    edges[1:-1] = (logs[:-1] + logs[1:]) / 2.0
    edges[0] = logs[0] - (logs[1] - logs[0]) / 2.0
    edges[-1] = logs[-1] + (logs[-1] - logs[-2]) / 2.0
    return np.diff(edges)


def calculate_cs_metrics(
    df: pd.DataFrame,
    diameter_columns: list[str],
    diameters_nm: np.ndarray,
    minimum_valid_fraction: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cs_mask = (diameters_nm >= CS_MIN_DIAMETER_NM) & (diameters_nm <= CS_MAX_DIAMETER_NM)
    cs_columns = [column for column, keep in zip(diameter_columns, cs_mask) if keep]
    cs_diameters_nm = diameters_nm[cs_mask]
    if len(cs_columns) < 2:
        raise ValueError("Not enough common bins in the 9.82-414.2 nm CS range.")

    values = df[cs_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values) & (values >= 0)
    valid_fraction = valid.mean(axis=1)

    widths = dlog10_widths(cs_diameters_nm)
    dp_m = cs_diameters_nm * 1.0e-9
    number_bin_m3 = np.where(valid, values * widths[None, :] * 1.0e6, np.nan)

    diffusion = DEFAULT_H2SO4_DIFFUSION_M2_S
    lambda_vapor_m = 3.0 * diffusion * math.sqrt(
        math.pi * M_H2SO4_MOLECULE_KG / (8.0 * K_B * DEFAULT_TEMPERATURE_K)
    )
    knudsen = 2.0 * lambda_vapor_m / dp_m
    alpha = DEFAULT_ACCOMMODATION_COEFFICIENT
    beta = (1.0 + knudsen) / (
        1.0
        + (4.0 / (3.0 * alpha) + 0.377) * knudsen
        + (4.0 / (3.0 * alpha)) * knudsen**2
    )
    bin_cs = 2.0 * math.pi * diffusion * beta[None, :] * dp_m[None, :] * number_bin_m3
    enough_valid = valid_fraction >= minimum_valid_fraction

    out = df[["source_file", "sample_number", "datetime", "date_local"]].copy()
    out[CS_TOTAL_COLUMN] = np.nansum(bin_cs, axis=1)
    out.loc[~enough_valid, CS_TOTAL_COLUMN] = np.nan
    out["condensation_lifetime_min"] = np.where(
        out[CS_TOTAL_COLUMN] > 0,
        1.0 / (60.0 * out[CS_TOTAL_COLUMN]),
        np.nan,
    )

    ranges = {
        "CS_10_25nm_s1": (cs_diameters_nm >= 10.0) & (cs_diameters_nm < 25.0),
        "CS_25_50nm_s1": (cs_diameters_nm >= 25.0) & (cs_diameters_nm < 50.0),
        "CS_50_100nm_s1": (cs_diameters_nm >= 50.0) & (cs_diameters_nm < 100.0),
        "CS_100_300nm_s1": (cs_diameters_nm >= 100.0) & (cs_diameters_nm < 300.0),
        "CS_300_414nm_s1": (cs_diameters_nm >= 300.0) & (cs_diameters_nm <= CS_MAX_DIAMETER_NM),
    }
    for column, mask in ranges.items():
        out[column] = np.nansum(bin_cs[:, mask], axis=1)
        out.loc[~enough_valid, column] = np.nan

    out["valid_cs_bin_fraction"] = valid_fraction
    out["temperature_used_K"] = DEFAULT_TEMPERATURE_K
    out["pressure_used_Pa"] = DEFAULT_PRESSURE_PA
    out["diffusion_coefficient_used_m2_s"] = diffusion
    out.replace([np.inf, -np.inf], np.nan, inplace=True)

    details = {
        "cs_diameter_min_nm": float(cs_diameters_nm.min()),
        "cs_diameter_max_nm": float(cs_diameters_nm.max()),
        "cs_diameter_bins": int(len(cs_diameters_nm)),
        "cs_minimum_valid_bin_fraction": minimum_valid_fraction,
        "cs_temperature_used_K": DEFAULT_TEMPERATURE_K,
        "cs_pressure_used_Pa": DEFAULT_PRESSURE_PA,
        "cs_h2so4_diffusion_m2_s": diffusion,
        "cs_accommodation_coefficient": DEFAULT_ACCOMMODATION_COEFFICIENT,
        "cs_valid_scans": int(out[CS_TOTAL_COLUMN].notna().sum()),
    }
    return out, details


def load_acsm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["datetime_local"] = (
        pd.to_datetime(df["date"], utc=True, errors="coerce")
        .dt.tz_convert(LOCAL_TZ)
        .dt.tz_localize(None)
    )
    for column in ACSM_COMPONENTS + ["BC (ug/m3)"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    present_components = [column for column in ACSM_COMPONENTS if column in df.columns]
    df["NR_PM1"] = df[present_components].sum(axis=1, min_count=1) if present_components else np.nan
    if "SO4 (ug/m3)" in df.columns:
        df["SO4"] = df["SO4 (ug/m3)"]
    if "BC (ug/m3)" in df.columns:
        df["BC"] = df["BC (ug/m3)"]
    df = df.loc[df["datetime_local"].notna()].copy()
    df["date_local"] = df["datetime_local"].dt.date.astype(str)
    return df


def load_pm25(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "QC Name" in df.columns:
        df = df.loc[df["QC Name"].astype(str).str.lower().eq("valid")].copy()
    df["datetime_local"] = pd.to_datetime(df["Date (LT)"], errors="coerce")
    df["PM25"] = pd.to_numeric(df["Raw Conc."], errors="coerce")
    df = df.loc[df["datetime_local"].notna()].copy()
    df["date_local"] = df["datetime_local"].dt.date.astype(str)
    return df


def load_smps_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "source_file" in df.columns:
        df = df.loc[df["source_file"].astype(str) != KNOWN_NONSTANDARD_SOURCE].copy()
    df["datetime_local"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.loc[df["datetime_local"].notna()].copy()
    df["date_local"] = df["datetime_local"].dt.date.astype(str)
    for column in SMPS_METRICS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def add_condensation_sink_to_smps_metrics(smps_metrics: pd.DataFrame, cs_metrics: pd.DataFrame) -> pd.DataFrame:
    cs_cols = ["source_file", "sample_number", "datetime", "date_local", *CS_OUTPUT_COLUMNS]
    cs = cs_metrics[cs_cols].copy()
    cs["datetime"] = pd.to_datetime(cs["datetime"], errors="coerce")
    if smps_metrics.empty:
        out = cs.copy()
        out["datetime_local"] = out["datetime"]
        return out

    join_keys = ["source_file", "sample_number", "datetime"]
    out = smps_metrics.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    for column in CS_OUTPUT_COLUMNS:
        if column in out.columns:
            out = out.drop(columns=column)
    out = out.merge(cs[join_keys + CS_OUTPUT_COLUMNS], on=join_keys, how="left")
    return out


def date_bounds(day: str) -> tuple[datetime, datetime]:
    start = datetime.combine(pd.to_datetime(day).date(), time.min)
    return start, start + timedelta(days=1)


def diameter_edges(diameters: np.ndarray) -> np.ndarray:
    logs = np.log10(diameters)
    edges = np.empty(len(logs) + 1)
    edges[1:-1] = (logs[:-1] + logs[1:]) / 2.0
    edges[0] = logs[0] - (logs[1] - logs[0]) / 2.0
    edges[-1] = logs[-1] + (logs[-1] - logs[-2]) / 2.0
    return 10**edges


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


def format_time_axis(ax: plt.Axes, day_start: datetime, day_end: datetime) -> None:
    ax.set_xlim(mdates.date2num(day_start), mdates.date2num(day_end))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def add_candidate_window(ax: plt.Axes, day: str) -> None:
    if day not in CANDIDATE_DAYS:
        return
    start = datetime.combine(pd.to_datetime(day).date(), time(11, 0))
    end = datetime.combine(pd.to_datetime(day).date(), time(12, 0))
    ax.axvspan(start, end, color="#fdd49e", alpha=0.35, lw=0)


def plot_smps_contour_on_axis(
    ax: plt.Axes,
    day_df: pd.DataFrame,
    day: str,
    diameter_columns: list[str],
    diameters: np.ndarray,
    y_edges: np.ndarray,
    vmin: float,
    vmax: float,
    max_gap_minutes: float,
) -> object | None:
    day_start, day_end = date_bounds(day)
    day_df = day_df.sort_values("datetime_local")
    times = [ts.to_pydatetime() for ts in day_df["datetime_local"]]
    values = day_df[diameter_columns].to_numpy(dtype=float)
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
    ax.set_yscale("log")
    ax.set_ylabel(r"$D_p$ (nm)")
    ax.set_ylim(max(8.0, float(np.nanmin(diameters))), min(700.0, float(np.nanmax(diameters))))
    ax.grid(True, axis="x", color="0.82", linewidth=0.6)
    add_candidate_window(ax, day)
    format_time_axis(ax, day_start, day_end)
    return mesh


def plot_contour_only(
    day: str,
    day_df: pd.DataFrame,
    diameter_columns: list[str],
    diameters: np.ndarray,
    y_edges: np.ndarray,
    vmin: float,
    vmax: float,
    max_gap_minutes: float,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    mesh = plot_smps_contour_on_axis(
        ax, day_df, day, diameter_columns, diameters, y_edges, vmin, vmax, max_gap_minutes
    )
    ax.set_title(f"Kigali SMPS dN/dlogDp - {day}")
    ax.set_xlabel("Local time")
    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(r"$dN/d\log_{10}D_p$ (cm$^{-3}$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def day_variable_frames(day: str, acsm: pd.DataFrame, pm25: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    acsm_day = acsm.loc[acsm["date_local"] == day].sort_values("datetime_local")
    pm_day = pm25.loc[pm25["date_local"] == day].sort_values("datetime_local")
    for var in ["SO4", "BC", "NR_PM1"]:
        if var in acsm_day.columns and acsm_day[var].notna().any():
            frames[var] = acsm_day[["datetime_local", var]].rename(columns={var: "value"})
    if "PM25" in pm_day.columns and pm_day["PM25"].notna().any():
        frames["PM25"] = pm_day[["datetime_local", "PM25"]].rename(columns={"PM25": "value"})
    return frames


def plot_line_panels(day: str, frames: dict[str, pd.DataFrame], out_path: Path, dpi: int) -> None:
    day_start, day_end = date_bounds(day)
    variables = [var for var in ["SO4", "BC", "NR_PM1", "PM25"] if var in frames]
    if not variables:
        return
    fig, axes = plt.subplots(len(variables), 1, figsize=(10.5, max(2.1 * len(variables), 3.0)), sharex=True)
    if len(variables) == 1:
        axes = [axes]
    for ax, var in zip(axes, variables):
        meta = VARIABLES[var]
        frame = frames[var]
        ax.plot(frame["datetime_local"], frame["value"], marker="o", linewidth=1.4, markersize=3, color=meta["color"])
        ax.set_ylabel(f"{meta['label']}\n({meta['units']})")
        ax.grid(True, color="0.88")
        add_candidate_window(ax, day)
        format_time_axis(ax, day_start, day_end)
    axes[0].set_title(f"Kigali ACSM/BC and PM2.5 daily measurements - {day}")
    axes[-1].set_xlabel("Local time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_line_compact(day: str, frames: dict[str, pd.DataFrame], out_path: Path, dpi: int) -> None:
    day_start, day_end = date_bounds(day)
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    plotted = False
    for var in ["SO4", "BC", "NR_PM1", "PM25"]:
        if var not in frames:
            continue
        meta = VARIABLES[var]
        frame = frames[var]
        ax.plot(frame["datetime_local"], frame["value"], marker="o", linewidth=1.4, markersize=3, label=meta["label"], color=meta["color"])
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(f"Kigali daily measurement overlay - {day}")
    ax.set_xlabel("Local time")
    ax.set_ylabel(r"Concentration (ug m$^{-3}$)")
    ax.grid(True, color="0.88")
    ax.legend(frameon=False, ncol=4)
    add_candidate_window(ax, day)
    format_time_axis(ax, day_start, day_end)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_combined(
    day: str,
    day_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    diameter_columns: list[str],
    diameters: np.ndarray,
    y_edges: np.ndarray,
    vmin: float,
    vmax: float,
    max_gap_minutes: float,
    out_path: Path,
    dpi: int,
) -> None:
    variables = [var for var in ["SO4", "BC", "NR_PM1", "PM25"] if var in frames]
    if not variables:
        return
    day_start, day_end = date_bounds(day)
    height_ratios = [2.2] + [1.0] * len(variables)
    fig, axes = plt.subplots(
        len(variables) + 1,
        1,
        figsize=(11.5, 3.0 + 1.45 * len(variables)),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    mesh = plot_smps_contour_on_axis(
        axes[0], day_df, day, diameter_columns, diameters, y_edges, vmin, vmax, max_gap_minutes
    )
    axes[0].set_title(f"Kigali SMPS, ACSM/BC, and PM2.5 - {day}")
    for ax, var in zip(axes[1:], variables):
        meta = VARIABLES[var]
        frame = frames[var]
        ax.plot(frame["datetime_local"], frame["value"], marker="o", linewidth=1.3, markersize=2.8, color=meta["color"])
        ax.set_ylabel(f"{meta['label']}\n({meta['units']})")
        ax.grid(True, color="0.88")
        add_candidate_window(ax, day)
        format_time_axis(ax, day_start, day_end)
    axes[-1].set_xlabel("Local time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def hourly_metric_frame(day: str, smps_metrics: pd.DataFrame, metric: str) -> pd.DataFrame:
    if smps_metrics.empty or metric not in smps_metrics.columns:
        return pd.DataFrame(columns=["datetime_local", "value"])
    day_df = smps_metrics.loc[smps_metrics["date_local"] == day, ["datetime_local", metric]].dropna()
    if day_df.empty:
        return pd.DataFrame(columns=["datetime_local", "value"])
    hourly = (
        day_df.set_index("datetime_local")[metric]
        .resample("1h")
        .median()
        .dropna()
        .reset_index()
        .rename(columns={metric: "value"})
    )
    return hourly


def candidate_periods(day: str) -> list[tuple[time, time, str, str]]:
    if day == "2024-03-06":
        return [
            (time(6, 0), time(10, 0), "morning", "#d9d9d9"),
            (time(10, 0), time(12, 0), "transition", "#fdd49e"),
            (time(12, 0), time(17, 0), "growth period", "#c7e9c0"),
        ]
    if day == "2024-02-06":
        return [
            (time(6, 0), time(10, 0), "morning", "#d9d9d9"),
            (time(10, 0), time(12, 0), "onset", "#fdd49e"),
            (time(14, 0), time(17, 0), "later period", "#c7e9c0"),
        ]
    return []


def add_candidate_periods(ax: plt.Axes, day: str, label: bool = False) -> None:
    periods = candidate_periods(day)
    if not periods:
        return
    base_date = pd.to_datetime(day).date()
    for start_t, end_t, name, color in periods:
        start = datetime.combine(base_date, start_t)
        end = datetime.combine(base_date, end_t)
        ax.axvspan(start, end, color=color, alpha=0.28, lw=0)
        if label:
            ax.text(start + (end - start) / 2, 0.98, name, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)


def plot_candidate_evolution(
    day: str,
    smps_day: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    smps_metrics: pd.DataFrame,
    diameter_columns: list[str],
    diameters: np.ndarray,
    y_edges: np.ndarray,
    vmin: float,
    vmax: float,
    max_gap_minutes: float,
    out_path: Path,
    dpi: int,
) -> None:
    day_start, day_end = date_bounds(day)
    variable_order = ["PM25", "NR_PM1", "SO4", "BC", "surface_area_um2_cm3", "N_total_cm3"]
    metric_frames: dict[str, pd.DataFrame] = dict(frames)
    metric_frames["surface_area_um2_cm3"] = hourly_metric_frame(day, smps_metrics, "surface_area_um2_cm3")
    metric_frames["N_total_cm3"] = hourly_metric_frame(day, smps_metrics, "N_total_cm3")
    available = [var for var in variable_order if var in metric_frames and not metric_frames[var].empty]
    if not available:
        return

    fig, axes = plt.subplots(
        len(available) + 1,
        1,
        figsize=(11.5, 3.0 + 1.35 * len(available)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25] + [1.0] * len(available)},
    )
    plot_smps_contour_on_axis(
        axes[0], smps_day, day, diameter_columns, diameters, y_edges, vmin, vmax, max_gap_minutes
    )
    add_candidate_periods(axes[0], day, label=True)
    title = f"Candidate-day evolution - {day}"
    if day == "2024-02-06":
        title += " (gap limits continuous-growth inference)"
    axes[0].set_title(title)

    labels = {
        "surface_area_um2_cm3": ("SMPS SA", r"um$^2$ cm$^{-3}$", "#9467bd"),
        "N_total_cm3": ("SMPS N", r"cm$^{-3}$", "#8c564b"),
    }
    for ax, var in zip(axes[1:], available):
        if var in VARIABLES:
            label = VARIABLES[var]["label"]
            units = VARIABLES[var]["units"]
            color = VARIABLES[var]["color"]
        else:
            label, units, color = labels[var]
        frame = metric_frames[var]
        ax.plot(frame["datetime_local"], frame["value"], marker="o", linewidth=1.3, markersize=2.8, color=color)
        ax.set_ylabel(f"{label}\n({units})")
        ax.grid(True, color="0.88")
        add_candidate_periods(ax, day)
        format_time_axis(ax, day_start, day_end)
    axes[-1].set_xlabel("Local time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def window_mean(df: pd.DataFrame, day: str, column: str, start_time: time, end_time: time) -> float:
    if df.empty or column not in df.columns:
        return np.nan
    day_start = datetime.combine(pd.to_datetime(day).date(), start_time)
    day_end = datetime.combine(pd.to_datetime(day).date(), end_time)
    mask = (df["datetime_local"] >= day_start) & (df["datetime_local"] < day_end)
    return float(df.loc[mask, column].mean()) if mask.any() else np.nan


def summarize_day(
    day: str,
    acsm: pd.DataFrame,
    pm25: pd.DataFrame,
    smps_metrics: pd.DataFrame,
    availability_row: dict[str, object],
) -> dict[str, object]:
    acsm_day = acsm.loc[acsm["date_local"] == day]
    pm_day = pm25.loc[pm25["date_local"] == day]
    smps_day = smps_metrics.loc[smps_metrics["date_local"] == day] if not smps_metrics.empty else pd.DataFrame()
    row: dict[str, object] = {"date_local": day, **availability_row}
    source_by_var = {
        "SO4": acsm_day,
        "BC": acsm_day,
        "NR_PM1": acsm_day,
        "PM25": pm_day,
    }
    for var, frame in source_by_var.items():
        row[f"mean_{var}"] = float(frame[var].mean()) if var in frame.columns else np.nan
        row[f"max_{var}"] = float(frame[var].max()) if var in frame.columns else np.nan
        for suffix, (start_time, end_time) in DAYTIME_WINDOWS.items():
            row[f"mean_{var}_{suffix}"] = window_mean(frame, day, var, start_time, end_time)
        morning = row[f"mean_{var}_06_10"]
        late_day = row[f"mean_{var}_10_18"]
        row[f"{var}_10_18_over_06_10"] = safe_ratio(late_day, morning)
        row[f"{var}_10_13_over_06_10"] = safe_ratio(row[f"mean_{var}_10_13"], morning)
        row[f"{var}_11_12_over_06_18"] = safe_ratio(row[f"mean_{var}_11_12"], row[f"mean_{var}_06_18"])

    for metric in SMPS_METRICS:
        if metric not in smps_day.columns:
            continue
        row[f"mean_{metric}"] = float(smps_day[metric].mean())
        row[f"median_{metric}"] = float(smps_day[metric].median())
        for suffix, (start_time, end_time) in DAYTIME_WINDOWS.items():
            row[f"mean_{metric}_{suffix}"] = window_mean(smps_day, day, metric, start_time, end_time)
        row[f"{metric}_10_13_over_06_10"] = safe_ratio(
            row.get(f"mean_{metric}_10_13"), row.get(f"mean_{metric}_06_10")
        )

    for suffix in ["06_10", "10_13", "06_18"]:
        row[f"N_over_surface_area_{suffix}"] = safe_ratio(
            row.get(f"mean_N_total_cm3_{suffix}"), row.get(f"mean_surface_area_um2_cm3_{suffix}")
        )
    row["N_over_surface_area_10_13_over_06_10"] = safe_ratio(
        row.get("N_over_surface_area_10_13"), row.get("N_over_surface_area_06_10")
    )
    return row


def safe_ratio(num: object, den: object) -> float:
    try:
        n = float(num)
        d = float(den)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(n) or not np.isfinite(d) or d == 0:
        return np.nan
    return n / d


def build_availability(smps: pd.DataFrame, acsm: pd.DataFrame, pm25: pd.DataFrame) -> pd.DataFrame:
    smps_counts = smps.groupby("date_local").size().rename("n_smps_scans")
    acsm_counts = acsm.groupby("date_local").size().rename("n_acsm_hours")
    pm25_counts = pm25.groupby("date_local").size().rename("n_pm25_hours")
    availability = pd.concat([smps_counts, acsm_counts, pm25_counts], axis=1).fillna(0).astype(int)
    availability = availability.loc[availability["n_smps_scans"] > 0].copy()
    availability["has_smps"] = availability["n_smps_scans"] > 0
    availability["has_acsm"] = availability["n_acsm_hours"] > 0
    availability["has_pm25"] = availability["n_pm25_hours"] > 0
    availability = availability.reset_index().rename(columns={"index": "date_local"})
    cols = ["date_local", "has_smps", "has_acsm", "has_pm25", "n_smps_scans", "n_acsm_hours", "n_pm25_hours"]
    return availability[cols].sort_values("date_local")


def scatter_filename(x: str, y: str) -> str:
    return f"daytime_scatter_{x.lower()}_vs_{y.lower()}.png"


def positive_log_norm(values: pd.Series) -> LogNorm | None:
    positive = pd.to_numeric(values, errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.empty:
        return None
    vmin = float(positive.quantile(0.05))
    vmax = float(positive.quantile(0.95))
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = float(positive.min())
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(positive.max())
    if vmax <= vmin:
        vmax = vmin * 1.01
    return LogNorm(vmin=vmin, vmax=vmax)


def plot_daytime_scatter(summary: pd.DataFrame, x_var: str, y_var: str, out_path: Path, dpi: int) -> None:
    x_col = f"mean_{x_var}_06_18"
    y_col = f"mean_{y_var}_06_18"
    color_col = f"mean_{CS_TOTAL_COLUMN}_06_18"
    required = ["date_local", x_col, y_col, color_col]
    if any(column not in summary.columns for column in required):
        return
    data = summary[required].dropna()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    is_candidate = data["date_local"].isin(CANDIDATE_DAYS)
    norm = positive_log_norm(data[color_col])
    if norm is None:
        return
    sc = ax.scatter(
        data.loc[~is_candidate, x_col],
        data.loc[~is_candidate, y_col],
        c=data.loc[~is_candidate, color_col],
        cmap="viridis",
        norm=norm,
        s=28,
        alpha=0.8,
        edgecolor="none",
        label="Other days",
    )
    ax.scatter(
        data.loc[is_candidate, x_col],
        data.loc[is_candidate, y_col],
        c=data.loc[is_candidate, color_col],
        cmap="viridis",
        norm=norm,
        marker="*",
        s=190,
        edgecolor="black",
        linewidth=0.6,
        label="Candidate days",
    )
    for _, row in data.loc[is_candidate].iterrows():
        ax.annotate(row["date_local"], (row[x_col], row[y_col]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"Daytime {VARIABLES[x_var]['label']} ({VARIABLES[x_var]['units']})")
    ax.set_ylabel(f"Daytime {VARIABLES[y_var]['label']} ({VARIABLES[y_var]['units']})")
    ax.set_title(f"Daytime {VARIABLES[y_var]['label']} vs {VARIABLES[x_var]['label']} colored by CS")
    ax.grid(True, color="0.9")
    ax.legend(frameon=False)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Daytime SMPS-range CS (s$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_flagged_scatter(
    summary: pd.DataFrame,
    x_var: str,
    y_var: str,
    flag_col: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    x_col = f"mean_{x_var}_06_18"
    y_col = f"mean_{y_var}_06_18"
    data = summary[["date_local", x_col, y_col, flag_col]].dropna(subset=[x_col, y_col])
    if data.empty:
        return
    is_candidate = data["date_local"].isin(CANDIDATE_DAYS)
    is_flag = data[flag_col].fillna(False).astype(bool)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(data.loc[~is_flag & ~is_candidate, x_col], data.loc[~is_flag & ~is_candidate, y_col], s=25, color="0.55", alpha=0.6, label="Other days")
    ax.scatter(data.loc[is_flag, x_col], data.loc[is_flag, y_col], s=52, color="#1f77b4", alpha=0.85, label="Flagged")
    ax.scatter(data.loc[is_candidate, x_col], data.loc[is_candidate, y_col], marker="*", s=180, color="#d62728", edgecolor="black", linewidth=0.6, label="Candidate days")
    for _, row in data.loc[is_candidate | is_flag].iterrows():
        ax.annotate(row["date_local"], (row[x_col], row[y_col]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel(f"Daytime {VARIABLES[x_var]['label']} ({VARIABLES[x_var]['units']})")
    ax.set_ylabel(f"Daytime {VARIABLES[y_var]['label']} ({VARIABLES[y_var]['units']})")
    ax.set_title(title)
    ax.grid(True, color="0.9")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_ratio_scatter(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x_col = "NR_PM1_10_18_over_06_10"
    y_col = "BC_10_18_over_06_10"
    data = summary[["date_local", x_col, y_col]].dropna()
    if data.empty:
        return
    is_candidate = data["date_local"].isin(CANDIDATE_DAYS)
    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    ax.axvline(1.0, color="0.8", linewidth=1)
    ax.axhline(1.0, color="0.8", linewidth=1)
    ax.scatter(data.loc[~is_candidate, x_col], data.loc[~is_candidate, y_col], s=28, color="0.5", alpha=0.65)
    ax.scatter(data.loc[is_candidate, x_col], data.loc[is_candidate, y_col], marker="*", s=180, color="#d62728", edgecolor="black", linewidth=0.6)
    for _, row in data.loc[is_candidate].iterrows():
        ax.annotate(row["date_local"], (row[x_col], row[y_col]), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("NR-PM1 10-18 / 06-10")
    ax.set_ylabel("BC 10-18 / 06-10")
    ax.set_title("Morning-to-later-day concentration change")
    ax.grid(True, color="0.9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def transition_ratio_column(var: str) -> str:
    return f"{var}_10_13_over_06_10"


def plot_transition_ratio_bars(summary: pd.DataFrame, var: str, out_path: Path, dpi: int) -> None:
    col = transition_ratio_column(var)
    if col not in summary.columns:
        return
    data = summary[["date_local", col]].dropna().sort_values(col)
    if data.empty:
        return
    colors = np.where(data["date_local"].isin(CANDIDATE_DAYS), "#d62728", "0.55")
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar(np.arange(len(data)), data[col], color=colors, width=0.85)
    ax.axhline(1.0, color="0.2", linewidth=0.8)
    for idx, row in data.reset_index(drop=True).iterrows():
        if row["date_local"] in CANDIDATE_DAYS:
            ax.text(idx, row[col], row["date_local"], rotation=90, ha="center", va="bottom", fontsize=7)
    meta = TRANSITION_VARIABLES[var]
    ax.set_xticks([])
    ax.set_ylabel("10-13 / 06-10 mean")
    ax.set_title(f"Morning-to-onset change: {meta['label']}")
    ax.grid(True, axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_surface_area_vs_number_transition(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x_col = "surface_area_um2_cm3_10_13_over_06_10"
    y_col = "N_total_cm3_10_13_over_06_10"
    data = summary[["date_local", x_col, y_col]].dropna()
    if data.empty:
        return
    is_candidate = data["date_local"].isin(CANDIDATE_DAYS)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.axvline(1.0, color="0.8", linewidth=1)
    ax.axhline(1.0, color="0.8", linewidth=1)
    ax.scatter(data.loc[~is_candidate, x_col], data.loc[~is_candidate, y_col], s=30, color="0.5", alpha=0.65)
    ax.scatter(
        data.loc[is_candidate, x_col],
        data.loc[is_candidate, y_col],
        marker="*",
        s=190,
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
        label="Candidate days",
    )
    for _, row in data.loc[is_candidate].iterrows():
        ax.annotate(row["date_local"], (row[x_col], row[y_col]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("SMPS surface area 10-13 / 06-10")
    ax.set_ylabel("SMPS total number 10-13 / 06-10")
    ax.set_title("Surface-area loss vs particle-number change")
    ax.grid(True, color="0.9")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_n_over_sa_transition(summary: pd.DataFrame, out_path: Path, dpi: int) -> None:
    x_col = "N_over_surface_area_10_13"
    y_col = "N_over_surface_area_10_13_over_06_10"
    data = summary[["date_local", x_col, y_col]].dropna()
    if data.empty:
        return
    is_candidate = data["date_local"].isin(CANDIDATE_DAYS)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.axhline(1.0, color="0.8", linewidth=1)
    ax.scatter(data.loc[~is_candidate, x_col], data.loc[~is_candidate, y_col], s=30, color="0.5", alpha=0.65)
    ax.scatter(
        data.loc[is_candidate, x_col],
        data.loc[is_candidate, y_col],
        marker="*",
        s=190,
        color="#d62728",
        edgecolor="black",
        linewidth=0.6,
        label="Candidate days",
    )
    for _, row in data.loc[is_candidate].iterrows():
        ax.annotate(row["date_local"], (row[x_col], row[y_col]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("N / surface area during 10-13")
    ax.set_ylabel("(N / surface area) 10-13 / 06-10")
    ax.set_title("Number-rich, surface-area-poor transition diagnostic")
    ax.grid(True, color="0.9")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_event_ratio_bars(summary: pd.DataFrame, var: str, out_path: Path, dpi: int) -> None:
    col = f"{var}_11_12_over_06_18"
    data = summary[["date_local", col]].dropna().sort_values(col)
    if data.empty:
        return
    colors = np.where(data["date_local"].isin(CANDIDATE_DAYS), "#d62728", "0.55")
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.bar(np.arange(len(data)), data[col], color=colors, width=0.85)
    ax.axhline(1.0, color="0.2", linewidth=0.8)
    for idx, row in data.reset_index(drop=True).iterrows():
        if row["date_local"] in CANDIDATE_DAYS:
            ax.text(idx, row[col], row["date_local"], rotation=90, ha="center", va="bottom", fontsize=7)
    ax.set_xticks([])
    ax.set_ylabel("11-12 / 06-18 mean")
    ax.set_title(f"Event-window ratio: {VARIABLES[var]['label']}")
    ax.grid(True, axis="y", color="0.9")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def add_interesting_flags(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    nr = out["mean_NR_PM1_06_18"]
    bc = out["mean_BC_06_18"]
    so4 = out["mean_SO4_06_18"]
    low_nr_cut = nr.quantile(0.10)
    low_so4_cut = so4.quantile(0.10)
    low_nr = nr <= low_nr_cut
    high_bc_among_low = bc >= bc.loc[low_nr].quantile(0.75) if low_nr.any() else pd.Series(False, index=out.index)
    out["flag_low_nr_high_bc"] = low_nr & high_bc_among_low
    out["flag_low_so4"] = so4 <= low_so4_cut
    return out


def rank_percentiles(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in sorted(CANDIDATE_DAYS):
        for var in ["SO4", "BC", "NR_PM1", "PM25", CS_TOTAL_COLUMN]:
            for label, prefix in [("all_day", "mean"), ("daytime_06_18", "mean")]:
                col = f"{prefix}_{var}" if label == "all_day" else f"{prefix}_{var}_06_18"
                if col not in summary.columns:
                    continue
                data = summary[["date_local", col]].dropna()
                match = data.loc[data["date_local"] == day, col]
                if match.empty:
                    rows.append({"date_local": day, "variable": var, "window": label, "value": np.nan, "rank_low_to_high": np.nan, "percentile_low_to_high": np.nan, "n_valid": len(data)})
                    continue
                value = float(match.iloc[0])
                rank = int((data[col] <= value).sum())
                percentile = 100.0 * rank / len(data) if len(data) else np.nan
                rows.append({"date_local": day, "variable": var, "window": label, "value": value, "rank_low_to_high": rank, "percentile_low_to_high": percentile, "n_valid": len(data)})
    return pd.DataFrame(rows)


def transition_ratio_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in sorted(CANDIDATE_DAYS):
        for var, meta in TRANSITION_VARIABLES.items():
            col = transition_ratio_column(var)
            if col not in summary.columns:
                continue
            data = summary[["date_local", col]].dropna()
            match = data.loc[data["date_local"] == day, col]
            if match.empty:
                rows.append(
                    {
                        "date_local": day,
                        "variable": var,
                        "label": meta["label"],
                        "ratio_10_13_over_06_10": np.nan,
                        "rank_low_to_high": np.nan,
                        "percentile_low_to_high": np.nan,
                        "n_valid": len(data),
                    }
                )
                continue
            value = float(match.iloc[0])
            rank = int((data[col] <= value).sum())
            percentile = 100.0 * rank / len(data) if len(data) else np.nan
            rows.append(
                {
                    "date_local": day,
                    "variable": var,
                    "label": meta["label"],
                    "ratio_10_13_over_06_10": value,
                    "rank_low_to_high": rank,
                    "percentile_low_to_high": percentile,
                    "n_valid": len(data),
                }
            )
    return pd.DataFrame(rows)


def make_summary_text(
    availability: pd.DataFrame, output_root: Path, ranks: pd.DataFrame, transition_ranks: pd.DataFrame
) -> str:
    n_smps_days = int(availability["has_smps"].sum())
    n_acsm = int((availability["has_smps"] & availability["has_acsm"]).sum())
    n_pm25 = int((availability["has_smps"] & availability["has_pm25"]).sum())
    n_both = int((availability["has_smps"] & availability["has_acsm"] & availability["has_pm25"]).sum())
    lines = [
        "Kigali SMPS + ACSM/BC + PM2.5 batch workflow complete.",
        f"SMPS days: {n_smps_days}",
        f"SMPS days overlapping ACSM/BC: {n_acsm}",
        f"SMPS days overlapping PM2.5: {n_pm25}",
        f"SMPS days overlapping both ACSM/BC and PM2.5: {n_both}",
        f"Outputs saved to: {output_root}",
        "",
        "Candidate-day rank summary (low-to-high percentiles):",
    ]
    if ranks.empty:
        lines.append("No candidate-day ranks available.")
    else:
        for _, row in ranks.iterrows():
            value = "NA" if not np.isfinite(row["value"]) else f"{row['value']:.3g}"
            percentile = "NA" if not np.isfinite(row["percentile_low_to_high"]) else f"{row['percentile_low_to_high']:.1f}"
            rank = "NA" if not np.isfinite(row["rank_low_to_high"]) else str(int(row["rank_low_to_high"]))
            lines.append(
                f"{row['date_local']} {row['variable']} {row['window']}: "
                f"value={value}, rank={rank}/{int(row['n_valid'])}, percentile={percentile}"
            )
    lines.extend(["", "Candidate-day transition ratio summary (10-13 / 06-10, low-to-high percentiles):"])
    if transition_ranks.empty:
        lines.append("No transition-ratio ranks available.")
    else:
        for _, row in transition_ranks.iterrows():
            value = "NA" if not np.isfinite(row["ratio_10_13_over_06_10"]) else f"{row['ratio_10_13_over_06_10']:.3g}"
            percentile = "NA" if not np.isfinite(row["percentile_low_to_high"]) else f"{row['percentile_low_to_high']:.1f}"
            rank = "NA" if not np.isfinite(row["rank_low_to_high"]) else str(int(row["rank_low_to_high"]))
            lines.append(
                f"{row['date_local']} {row['label']}: "
                f"ratio={value}, rank={rank}/{int(row['n_valid'])}, percentile={percentile}"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    apply_style()
    dirs = output_dirs(args.output)

    smps, diameter_columns, diameters, details = load_smps_wide(args.smps_wide, args.common_valid_fraction)
    cs_metrics, cs_details = calculate_cs_metrics(smps, diameter_columns, diameters, args.common_valid_fraction)
    details.update(cs_details)
    acsm = load_acsm(args.acsm)
    pm25 = load_pm25(args.pm25)
    smps_metrics = load_smps_metrics(args.smps_metrics)
    smps_metrics = add_condensation_sink_to_smps_metrics(smps_metrics, cs_metrics)
    availability = build_availability(smps, acsm, pm25)
    availability.to_csv(dirs["tables"] / "daily_overlap_availability.csv", index=False)
    cs_metrics.to_csv(dirs["tables"] / "smps_condensation_sink_scan_metrics.csv", index=False)

    all_values = smps[diameter_columns].to_numpy(dtype=float)
    vmin, vmax = finite_positive_range(all_values)
    y_edges = diameter_edges(diameters)

    contour_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for _, avail in availability.iterrows():
        day = str(avail["date_local"])
        smps_day = smps.loc[smps["date_local"] == day]
        contour_path = dirs["contour"] / f"{day}_contour.png"
        plot_contour_only(
            day,
            smps_day,
            diameter_columns,
            diameters,
            y_edges,
            vmin,
            vmax,
            args.max_gap_minutes,
            contour_path,
            args.dpi,
        )
        contour_records.append({"date_local": day, "contour_file": str(contour_path), "n_smps_scans": int(avail["n_smps_scans"])})

        has_overlap = bool(avail["has_acsm"]) or bool(avail["has_pm25"])
        if not has_overlap:
            continue
        frames = day_variable_frames(day, acsm, pm25)
        availability_row = {
            "has_acsm": bool(avail["has_acsm"]),
            "has_pm25": bool(avail["has_pm25"]),
            "n_smps_scans": int(avail["n_smps_scans"]),
            "n_acsm_hours": int(avail["n_acsm_hours"]),
            "n_pm25_hours": int(avail["n_pm25_hours"]),
        }
        summary_rows.append(summarize_day(day, acsm, pm25, smps_metrics, availability_row))
        if frames:
            plot_line_panels(day, frames, dirs["lines"] / f"{day}_daily_lines.png", args.dpi)
            plot_line_compact(day, frames, dirs["lines"] / f"{day}_daily_lines_compact.png", args.dpi)
            plot_combined(
                day,
                smps_day,
                frames,
                diameter_columns,
                diameters,
                y_edges,
                vmin,
                vmax,
                args.max_gap_minutes,
                dirs["combined"] / f"{day}_contour_plus_acsm_bc_pm25.png",
                args.dpi,
            )
            if day in CANDIDATE_DAYS:
                plot_candidate_evolution(
                    day,
                    smps_day,
                    frames,
                    smps_metrics,
                    diameter_columns,
                    diameters,
                    y_edges,
                    vmin,
                    vmax,
                    args.max_gap_minutes,
                    dirs["candidate"] / f"{day}_candidate_evolution.png",
                    args.dpi,
                )

    pd.DataFrame(contour_records).to_csv(dirs["tables"] / "daily_contour_files.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("date_local")
    if not summary.empty:
        summary = add_interesting_flags(summary)
    summary.to_csv(dirs["tables"] / "daily_summary_metrics.csv", index=False)

    scatter_pairs = [
        ("NR_PM1", "BC"),
        ("SO4", "BC"),
        ("SO4", "NR_PM1"),
        ("PM25", "BC"),
        ("PM25", "NR_PM1"),
        ("PM25", "SO4"),
    ]
    for x_var, y_var in scatter_pairs:
        plot_daytime_scatter(summary, x_var, y_var, dirs["scatter"] / scatter_filename(x_var, y_var), args.dpi)

    if not summary.empty:
        plot_flagged_scatter(
            summary,
            "NR_PM1",
            "BC",
            "flag_low_nr_high_bc",
            "Low daytime NR-PM1 with relatively high BC",
            dirs["interesting"] / "interesting_low_nrpm1_high_bc.png",
            args.dpi,
        )
        plot_flagged_scatter(
            summary,
            "NR_PM1",
            "SO4",
            "flag_low_so4",
            "Lowest 10% daytime sulfate days",
            dirs["interesting"] / "interesting_sulfate_poor_days.png",
            args.dpi,
        )
        plot_ratio_scatter(summary, dirs["interesting"] / "interesting_morning_to_late_day_change.png", args.dpi)
        plot_surface_area_vs_number_transition(
            summary,
            dirs["interesting"] / "transition_surface_area_vs_total_number_ratio.png",
            args.dpi,
        )
        plot_n_over_sa_transition(
            summary,
            dirs["interesting"] / "transition_n_over_surface_area_diagnostic.png",
            args.dpi,
        )
        for var in TRANSITION_VARIABLES:
            plot_transition_ratio_bars(
                summary,
                var,
                dirs["interesting"] / f"transition_ratio_10_13_over_06_10_{var.lower()}.png",
                args.dpi,
            )
        for stale in dirs["interesting"].glob("interesting_event_window_ratio_*.png"):
            stale.unlink()

    ranks = rank_percentiles(summary)
    ranks.to_csv(dirs["tables"] / "candidate_day_rank_percentiles.csv", index=False)
    transition_ranks = transition_ratio_ranks(summary)
    transition_ranks.to_csv(dirs["tables"] / "candidate_transition_ratio_ranks.csv", index=False)

    details.update(
        {
            "acsm_input": str(args.acsm),
            "pm25_input": str(args.pm25),
            "smps_metrics_input": str(args.smps_metrics),
            "global_color_vmin": vmin,
            "global_color_vmax": vmax,
        }
    )
    pd.DataFrame([details]).to_csv(dirs["tables"] / "run_configuration_summary.csv", index=False)

    summary_text = make_summary_text(availability, args.output, ranks, transition_ranks)
    (dirs["tables"] / "run_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
