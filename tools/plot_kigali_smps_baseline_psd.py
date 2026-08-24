#!/usr/bin/env python3
"""Baseline Kigali SMPS PSD visualizations.

This script makes descriptive PSD figures only. It intentionally does not
classify NPF events, estimate growth rates, or interpret these plots as NPF
evidence.

Primary outputs:
- kigali_overall_psd_quantile_envelope.png
- kigali_diurnal_size_range_number.png
- kigali_seasonal_psd_quantile_envelopes.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SUMMARY_DIR = Path("outputs/psd_basic_analysis")
DEFAULT_RAW_WIDE = Path("merged outputs/master_all_processed/smps_master_wide_time_ordered.csv")
KNOWN_NONSTANDARD_SOURCE = "20240219_0915-20240220_1158.txt"

SEASONS = [
    ("Short dry", "Jan-Feb", {1, 2}),
    ("Long rains", "Mar-May", {3, 4, 5}),
    ("Long dry", "Jun-Sep", {6, 7, 8, 9}),
    ("Short rains", "Oct-Dec", {10, 11, 12}),
]

SIZE_RANGE_COLUMNS = [
    ("N_10_25_cm3", "10-25 nm"),
    ("N_25_50_cm3", "25-50 nm"),
    ("N_50_100_cm3", "50-100 nm"),
    ("N_100_300_cm3", "100-300 nm"),
    ("N_total_cm3", "Total"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make baseline descriptive SMPS PSD figures for Kigali."
    )
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--raw-wide", type=Path, default=DEFAULT_RAW_WIDE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument(
        "--common-valid-fraction",
        type=float,
        default=0.95,
        help=(
            "Minimum non-missing fraction used to infer the common diameter "
            "configuration after source-file QC."
        ),
    )
    return parser.parse_args()


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.titlesize": 13,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            "savefig.dpi": 300,
        }
    )


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


def clean_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].apply(pd.to_numeric, errors="coerce")
    return out.mask(out < 0)


def read_raw_psd(
    path: Path, common_valid_fraction: float
) -> tuple[pd.DataFrame | None, list[str], np.ndarray, dict[str, object]]:
    if not path.exists():
        return None, [], np.array([], dtype=float), {"raw_available": False}

    df = pd.read_csv(path)
    diameter_columns, diameters = parse_diameter_columns(list(df.columns))
    qc_notes: dict[str, object] = {
        "raw_available": True,
        "raw_input": str(path),
        "raw_rows_before_qc": len(df),
        "excluded_source_file": KNOWN_NONSTANDARD_SOURCE,
        "excluded_rows": 0,
    }

    if "source_file" in df.columns:
        keep = df["source_file"].astype(str) != KNOWN_NONSTANDARD_SOURCE
        qc_notes["excluded_rows"] = int((~keep).sum())
        df = df.loc[keep].copy()
    else:
        qc_notes["excluded_rows"] = "not available"

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.loc[df["datetime"].notna()].copy()
    values = clean_numeric_frame(df, diameter_columns)

    # Infer the main consistent SMPS configuration after source-file QC.
    # This removes sparse bins from the nonstandard diameter configuration
    # while preserving the common diameter range used by most scans.
    valid_fraction = values.notna().mean(axis=0)
    common_mask = valid_fraction >= common_valid_fraction
    common_columns = [col for col, ok in zip(diameter_columns, common_mask) if ok]
    common_diameters = diameters[common_mask.to_numpy()]
    if not common_columns:
        raise ValueError("No common diameter bins passed the valid-fraction threshold.")

    qc_notes["raw_rows_after_qc"] = len(df)
    qc_notes["common_valid_fraction"] = common_valid_fraction
    qc_notes["common_diameter_min_nm"] = float(common_diameters.min())
    qc_notes["common_diameter_max_nm"] = float(common_diameters.max())
    qc_notes["common_diameter_bins"] = int(len(common_diameters))
    return df, common_columns, common_diameters, qc_notes


def quantile_envelope_from_raw(
    df: pd.DataFrame, diameter_columns: list[str], diameters: np.ndarray
) -> pd.DataFrame:
    values = clean_numeric_frame(df, diameter_columns)
    return pd.DataFrame(
        {
            "diameter_nm": diameters,
            "p10": values.quantile(0.10).to_numpy(dtype=float),
            "p25": values.quantile(0.25).to_numpy(dtype=float),
            "median": values.quantile(0.50).to_numpy(dtype=float),
            "p75": values.quantile(0.75).to_numpy(dtype=float),
            "p90": values.quantile(0.90).to_numpy(dtype=float),
        }
    )


def quantile_envelope_from_summary(summary_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = summary_dir / "smps_psd_diameter_bin_stats.csv"
    stats = pd.read_csv(path)

    # Fallback harmonization when source_file is unavailable in this summary:
    # keep bins represented by at least 95% of the maximum valid-bin count.
    max_count = stats["valid_count"].max()
    stats = stats.loc[stats["valid_count"] >= 0.95 * max_count].copy()
    envelope = pd.DataFrame(
        {
            "diameter_nm": stats["diameter_nm"].astype(float),
            "p10": stats["p10_dndlogdp_cm3"].astype(float),
            "p25": stats["p25_dndlogdp_cm3"].astype(float),
            "median": stats["median_dndlogdp_cm3"].astype(float),
            "p75": stats["p75_dndlogdp_cm3"].astype(float),
            "p90": stats["p90_dndlogdp_cm3"].astype(float),
        }
    )
    return envelope, {
        "overall_fallback": str(path),
        "fallback_note": "source_file unavailable; kept bins with >=95% of max valid count",
    }


def positive_limits(*series: pd.Series | np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return 1.0, 10.0
    return float(values.min() * 0.8), float(values.max() * 1.2)


def safe_fill_between(ax: plt.Axes, x: np.ndarray, low: np.ndarray, high: np.ndarray, **kwargs) -> None:
    mask = np.isfinite(x) & np.isfinite(low) & np.isfinite(high) & (low > 0) & (high > 0)
    if np.any(mask):
        ax.fill_between(x[mask], low[mask], high[mask], **kwargs)


def plot_envelope(
    envelope: pd.DataFrame,
    output_path: Path,
    title: str,
    n_label: str | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    x = envelope["diameter_nm"].to_numpy(dtype=float)
    p10 = envelope["p10"].to_numpy(dtype=float)
    p25 = envelope["p25"].to_numpy(dtype=float)
    median = envelope["median"].to_numpy(dtype=float)
    p75 = envelope["p75"].to_numpy(dtype=float)
    p90 = envelope["p90"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    safe_fill_between(ax, x, p10, p90, color="#9ecae1", alpha=0.35, label="P10-P90")
    safe_fill_between(ax, x, p25, p75, color="#3182bd", alpha=0.35, label="P25-P75")
    ax.plot(x, median, color="#08519c", linewidth=2.0, label="Median")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Particle diameter, $D_p$ (nm)")
    ax.set_ylabel(r"$dN/d\log_{10}D_p$ (cm$^{-3}$)")
    ax.set_title(title)
    ax.grid(True, which="major", color="0.85")
    ax.grid(True, which="minor", color="0.93")
    ax.legend(frameon=False)
    if n_label:
        ax.text(0.03, 0.95, n_label, transform=ax.transAxes, ha="left", va="top")
    if ylim:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def season_name(month: int) -> str:
    for name, _, months in SEASONS:
        if month in months:
            return name
    raise ValueError(f"Unexpected month: {month}")


def build_daily_median_from_raw(
    df: pd.DataFrame, diameter_columns: list[str], diameters: np.ndarray
) -> pd.DataFrame:
    working = df[["datetime", *diameter_columns]].copy()
    working["date"] = working["datetime"].dt.date.astype(str)
    values = clean_numeric_frame(working, diameter_columns)
    working.loc[:, diameter_columns] = values

    rows: list[pd.DataFrame] = []
    for date, day in working.groupby("date", sort=True):
        med = day[diameter_columns].median(axis=0, skipna=True)
        out = pd.DataFrame(
            {
                "date": date,
                "diameter_nm": diameters,
                "median_dndlogdp_cm3": med.to_numpy(dtype=float),
                "scan_count": day[diameter_columns].notna().any(axis=1).sum(),
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def load_daily_median_fallback(summary_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = summary_dir / "smps_psd_daily_median_distribution.csv"
    daily = pd.read_csv(path)
    valid_counts = daily.groupby("diameter_nm")["scan_count"].sum()
    max_count = valid_counts.max()
    keep_diameters = valid_counts[valid_counts >= 0.95 * max_count].index
    daily = daily.loc[daily["diameter_nm"].isin(keep_diameters)].copy()
    return daily, {
        "seasonal_fallback": str(path),
        "fallback_note": "source_file unavailable; kept bins with >=95% of max scan count",
    }


def seasonal_envelopes(daily: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    working = daily.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.loc[working["date"].notna()]
    working["season"] = working["date"].dt.month.map(season_name)

    envelopes: dict[str, pd.DataFrame] = {}
    n_days: dict[str, int] = {}
    for name, _, _ in SEASONS:
        season = working.loc[working["season"] == name]
        n_days[name] = int(season["date"].dt.date.nunique())
        if season.empty:
            envelopes[name] = pd.DataFrame()
            continue
        q = (
            season.groupby("diameter_nm")["median_dndlogdp_cm3"]
            .quantile([0.10, 0.25, 0.50, 0.75, 0.90])
            .unstack()
            .reset_index()
            .rename(columns={0.10: "p10", 0.25: "p25", 0.50: "median", 0.75: "p75", 0.90: "p90"})
        )
        envelopes[name] = q.sort_values("diameter_nm")
    return envelopes, n_days


def plot_seasonal(
    envelopes: dict[str, pd.DataFrame],
    n_days: dict[str, int],
    output_path: Path,
) -> None:
    all_nonempty = [env for env in envelopes.values() if not env.empty]
    if all_nonempty:
        y_limits = positive_limits(
            *[env[["p10", "p25", "median", "p75", "p90"]].to_numpy().ravel() for env in all_nonempty]
        )
        x_min = min(float(env["diameter_nm"].min()) for env in all_nonempty)
        x_max = max(float(env["diameter_nm"].max()) for env in all_nonempty)
    else:
        y_limits = (1.0, 10.0)
        x_min, x_max = 10.0, 500.0

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2), sharex=True, sharey=True)
    for ax, (name, months, _) in zip(axes.ravel(), SEASONS):
        env = envelopes[name]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(*y_limits)
        ax.grid(True, which="major", color="0.85")
        ax.grid(True, which="minor", color="0.93")
        ax.set_title(f"{name} ({months})")
        ax.text(0.03, 0.95, f"n = {n_days.get(name, 0)} days", transform=ax.transAxes, ha="left", va="top")

        if env.empty or n_days.get(name, 0) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            continue

        x = env["diameter_nm"].to_numpy(dtype=float)
        safe_fill_between(
            ax,
            x,
            env["p10"].to_numpy(dtype=float),
            env["p90"].to_numpy(dtype=float),
            color="#9ecae1",
            alpha=0.35,
            label="P10-P90",
        )
        safe_fill_between(
            ax,
            x,
            env["p25"].to_numpy(dtype=float),
            env["p75"].to_numpy(dtype=float),
            color="#3182bd",
            alpha=0.35,
            label="P25-P75",
        )
        ax.plot(x, env["median"].to_numpy(dtype=float), color="#08519c", linewidth=1.8, label="Median")

    for ax in axes[:, 0]:
        ax.set_ylabel(r"$dN/d\log_{10}D_p$ (cm$^{-3}$)")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Particle diameter, $D_p$ (nm)")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Kigali SMPS seasonal particle size distributions", y=1.03)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_scan_metrics(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "smps_psd_scan_metrics.csv"
    scan = pd.read_csv(path)
    if "source_file" in scan.columns:
        scan = scan.loc[scan["source_file"].astype(str) != KNOWN_NONSTANDARD_SOURCE].copy()
    scan["datetime"] = pd.to_datetime(scan["datetime"], errors="coerce")
    scan = scan.loc[scan["datetime"].notna()].copy()
    return scan


def plot_diurnal(scan: pd.DataFrame, output_path: Path) -> dict[str, object]:
    scan = scan.copy()
    scan["hour"] = scan["datetime"].dt.hour
    available = [(col, label) for col, label in SIZE_RANGE_COLUMNS if col in scan.columns]
    if not available:
        raise ValueError("No requested size-range concentration columns found.")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {
        "10-25 nm": "#08519c",
        "25-50 nm": "#238b45",
        "50-100 nm": "#b35806",
        "100-300 nm": "#54278f",
        "Total": "#252525",
    }
    for column, label in available:
        hourly = scan.groupby("hour")[column].agg(
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
        )
        hours = hourly.index.to_numpy(dtype=float)
        median = hourly["median"].to_numpy(dtype=float)
        p25 = hourly["p25"].to_numpy(dtype=float)
        p75 = hourly["p75"].to_numpy(dtype=float)
        color = colors.get(label, None)
        safe_fill_between(ax, hours, p25, p75, color=color, alpha=0.12)
        ax.plot(hours, median, marker="o", markersize=3.5, linewidth=1.8, color=color, label=label)

    ymin, ymax = positive_limits(scan[[col for col, _ in available]].to_numpy().ravel())
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, 23)
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xlabel("Local hour")
    ax.set_ylabel(r"Number concentration (cm$^{-3}$)")
    ax.set_title("Kigali SMPS diurnal cycle by particle size range")
    ax.grid(True, which="major", color="0.85")
    ax.grid(True, which="minor", color="0.93")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return {"diurnal_rows_after_qc": len(scan), "diurnal_columns": [label for _, label in available]}


def write_run_summary(path: Path, details: dict[str, object]) -> None:
    lines = [
        "# Kigali SMPS Baseline PSD Figure Summary",
        "",
        "These figures are descriptive baseline summaries only. They are not NPF event classifications.",
        "",
        "## Files used",
    ]
    for key in [
        "raw_input",
        "overall_source",
        "seasonal_source",
        "diurnal_source",
    ]:
        if key in details:
            lines.append(f"- {key}: `{details[key]}`")

    lines.extend(
        [
            "",
            "## Season mapping",
            "- Short dry season: January-February",
            "- Long rainy season: March-May",
            "- Long dry season: June-September",
            "- Short rainy season: October-December",
            "",
            "## QC filtering",
            f"- Known nonstandard source: `{KNOWN_NONSTANDARD_SOURCE}`",
            f"- Rows excluded from raw PSD source: {details.get('excluded_rows', 'not available')}",
            f"- Common diameter range used: {details.get('common_diameter_min_nm', 'NA')} to {details.get('common_diameter_max_nm', 'NA')} nm",
            f"- Common diameter bins used: {details.get('common_diameter_bins', 'NA')}",
            "",
            "## Coverage",
            f"- Raw rows before QC: {details.get('raw_rows_before_qc', 'NA')}",
            f"- Raw rows after QC: {details.get('raw_rows_after_qc', 'NA')}",
            f"- Diurnal scan rows after QC: {details.get('diurnal_rows_after_qc', 'NA')}",
        ]
    )
    for name, _, _ in SEASONS:
        lines.append(f"- {name}: {details.get(f'{name}_days', 0)} days")

    if "fallback_note" in details:
        lines.extend(["", "## Limitations", f"- {details['fallback_note']}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    apply_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    details: dict[str, object] = {}
    raw, diameter_columns, diameters, qc_notes = read_raw_psd(args.raw_wide, args.common_valid_fraction)
    details.update(qc_notes)

    if raw is not None:
        overall = quantile_envelope_from_raw(raw, diameter_columns, diameters)
        daily = build_daily_median_from_raw(raw, diameter_columns, diameters)
        details["overall_source"] = str(args.raw_wide)
        details["seasonal_source"] = str(args.raw_wide)
    else:
        overall, fallback_details = quantile_envelope_from_summary(args.summary_dir)
        details.update(fallback_details)
        daily, daily_fallback_details = load_daily_median_fallback(args.summary_dir)
        details.update(daily_fallback_details)
        details["overall_source"] = str(args.summary_dir / "smps_psd_diameter_bin_stats.csv")
        details["seasonal_source"] = str(args.summary_dir / "smps_psd_daily_median_distribution.csv")

    overall_ylim = positive_limits(overall[["p10", "p25", "median", "p75", "p90"]].to_numpy().ravel())
    plot_envelope(
        overall,
        args.output_dir / "kigali_overall_psd_quantile_envelope.png",
        "Kigali SMPS overall particle size distribution",
        n_label=f"n = {details.get('raw_rows_after_qc', 'summary')} scans",
        ylim=overall_ylim,
    )

    scan = load_scan_metrics(args.summary_dir)
    details["diurnal_source"] = str(args.summary_dir / "smps_psd_scan_metrics.csv")
    details.update(plot_diurnal(scan, args.output_dir / "kigali_diurnal_size_range_number.png"))

    envs, n_days = seasonal_envelopes(daily)
    for name, days in n_days.items():
        details[f"{name}_days"] = days
    plot_seasonal(envs, n_days, args.output_dir / "kigali_seasonal_psd_quantile_envelopes.png")

    write_run_summary(args.output_dir / "kigali_baseline_psd_figure_summary.md", details)
    for filename in [
        "kigali_overall_psd_quantile_envelope.png",
        "kigali_diurnal_size_range_number.png",
        "kigali_seasonal_psd_quantile_envelopes.png",
        "kigali_baseline_psd_figure_summary.md",
    ]:
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
