#!/usr/bin/env python3
"""Simple PSD-only statistics for merged Kigali SMPS data.

Input values are interpreted as dN/dlog10(Dp) in cm^-3. This script avoids
NPF event classification, growth rates, formation rates, and condensation-sink
assumptions. It only calculates quantities that follow directly from the PSD.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("merged outputs/master_all_processed/smps_master_wide_time_ordered.csv")
DEFAULT_OUTPUT = Path("outputs/psd_basic_analysis")


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


def dlog10_widths(diameters_nm: np.ndarray) -> np.ndarray:
    logs = np.log10(diameters_nm)
    if len(logs) == 1:
        return np.array([1.0])
    edges = np.empty(len(logs) + 1, dtype=float)
    edges[1:-1] = (logs[:-1] + logs[1:]) / 2.0
    edges[0] = logs[0] - (logs[1] - logs[0]) / 2.0
    edges[-1] = logs[-1] + (logs[-1] - logs[-2]) / 2.0
    return np.diff(edges)


def weighted_percentile(x: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    ok = np.isfinite(x) & np.isfinite(weights) & (weights > 0)
    if not np.any(ok):
        return np.nan
    x = x[ok]
    weights = weights[ok]
    order = np.argsort(x)
    x = x[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = percentile / 100.0 * cumulative[-1]
    return float(x[np.searchsorted(cumulative, cutoff, side="left")])


def weighted_diameter_stats(diameters_nm: np.ndarray, number_by_bin: np.ndarray) -> tuple[float, float, float, float]:
    weights = np.where(np.isfinite(number_by_bin) & (number_by_bin > 0), number_by_bin, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return np.nan, np.nan, np.nan, np.nan
    arithmetic_mean = float(np.sum(diameters_nm * weights) / total)
    median = weighted_percentile(diameters_nm, weights, 50.0)
    ln_d = np.log(diameters_nm)
    mean_ln = float(np.sum(ln_d * weights) / total)
    variance_ln = float(np.sum(weights * (ln_d - mean_ln) ** 2) / total)
    geometric_mean = math.exp(mean_ln)
    geometric_sd = math.exp(math.sqrt(max(variance_ln, 0.0)))
    return arithmetic_mean, median, geometric_mean, geometric_sd


def add_size_ranges(out: pd.DataFrame, dn: np.ndarray, diameters_nm: np.ndarray) -> None:
    ranges = {
        "N_total_cm3": (diameters_nm >= diameters_nm.min()) & (diameters_nm <= diameters_nm.max()),
        "N_10_25_cm3": (diameters_nm >= 10.0) & (diameters_nm < 25.0),
        "N_25_50_cm3": (diameters_nm >= 25.0) & (diameters_nm < 50.0),
        "N_50_100_cm3": (diameters_nm >= 50.0) & (diameters_nm < 100.0),
        "N_100_300_cm3": (diameters_nm >= 100.0) & (diameters_nm < 300.0),
        "N_300_max_cm3": (diameters_nm >= 300.0) & (diameters_nm <= diameters_nm.max()),
    }
    for name, mask in ranges.items():
        out[name] = dn[:, mask].sum(axis=1)


def build_scan_metrics(df: pd.DataFrame, diameter_columns: list[str], diameters_nm: np.ndarray) -> pd.DataFrame:
    values = df[diameter_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    values = np.where(values < 0, np.nan, values)
    widths = dlog10_widths(diameters_nm)
    dn = np.where(np.isfinite(values), values * widths[None, :], 0.0)

    out = df[["source_file", "sample_number", "datetime"]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    valid = out["datetime"].notna()
    out = out.loc[valid].reset_index(drop=True)
    values = values[valid.to_numpy(), :]
    dn = dn[valid.to_numpy(), :]
    out["date"] = out["datetime"].dt.date.astype(str)

    add_size_ranges(out, dn, diameters_nm)

    positive_values = np.where(np.isfinite(values), values, -np.inf)
    mode_idx = np.argmax(positive_values, axis=1)
    all_missing = ~np.isfinite(values).any(axis=1)
    out["mode_diameter_nm"] = diameters_nm[mode_idx]
    out["mode_dndlogdp_cm3"] = np.nanmax(positive_values, axis=1)
    out.loc[all_missing, ["mode_diameter_nm", "mode_dndlogdp_cm3"]] = np.nan

    stats = [weighted_diameter_stats(diameters_nm, row) for row in dn]
    out["number_mean_diameter_nm"] = [item[0] for item in stats]
    out["number_median_diameter_nm"] = [item[1] for item in stats]
    out["geometric_mean_diameter_nm"] = [item[2] for item in stats]
    out["geometric_sd"] = [item[3] for item in stats]

    diam_um = diameters_nm * 1.0e-3
    out["surface_area_um2_cm3"] = np.sum(math.pi * (diam_um[None, :] ** 2) * dn, axis=1)
    out["volume_um3_cm3"] = np.sum((math.pi / 6.0) * (diam_um[None, :] ** 3) * dn, axis=1)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def summarize_metrics_by_day(scan_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "N_total_cm3",
        "N_10_25_cm3",
        "N_25_50_cm3",
        "N_50_100_cm3",
        "N_100_300_cm3",
        "N_300_max_cm3",
        "mode_diameter_nm",
        "number_mean_diameter_nm",
        "number_median_diameter_nm",
        "geometric_mean_diameter_nm",
        "geometric_sd",
        "surface_area_um2_cm3",
        "volume_um3_cm3",
    ]
    rows: list[dict[str, object]] = []
    for date, day in scan_metrics.groupby("date", sort=True):
        times = day["datetime"].sort_values()
        intervals_min = times.diff().dt.total_seconds().dropna() / 60.0
        row: dict[str, object] = {
            "date": date,
            "scan_count": len(day),
            "first_datetime": times.iloc[0],
            "last_datetime": times.iloc[-1],
            "coverage_hours": (times.iloc[-1] - times.iloc[0]).total_seconds() / 3600.0 if len(times) > 1 else 0.0,
            "median_scan_interval_min": float(intervals_min.median()) if len(intervals_min) else np.nan,
            "largest_gap_min": float(intervals_min.max()) if len(intervals_min) else np.nan,
        }
        for metric in metric_columns:
            s = day[metric]
            row[f"{metric}_mean"] = float(s.mean())
            row[f"{metric}_median"] = float(s.median())
            row[f"{metric}_std"] = float(s.std())
            row[f"{metric}_p25"] = float(s.quantile(0.25))
            row[f"{metric}_p75"] = float(s.quantile(0.75))
            row[f"{metric}_min"] = float(s.min())
            row[f"{metric}_max"] = float(s.max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_diameter_bins(df: pd.DataFrame, diameter_columns: list[str], diameters_nm: np.ndarray) -> pd.DataFrame:
    rows = []
    for column, diameter in zip(diameter_columns, diameters_nm):
        s = pd.to_numeric(df[column], errors="coerce")
        s = s[s >= 0]
        rows.append(
            {
                "diameter_nm": diameter,
                "valid_count": int(s.count()),
                "mean_dndlogdp_cm3": float(s.mean()),
                "median_dndlogdp_cm3": float(s.median()),
                "std_dndlogdp_cm3": float(s.std()),
                "min_dndlogdp_cm3": float(s.min()),
                "p10_dndlogdp_cm3": float(s.quantile(0.10)),
                "p25_dndlogdp_cm3": float(s.quantile(0.25)),
                "p75_dndlogdp_cm3": float(s.quantile(0.75)),
                "p90_dndlogdp_cm3": float(s.quantile(0.90)),
                "p95_dndlogdp_cm3": float(s.quantile(0.95)),
                "max_dndlogdp_cm3": float(s.max()),
            }
        )
    return pd.DataFrame(rows)


def daily_median_psd(df: pd.DataFrame, diameter_columns: list[str], diameters_nm: np.ndarray) -> pd.DataFrame:
    working = df[["datetime", *diameter_columns]].copy()
    working["datetime"] = pd.to_datetime(working["datetime"], errors="coerce")
    working = working[working["datetime"].notna()]
    working["date"] = working["datetime"].dt.date.astype(str)
    rows = []
    for date, day in working.groupby("date", sort=True):
        for column, diameter in zip(diameter_columns, diameters_nm):
            s = pd.to_numeric(day[column], errors="coerce")
            s = s[s >= 0]
            rows.append(
                {
                    "date": date,
                    "diameter_nm": diameter,
                    "median_dndlogdp_cm3": float(s.median()),
                    "mean_dndlogdp_cm3": float(s.mean()),
                    "scan_count": int(s.count()),
                }
            )
    return pd.DataFrame(rows)


def overall_summary(scan_metrics: pd.DataFrame, diameters_nm: np.ndarray) -> pd.DataFrame:
    items = [
        ("input_scans_used", len(scan_metrics), "Valid-datetime de-duplicated scans from master table"),
        ("first_datetime", scan_metrics["datetime"].min(), ""),
        ("last_datetime", scan_metrics["datetime"].max(), ""),
        ("calendar_days", scan_metrics["date"].nunique(), ""),
        ("diameter_min_nm", float(diameters_nm.min()), ""),
        ("diameter_max_nm", float(diameters_nm.max()), ""),
        ("median_N_total_cm3", scan_metrics["N_total_cm3"].median(), ""),
        ("mean_N_total_cm3", scan_metrics["N_total_cm3"].mean(), ""),
        ("median_N_10_25_cm3", scan_metrics["N_10_25_cm3"].median(), "Size-range concentration only; not a nucleation rate"),
        ("median_N_25_50_cm3", scan_metrics["N_25_50_cm3"].median(), ""),
        ("median_N_50_100_cm3", scan_metrics["N_50_100_cm3"].median(), ""),
        ("median_N_100_300_cm3", scan_metrics["N_100_300_cm3"].median(), ""),
        ("median_N_300_max_cm3", scan_metrics["N_300_max_cm3"].median(), ""),
        ("median_mode_diameter_nm", scan_metrics["mode_diameter_nm"].median(), ""),
        ("median_number_median_diameter_nm", scan_metrics["number_median_diameter_nm"].median(), ""),
        ("median_geometric_mean_diameter_nm", scan_metrics["geometric_mean_diameter_nm"].median(), ""),
        ("median_surface_area_um2_cm3", scan_metrics["surface_area_um2_cm3"].median(), ""),
        ("median_volume_um3_cm3", scan_metrics["volume_um3_cm3"].median(), ""),
    ]
    return pd.DataFrame(items, columns=["item", "value", "note"])


def write_methods(path: Path, diameters_nm: np.ndarray) -> None:
    text = f"""# Basic SMPS PSD Calculations

This is a PSD-only analysis. It intentionally excludes NPF event classification,
growth rate, formation rate, and condensation sink.

Input
- Source: `merged outputs/master_all_processed/smps_master_wide_time_ordered.csv`
- Rows: de-duplicated SMPS scans.
- Size bins: {diameters_nm.min():g}-{diameters_nm.max():g} nm.
- Input units assumed: dN/dlog10(Dp), cm^-3.
- Negative values are treated as missing.

Formulas
- Log-bin width: Delta log10(Dp_i) = log10(edge_hi_i) - log10(edge_lo_i).
- Bin number concentration: N_i = (dN/dlog10Dp)_i * Delta log10(Dp_i).
- Size-range concentration: N_D1-D2 = sum(N_i) for bins with D1 <= Dp_i < D2.
- Total number concentration: N_total = sum(N_i) across all SMPS bins.
- Mode diameter: Dp bin midpoint where dN/dlog10Dp is largest.
- Number-weighted mean diameter: mean Dp = sum(Dp_i * N_i) / sum(N_i).
- Number-weighted median diameter: Dp where cumulative sum(N_i) reaches 50% of total number.
- Geometric mean diameter: Dg = exp[sum(N_i * ln(Dp_i)) / sum(N_i)].
- Geometric standard deviation: GSD = exp(sqrt(sum(N_i * [ln(Dp_i)-ln(Dg)]^2) / sum(N_i))).
- Surface area concentration: S = sum(pi * Dp_i^2 * N_i), with Dp in micrometers; units um^2 cm^-3.
- Volume concentration: V = sum((pi/6) * Dp_i^3 * N_i), with Dp in micrometers; units um^3 cm^-3.

Outputs
- `smps_psd_scan_metrics.csv`: scan-by-scan PSD metrics.
- `smps_psd_daily_stats.csv`: daily mean, median, standard deviation, quartiles, min, and max for scan metrics.
- `smps_psd_diameter_bin_stats.csv`: whole-period stats for each SMPS diameter bin.
- `smps_psd_daily_median_distribution.csv`: daily mean and median PSD by diameter bin.
- `smps_psd_overall_summary.csv`: compact overall summary.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    diameter_columns, diameters_nm = parse_diameter_columns(list(df.columns))
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df[df["datetime"].notna()].sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    scan_metrics = build_scan_metrics(df, diameter_columns, diameters_nm)
    daily_stats = summarize_metrics_by_day(scan_metrics)
    bin_stats = summarize_diameter_bins(df, diameter_columns, diameters_nm)
    daily_psd = daily_median_psd(df, diameter_columns, diameters_nm)
    summary = overall_summary(scan_metrics, diameters_nm)

    scan_metrics.to_csv(args.out / "smps_psd_scan_metrics.csv", index=False)
    daily_stats.to_csv(args.out / "smps_psd_daily_stats.csv", index=False)
    bin_stats.to_csv(args.out / "smps_psd_diameter_bin_stats.csv", index=False)
    daily_psd.to_csv(args.out / "smps_psd_daily_median_distribution.csv", index=False)
    summary.to_csv(args.out / "smps_psd_overall_summary.csv", index=False)
    write_methods(args.out / "SMPS_PSD_basic_methods.md", diameters_nm)

    print(f"wrote {args.out}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
