#!/usr/bin/env python3
"""Plot AIM SMPS dN/dlogDp contour and average PSD for one exported .txt file."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, time
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DATE_FORMATS = ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d")
TIME_FORMATS = ("%H:%M:%S", "%H:%M")


def clean_cell(value: str) -> str:
    return value.strip().strip("\ufeff")


def read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [[clean_cell(cell) for cell in row] for row in csv.reader(handle, delimiter="\t")]


def find_row_index(rows: list[list[str]], label: str) -> int:
    for index, row in enumerate(rows):
        if row and clean_cell(row[0]).lower() == label.lower():
            return index
    raise ValueError(f"Missing required row: {label}")


def parse_datetime(date_value: str, time_value: str) -> datetime:
    parsed_date = None
    parsed_time = None
    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(clean_cell(date_value), fmt).date()
            break
        except ValueError:
            pass
    for fmt in TIME_FORMATS:
        try:
            parsed_time = datetime.strptime(clean_cell(time_value), fmt).time()
            break
        except ValueError:
            pass
    if parsed_date is None or parsed_time is None:
        raise ValueError(f"Could not parse date/time: {date_value} {time_value}")
    return datetime.combine(parsed_date, parsed_time)


def parse_float(value: str) -> float:
    value = clean_cell(value)
    return np.nan if value == "" else float(value)


def load_aim_export(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    rows = read_tsv(path)
    sample_index = find_row_index(rows, "Sample #")
    date_index = find_row_index(rows, "Date")
    time_index = find_row_index(rows, "Start Time")
    diameter_index = find_row_index(rows, "Diameter Midpoint")

    sample_count = len([cell for cell in rows[sample_index][1:] if cell])
    times = np.array(
        [
            parse_datetime(date_value, time_value)
            for date_value, time_value in zip(
                rows[date_index][1 : sample_count + 1],
                rows[time_index][1 : sample_count + 1],
            )
            if date_value and time_value
        ]
    )

    diameters: list[float] = []
    columns: list[list[float]] = []
    for row in rows[diameter_index + 1 :]:
        if not row or not row[0]:
            continue
        try:
            diameter = float(row[0])
        except ValueError:
            continue
        values = [parse_float(cell) for cell in row[1 : sample_count + 1]]
        if len(values) < sample_count:
            values.extend([np.nan] * (sample_count - len(values)))
        diameters.append(diameter)
        columns.append(values)

    z = np.array(columns, dtype=float).T
    if z.shape[0] != len(times):
        z = z[: len(times), :]

    metadata = {}
    for row in rows[:sample_index]:
        if len(row) >= 2 and row[0]:
            metadata[row[0]] = row[1]
    return times, np.array(diameters, dtype=float), z, metadata


def finite_positive_range(values: np.ndarray) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0, 10.0
    vmin = max(1.0, float(np.nanpercentile(positive, 5)))
    vmax = float(np.nanpercentile(positive, 99))
    if vmax <= vmin:
        vmax = vmin * 10
    return vmin, vmax


def parse_date_bound(value: str, is_end: bool = False) -> datetime:
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            parsed = datetime.strptime(value, fmt)
            if is_end and parsed.time() == time.min:
                return datetime.combine(parsed.date(), time(23, 59, 59, 999999))
            return parsed
        except ValueError:
            pass
    raise ValueError(f"Could not parse date: {value}")


def load_inputs(input_path: Path, start: datetime | None, end: datetime | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str], str]:
    paths = sorted(input_path.glob("*.txt")) if input_path.is_dir() else [input_path]
    all_times: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    reference_diameters: np.ndarray | None = None
    metadata: dict[str, str] = {}
    used_files: list[str] = []

    for path in paths:
        times, diameters, dndlogdp, file_metadata = load_aim_export(path)
        mask = np.ones(len(times), dtype=bool)
        if start is not None:
            mask &= times >= start
        if end is not None:
            mask &= times <= end
        if not np.any(mask):
            continue

        if reference_diameters is None:
            reference_diameters = diameters
            metadata = file_metadata
        elif len(reference_diameters) != len(diameters) or not np.allclose(reference_diameters, diameters):
            raise ValueError(f"Diameter bins differ in {path.name}; cannot combine with other exports.")

        all_times.append(times[mask])
        all_values.append(dndlogdp[mask, :])
        used_files.append(path.name)

    if reference_diameters is None or not all_times:
        raise ValueError("No scans found for the requested input/date range.")

    combined_times = np.concatenate(all_times)
    combined_values = np.vstack(all_values)
    order = np.argsort(combined_times)
    label = input_path.stem if input_path.is_file() else f"{used_files[0]}_to_{used_files[-1]}"
    return combined_times[order], reference_diameters, combined_values[order, :], metadata, label


def continuous_segments(times: np.ndarray, max_gap_seconds: int = 600) -> list[np.ndarray]:
    if len(times) == 0:
        return []
    splits = [0]
    for index in range(1, len(times)):
        if (times[index] - times[index - 1]).total_seconds() > max_gap_seconds:
            splits.append(index)
    splits.append(len(times))
    return [np.arange(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]


def plot_dataset(
    times: np.ndarray,
    diameters: np.ndarray,
    dndlogdp: np.ndarray,
    metadata: dict[str, str],
    stem: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    z = dndlogdp.copy()
    z[z <= 0] = np.nan
    vmin, vmax = finite_positive_range(z)
    time_index = np.array([mdates.date2num(item) for item in times])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    mesh = None
    for segment in continuous_segments(times):
        if len(segment) == 0:
            continue
        mesh = ax.pcolormesh(
            time_index[segment],
            diameters,
            z[segment, :].T,
            shading="auto",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="turbo",
        )
    if mesh is None:
        raise ValueError("No data available for contour plot.")
    ax.set_yscale("log")
    ax.set_ylim(max(8, np.nanmin(diameters)), min(700, np.nanmax(diameters)))
    ax.set_ylabel("Particle diameter, Dp [nm]")
    ax.set_xlabel("Time")
    ax.set_title(f"Kigali SMPS dN/dlogDp - {stem}")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("dN/dlogDp [cm^-3]")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
    if (times.max() - times.min()).total_seconds() > 36 * 3600:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlim(time_index.min(), time_index.max())
    fig.autofmt_xdate()
    fig.tight_layout()

    contour_path = out_dir / f"{stem}_dNdlogDp_contour.png"
    fig.savefig(contour_path, dpi=300)
    plt.close(fig)

    median = np.nanmedian(z, axis=0)
    p25 = np.nanpercentile(z, 25, axis=0)
    p75 = np.nanpercentile(z, 75, axis=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(diameters, p25, p75, color="#9ecae1", alpha=0.45, label="25-75%")
    ax.plot(diameters, median, color="#08519c", linewidth=2, label="Median")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Particle diameter, Dp [nm]")
    ax.set_ylabel("dN/dlogDp [cm^-3]")
    ax.set_title(f"Kigali SMPS median PSD - {stem}")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    psd_path = out_dir / f"{stem}_median_PSD.png"
    fig.savefig(psd_path, dpi=300)
    plt.close(fig)

    units = metadata.get("Units", "unknown")
    weight = metadata.get("Weight", "unknown")
    print(f"Loaded {stem}: {len(times)} scans, {len(diameters)} bins, Units={units}, Weight={weight}")
    print(f"Saved contour: {contour_path.resolve()}")
    print(f"Saved median PSD: {psd_path.resolve()}")
    return contour_path, psd_path


def plot_contour(path: Path, out_dir: Path) -> tuple[Path, Path]:
    times, diameters, dndlogdp, metadata = load_aim_export(path)
    return plot_dataset(times, diameters, dndlogdp, metadata, path.stem, out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot AIM SMPS .txt export(s) as dN/dlogDp contour.")
    parser.add_argument("input", help="AIM .txt export or folder of AIM .txt exports to plot")
    parser.add_argument("--out", default="figures/smps_contour_tests", help="Output folder for PNG plots")
    parser.add_argument("--start", help="Optional start date, e.g. 20240412 or 2024-04-12")
    parser.add_argument("--end", help="Optional end date, inclusive, e.g. 20240419 or 2024-04-19")
    parser.add_argument("--label", help="Optional output filename/title label")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.start or args.end or input_path.is_dir():
        start = parse_date_bound(args.start) if args.start else None
        end = parse_date_bound(args.end, is_end=True) if args.end else None
        times, diameters, dndlogdp, metadata, label = load_inputs(input_path, start, end)
        plot_dataset(times, diameters, dndlogdp, metadata, args.label or label, Path(args.out))
    else:
        plot_contour(input_path, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
