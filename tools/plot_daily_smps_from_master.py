#!/usr/bin/env python3
"""Create time-ordered SMPS master files and one 24-hour contour plot per day."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DIAMETER_RE = re.compile(r"^dndlogdp_(?P<diameter>[0-9.]+)_nm$")


def parse_datetime(value: str) -> datetime:
    return datetime.strptime(value, DATETIME_FORMAT)


def diameter_from_column(column: str) -> float:
    match = DIAMETER_RE.match(column)
    if not match:
        raise ValueError(f"Could not parse diameter from column name: {column}")
    return float(match.group("diameter"))


def read_wide(path: Path) -> tuple[list[str], list[dict[str, str]], list[str], np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        fieldnames = reader.fieldnames
        rows = list(reader)

    rows.sort(key=lambda row: parse_datetime(row["datetime"]))
    diameter_columns = [column for column in fieldnames if column.startswith("dndlogdp_")]
    diameters = np.array([diameter_from_column(column) for column in diameter_columns], dtype=float)
    order = np.argsort(diameters)
    diameter_columns = [diameter_columns[index] for index in order]
    diameters = diameters[order]
    return fieldnames, rows, diameter_columns, diameters


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_values_key(row: dict[str, str], diameter_columns: list[str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in diameter_columns)


def collapse_identical_duplicate_times(
    rows: list[dict[str, str]], diameter_columns: list[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    collapsed: list[dict[str, str]] = []
    report: list[dict[str, str]] = []
    index = 0
    while index < len(rows):
        current_datetime = rows[index]["datetime"]
        group = [rows[index]]
        index += 1
        while index < len(rows) and rows[index]["datetime"] == current_datetime:
            group.append(rows[index])
            index += 1

        if len(group) == 1:
            collapsed.append(group[0])
            continue

        first_key = row_values_key(group[0], diameter_columns)
        identical = all(row_values_key(row, diameter_columns) == first_key for row in group[1:])
        action = "collapsed_identical" if identical else "kept_conflicting"
        report.append(
            {
                "datetime": current_datetime,
                "duplicate_count": str(len(group)),
                "action": action,
                "source_files": ";".join(sorted({row["source_file"] for row in group})),
                "sample_numbers": ";".join(row["sample_number"] for row in group),
            }
        )
        if identical:
            collapsed.append(group[0])
        else:
            collapsed.extend(group)
    return collapsed, report


def sort_csv_by_datetime(input_path: Path, output_path: Path) -> None:
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {input_path}")
        rows = list(reader)
        fieldnames = reader.fieldnames
    rows.sort(key=lambda row: parse_datetime(row["datetime"]))
    write_rows(output_path, fieldnames, rows)


def values_for_rows(rows: list[dict[str, str]], diameter_columns: list[str]) -> np.ndarray:
    values = np.empty((len(rows), len(diameter_columns)), dtype=float)
    values.fill(np.nan)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(diameter_columns):
            value = row.get(column, "")
            if value:
                try:
                    values[row_index, column_index] = float(value)
                except ValueError:
                    pass
    values[values <= 0] = np.nan
    return values


def finite_positive_range(values: np.ndarray) -> tuple[float, float]:
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return 1.0, 10.0
    vmin = max(1.0, float(np.nanpercentile(positive, 5)))
    vmax = float(np.nanpercentile(positive, 99))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def diameter_edges(diameters: np.ndarray) -> np.ndarray:
    logs = np.log10(diameters)
    if len(logs) == 1:
        half_width = 0.05
        edges = np.array([logs[0] - half_width, logs[0] + half_width])
    else:
        edges = np.empty(len(logs) + 1)
        edges[1:-1] = (logs[:-1] + logs[1:]) / 2.0
        edges[0] = logs[0] - (logs[1] - logs[0]) / 2.0
        edges[-1] = logs[-1] + (logs[-1] - logs[-2]) / 2.0
    return 10**edges


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


def coverage_stem(day_rows: list[dict[str, str]], day: datetime) -> tuple[str, datetime, datetime]:
    times = [parse_datetime(row["datetime"]) for row in day_rows]
    first = min(times)
    last = max(times)
    full_start = datetime.combine(day.date(), time.min) + timedelta(minutes=15)
    full_end = datetime.combine(day.date(), time(23, 45))
    if first <= full_start and last >= full_end:
        return f"{day:%Y%m%d}", first, last
    return f"{day:%Y%m%d}_{first:%H%M}-{last:%H%M}", first, last


def largest_gap_minutes(times: list[datetime]) -> float:
    if len(times) < 2:
        return 0.0
    return max((times[i] - times[i - 1]).total_seconds() / 60.0 for i in range(1, len(times)))


def plot_day(
    day: datetime,
    day_rows: list[dict[str, str]],
    diameter_columns: list[str],
    diameters: np.ndarray,
    y_edges: np.ndarray,
    out_dir: Path,
    vmin: float,
    vmax: float,
    max_gap_minutes: float,
) -> dict[str, str]:
    day_rows.sort(key=lambda row: parse_datetime(row["datetime"]))
    times = [parse_datetime(row["datetime"]) for row in day_rows]
    values = values_for_rows(day_rows, diameter_columns)
    stem, first, last = coverage_stem(day_rows, day)
    day_start = datetime.combine(day.date(), time.min)
    day_end = day_start + timedelta(days=1)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    mesh = None
    for start, end in continuous_segments(times, timedelta(minutes=max_gap_minutes)):
        segment_times = times[start:end]
        segment_values = values[start:end, :]
        if not segment_times:
            continue
        mesh = ax.pcolormesh(
            time_edges(segment_times, day_start, day_end),
            y_edges,
            segment_values.T,
            shading="flat",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            cmap="turbo",
        )

    if mesh is None:
        raise ValueError(f"No plottable scans for {day:%Y-%m-%d}")

    ax.set_title(f"Kigali SMPS dN/dlogDp - {day:%Y-%m-%d}")
    ax.set_xlabel("Local time")
    ax.set_ylabel("Particle diameter, Dp [nm]")
    ax.set_yscale("log")
    ax.set_ylim(max(8, float(np.nanmin(diameters))), min(700, float(np.nanmax(diameters))))
    ax.set_xlim(mdates.date2num(day_start), mdates.date2num(day_end))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, axis="x", color="0.82", linewidth=0.6)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("dN/dlogDp [cm^-3]")
    fig.tight_layout()

    out_path = out_dir / f"{stem}_dNdlogDp_contour.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    source_files = sorted({row["source_file"] for row in day_rows})
    return {
        "date": f"{day:%Y-%m-%d}",
        "scan_count": str(len(day_rows)),
        "first_datetime": first.strftime(DATETIME_FORMAT),
        "last_datetime": last.strftime(DATETIME_FORMAT),
        "largest_gap_minutes": f"{largest_gap_minutes(times):.2f}",
        "source_file_count": str(len(source_files)),
        "source_files": ";".join(source_files),
        "plot_file": str(out_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write sorted SMPS master CSVs and daily 24-hour dN/dlogDp contour plots."
    )
    parser.add_argument(
        "--wide",
        default="merged outputs/master_all_processed/smps_merged_wide.csv",
        help="Merged wide CSV from merge_aim_txt_exports.py.",
    )
    parser.add_argument(
        "--summary",
        default="merged outputs/master_all_processed/smps_scan_summary.csv",
        help="Merged scan summary CSV from merge_aim_txt_exports.py.",
    )
    parser.add_argument(
        "--out",
        default="merged outputs/master_all_processed",
        help="Output folder for sorted master files and plots.",
    )
    parser.add_argument(
        "--max-gap-minutes",
        type=float,
        default=10.0,
        help="Break the contour into blank gaps when consecutive scans are farther apart than this.",
    )
    args = parser.parse_args()

    wide_path = Path(args.wide)
    summary_path = Path(args.summary)
    out_dir = Path(args.out)
    plot_dir = out_dir / "daily_24h_contours"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows, diameter_columns, diameters = read_wide(wide_path)
    all_scans_wide_path = out_dir / "smps_master_wide_all_scans_time_ordered.csv"
    write_rows(all_scans_wide_path, fieldnames, rows)
    rows, duplicate_report = collapse_identical_duplicate_times(rows, diameter_columns)
    sorted_wide_path = out_dir / "smps_master_wide_time_ordered.csv"
    write_rows(sorted_wide_path, fieldnames, rows)
    duplicate_report_path = out_dir / "duplicate_timestamp_report.csv"
    write_rows(
        duplicate_report_path,
        ["datetime", "duplicate_count", "action", "source_files", "sample_numbers"],
        duplicate_report,
    )
    sorted_summary_path = out_dir / "smps_master_scan_summary_time_ordered.csv"
    sort_csv_by_datetime(summary_path, sorted_summary_path)

    all_values = values_for_rows(rows, diameter_columns)
    vmin, vmax = finite_positive_range(all_values)
    y_edges = diameter_edges(diameters)

    rows_by_day: dict[datetime, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        timestamp = parse_datetime(row["datetime"])
        day = datetime.combine(timestamp.date(), time.min)
        rows_by_day[day].append(row)

    summaries = []
    for day in sorted(rows_by_day):
        summaries.append(
            plot_day(
                day,
                rows_by_day[day],
                diameter_columns,
                diameters,
                y_edges,
                plot_dir,
                vmin,
                vmax,
                args.max_gap_minutes,
            )
        )

    daily_summary_path = out_dir / "daily_24h_contour_summary.csv"
    write_rows(
        daily_summary_path,
        [
            "date",
            "scan_count",
            "first_datetime",
            "last_datetime",
            "largest_gap_minutes",
            "source_file_count",
            "source_files",
            "plot_file",
        ],
        summaries,
    )

    print(f"Sorted all-scans wide master: {all_scans_wide_path.resolve()}")
    print(f"Sorted de-duplicated wide master: {sorted_wide_path.resolve()}")
    print(f"Sorted scan summary: {sorted_summary_path.resolve()}")
    print(f"Duplicate timestamp report: {duplicate_report_path.resolve()}")
    print(f"Daily plot folder: {plot_dir.resolve()}")
    print(f"Daily plot summary: {daily_summary_path.resolve()}")
    print(f"Days plotted: {len(summaries)}")
    print(f"Scans plotted: {len(rows)}")
    print(f"Duplicate timestamp groups reported: {len(duplicate_report)}")
    print(f"Global color scale: {vmin:.4g} to {vmax:.4g} dN/dlogDp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
