#!/usr/bin/env python3
"""Merge AIM SMPS text exports into analysis-ready CSV files."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


DATE_FORMATS = ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d")
TIME_FORMATS = ("%H:%M:%S", "%H:%M")


def clean_cell(value: str) -> str:
    return value.strip().strip("\ufeff")


def is_number(value: str) -> bool:
    try:
        float(clean_cell(value))
    except ValueError:
        return False
    return True


def parse_float(value: str) -> float | None:
    value = clean_cell(value)
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_datetime(date_value: str, time_value: str) -> str:
    date_value = clean_cell(date_value)
    time_value = clean_cell(time_value)
    parsed_date = None
    parsed_time = None

    for fmt in DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date_value, fmt).date()
            break
        except ValueError:
            pass

    for fmt in TIME_FORMATS:
        try:
            parsed_time = datetime.strptime(time_value, fmt).time()
            break
        except ValueError:
            pass

    if parsed_date and parsed_time:
        return datetime.combine(parsed_date, parsed_time).isoformat(sep=" ")
    return f"{date_value} {time_value}".strip()


def diameter_label(diameter: float) -> str:
    text = f"{diameter:.6g}"
    return re.sub(r"[^0-9A-Za-z.]+", "_", text)


def dlog_widths(diameters: list[float]) -> list[float]:
    if len(diameters) == 1:
        return [1.0]

    logs = [math.log10(value) for value in diameters]
    edges = [logs[0] - (logs[1] - logs[0]) / 2.0]
    edges.extend((logs[i] + logs[i + 1]) / 2.0 for i in range(len(logs) - 1))
    edges.append(logs[-1] + (logs[-1] - logs[-2]) / 2.0)
    return [edges[i + 1] - edges[i] for i in range(len(logs))]


def read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [[clean_cell(cell) for cell in row] for row in csv.reader(handle, delimiter="\t")]


def first_row_index(rows: list[list[str]], label: str) -> int:
    for index, row in enumerate(rows):
        if row and clean_cell(row[0]).lower() == label.lower():
            return index
    raise ValueError(f"Missing required AIM export row: {label}")


def parse_export(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, str]]:
    rows = read_tsv(path)
    sample_index = first_row_index(rows, "Sample #")
    date_index = first_row_index(rows, "Date")
    time_index = first_row_index(rows, "Start Time")
    diameter_index = first_row_index(rows, "Diameter Midpoint")

    sample_numbers = [cell for cell in rows[sample_index][1:] if cell != ""]
    sample_count = len(sample_numbers)
    dates = rows[date_index][1 : sample_count + 1]
    times = rows[time_index][1 : sample_count + 1]

    metadata = OrderedDict()
    for row in rows[:sample_index]:
        if len(row) >= 2 and row[0] and row[1]:
            metadata[row[0]] = row[1]

    diameter_rows: list[tuple[float, str, list[float | None]]] = []
    for row in rows[diameter_index + 1 :]:
        if not row or not is_number(row[0]):
            continue
        diameter = float(row[0])
        values = [parse_float(cell) for cell in row[1 : sample_count + 1]]
        if len(values) < sample_count:
            values.extend([None] * (sample_count - len(values)))
        diameter_rows.append((diameter, clean_cell(row[0]), values))

    diameters = [item[0] for item in diameter_rows]
    widths = dlog_widths(diameters)
    long_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for sample_pos, sample_number in enumerate(sample_numbers):
        timestamp = parse_datetime(
            dates[sample_pos] if sample_pos < len(dates) else "",
            times[sample_pos] if sample_pos < len(times) else "",
        )
        values = [row_values[sample_pos] for _, _, row_values in diameter_rows]
        valid_pairs = [(diameters[i], values[i], widths[i]) for i in range(len(values)) if values[i] is not None]

        total_number = sum(value * width for _, value, width in valid_pairs)
        negative_count = sum(1 for value in values if value is not None and value < 0)
        missing_count = sum(1 for value in values if value is None)
        max_pair = max(valid_pairs, key=lambda item: item[1], default=(None, None, None))

        flags = []
        if negative_count:
            flags.append("negative_values")
        if missing_count:
            flags.append("missing_values")
        if len(valid_pairs) < max(1, len(diameters) * 0.9):
            flags.append("few_valid_bins")

        summary_rows.append(
            {
                "source_file": path.name,
                "sample_number": sample_number,
                "datetime": timestamp,
                "n_bins": str(len(diameters)),
                "n_valid_bins": str(len(valid_pairs)),
                "n_missing_bins": str(missing_count),
                "n_negative_bins": str(negative_count),
                "total_number_concentration_approx": f"{total_number:.8g}",
                "mode_diameter_nm": "" if max_pair[0] is None else f"{max_pair[0]:.8g}",
                "mode_dndlogdp": "" if max_pair[1] is None else f"{max_pair[1]:.8g}",
                "diameter_min_nm": "" if not diameters else f"{min(diameters):.8g}",
                "diameter_max_nm": "" if not diameters else f"{max(diameters):.8g}",
                "qc_flags": ";".join(flags),
            }
        )

        for diameter, raw_diameter, row_values in diameter_rows:
            value = row_values[sample_pos]
            long_rows.append(
                {
                    "source_file": path.name,
                    "sample_number": sample_number,
                    "datetime": timestamp,
                    "diameter_nm": raw_diameter,
                    "dndlogdp": "" if value is None else f"{value:.8g}",
                }
            )

    return long_rows, summary_rows, metadata


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_wide_rows(long_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
    diameter_order: OrderedDict[str, float] = OrderedDict()
    scan_rows: OrderedDict[tuple[str, str, str], dict[str, str]] = OrderedDict()

    for row in long_rows:
        diameter = row["diameter_nm"]
        diameter_order.setdefault(diameter, float(diameter))

    diameter_columns = [
        f"dndlogdp_{diameter_label(float(diameter))}_nm"
        for diameter, _ in sorted(diameter_order.items(), key=lambda item: item[1])
    ]
    diameter_to_column = {
        diameter: f"dndlogdp_{diameter_label(float(diameter))}_nm"
        for diameter in diameter_order
    }

    for row in long_rows:
        key = (row["source_file"], row["sample_number"], row["datetime"])
        if key not in scan_rows:
            scan_rows[key] = {
                "source_file": row["source_file"],
                "sample_number": row["sample_number"],
                "datetime": row["datetime"],
            }
        scan_rows[key][diameter_to_column[row["diameter_nm"]]] = row["dndlogdp"]

    return list(scan_rows.values()), diameter_columns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge AIM SMPS tab-delimited text exports into long, wide, and scan-summary CSV files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="final files selected",
        help="Folder containing AIM .txt exports, or a single AIM .txt export.",
    )
    parser.add_argument("--out", default="merged outputs", help="Output folder for merged CSV files.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_dir():
        txt_files = sorted(input_path.glob("*.txt"))
    else:
        txt_files = [input_path]

    if not txt_files:
        raise SystemExit(f"No .txt AIM exports found in {input_path}")

    all_long_rows: list[dict[str, str]] = []
    all_summary_rows: list[dict[str, str]] = []
    metadata_rows: list[dict[str, str]] = []

    for txt_file in txt_files:
        long_rows, summary_rows, metadata = parse_export(txt_file)
        all_long_rows.extend(long_rows)
        all_summary_rows.extend(summary_rows)
        for key, value in metadata.items():
            metadata_rows.append({"source_file": txt_file.name, "field": key, "value": value})

    output_dir = Path(args.out)
    write_csv(
        output_dir / "smps_merged_long.csv",
        all_long_rows,
        ["source_file", "sample_number", "datetime", "diameter_nm", "dndlogdp"],
    )
    wide_rows, diameter_columns = build_wide_rows(all_long_rows)
    write_csv(
        output_dir / "smps_merged_wide.csv",
        wide_rows,
        ["source_file", "sample_number", "datetime", *diameter_columns],
    )
    write_csv(
        output_dir / "smps_scan_summary.csv",
        all_summary_rows,
        [
            "source_file",
            "sample_number",
            "datetime",
            "n_bins",
            "n_valid_bins",
            "n_missing_bins",
            "n_negative_bins",
            "total_number_concentration_approx",
            "mode_diameter_nm",
            "mode_dndlogdp",
            "diameter_min_nm",
            "diameter_max_nm",
            "qc_flags",
        ],
    )
    write_csv(output_dir / "smps_export_metadata.csv", metadata_rows, ["source_file", "field", "value"])

    print(f"Merged {len(txt_files)} export file(s).")
    print(f"Scans: {len(wide_rows)}")
    print(f"Long rows: {len(all_long_rows)}")
    print(f"Output: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
