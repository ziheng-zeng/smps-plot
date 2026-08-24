#!/usr/bin/env python3
"""Rename AIM SMPS text exports using their first and last scan timestamps."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


DATE_FORMATS = ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d")
TIME_FORMATS = ("%H:%M:%S", "%H:%M")


def clean_cell(value: str) -> str:
    return value.strip().strip("\ufeff")


def read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [[clean_cell(cell) for cell in row] for row in csv.reader(handle, delimiter="\t")]


def find_row(rows: list[list[str]], label: str) -> list[str]:
    for row in rows:
        if row and clean_cell(row[0]).lower() == label.lower():
            return row
    raise ValueError(f"Missing required row: {label}")


def parse_datetime(date_value: str, time_value: str) -> datetime:
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

    if parsed_date is None or parsed_time is None:
        raise ValueError(f"Could not parse date/time: {date_value} {time_value}")
    return datetime.combine(parsed_date, parsed_time)


def scan_time_range(path: Path) -> tuple[datetime, datetime, int]:
    rows = read_tsv(path)
    sample_row = find_row(rows, "Sample #")
    date_row = find_row(rows, "Date")
    time_row = find_row(rows, "Start Time")

    sample_count = len([cell for cell in sample_row[1:] if cell])
    datetimes = []
    for date_value, time_value in zip(date_row[1 : sample_count + 1], time_row[1 : sample_count + 1]):
        if date_value and time_value:
            datetimes.append(parse_datetime(date_value, time_value))

    if not datetimes:
        raise ValueError("No scan timestamps found")
    return min(datetimes), max(datetimes), len(datetimes)


def build_name(start: datetime, end: datetime, suffix: str) -> str:
    if start.date() == end.date():
        return f"{start:%Y%m%d_%H%M}-{end:%H%M}{suffix}"
    return f"{start:%Y%m%d_%H%M}-{end:%Y%m%d_%H%M}{suffix}"


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for counter in range(2, 1000):
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Could not find a unique name for {path.name}")


def destination_path(current_path: Path, new_name: str) -> Path:
    wanted_path = current_path.with_name(new_name)
    if current_path.resolve() == wanted_path.resolve():
        return wanted_path
    return unique_path(wanted_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rename AIM .txt exports using first and last timestamps inside the file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="final files selected",
        help="Folder containing AIM .txt exports, or a single .txt export.",
    )
    parser.add_argument("--apply", action="store_true", help="Actually rename files. Default is preview only.")
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix, for example Kigali_ to make Kigali_20240219_0915-20240220_1158.txt.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    txt_files = sorted(input_path.glob("*.txt")) if input_path.is_dir() else [input_path]
    if not txt_files:
        raise SystemExit(f"No .txt files found in {input_path}")

    skipped = 0
    for txt_file in txt_files:
        try:
            start, end, scan_count = scan_time_range(txt_file)
        except ValueError as exc:
            skipped += 1
            print(f"SKIP    {txt_file.name}: {exc}")
            continue
        new_name = args.prefix + build_name(start, end, txt_file.suffix)
        new_path = destination_path(txt_file, new_name)

        if txt_file.resolve() == new_path.resolve():
            print(f"OK      {txt_file.name} already matches {new_path.name} ({scan_count} scans)")
        elif args.apply:
            txt_file.rename(new_path)
            print(f"RENAMED {txt_file.name} -> {new_path.name} ({scan_count} scans)")
        else:
            print(f"PREVIEW {txt_file.name} -> {new_path.name} ({scan_count} scans)")

    if not args.apply:
        print("\nPreview only. Add --apply to rename the files.")
    if skipped:
        print(f"\nSkipped {skipped} file(s) that did not contain a complete AIM timestamp table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
