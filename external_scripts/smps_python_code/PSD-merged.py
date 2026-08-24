"""
Merged Banana plot combining Nano + Long SMPS into single panel + Tide overlays
- Merges Nano (2.5-30nm) and Long (10-400nm) SMPS data
- Overlap region (10-30nm): averaged where both instruments measure
- Date range: configurable
- Log color scale with colorbar on right
- Larger fonts for readability
- Saves each day to D:\Documents\PSD Plots\banana_merged_YYYY-MM-DD.png
- ALSO generates a single PDF combining all days in order
- Overlays tide events: AM High, PM High, AM Low, PM Low
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from scipy.interpolate import interp1d


# -----------------------------
# Loader function
# -----------------------------
def load_smps_for_pcolormesh(file_path, dmin=None, dmax=None, drop_below_nm=1.0):
    """Load one SMPS CSV and prepare pcolormesh inputs (t_edges, d_edges, Z, scan_times)."""
    df = pd.read_csv(file_path, low_memory=False)
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["dia set"])
    df["dia set"] = pd.to_numeric(df["dia set"], errors="coerce")
    df["dNdlnDp"] = pd.to_numeric(df["dNdlnDp"], errors="coerce")
    df = df.sort_values(["Datetime", "dia set"])
    df["dNdlogDp"] = df["dNdlnDp"] * np.log(10)

    # Drop rows with invalid datetime
    df = df.dropna(subset=["Datetime"])

    sizes_all = np.sort(df["dia set"].dropna().unique())
    mask = np.ones_like(sizes_all, dtype=bool)
    if dmin is not None:
        mask &= sizes_all >= dmin
    if dmax is not None:
        mask &= sizes_all <= dmax
    if drop_below_nm is not None:
        mask &= sizes_all >= float(drop_below_nm)

    sizes = sizes_all[mask]
    if sizes.size < 2:
        raise ValueError(f"Not enough size bins after filtering: {file_path}")

    # Detect scans
    df["scan_id"] = (df["dia set"].diff() < 0).cumsum()
    pivot = df.pivot_table(index="scan_id", columns="dia set",
                           values="dNdlogDp", aggfunc="mean")
    pivot = pivot.reindex(columns=sizes)

    # Diameter edges
    d_edges = np.empty(sizes.size + 1)
    d_edges[1:-1] = np.sqrt(sizes[:-1] * sizes[1:])
    d_edges[0] = sizes[0] / np.sqrt(sizes[1] / sizes[0])
    d_edges[-1] = sizes[-1] * np.sqrt(sizes[-1] / sizes[-2])

    # Time edges
    scan_times = df.groupby("scan_id")["Datetime"].median().reindex(pivot.index)

    # Filter out any NaT values in scan_times
    valid_mask = scan_times.notna()
    scan_times = scan_times[valid_mask]
    pivot = pivot.loc[valid_mask]

    if len(scan_times) < 1:
        raise ValueError(f"No valid scan times found in: {file_path}")

    t_num = mdates.date2num(scan_times.to_numpy())

    # Check for non-finite values in t_num
    if not np.all(np.isfinite(t_num)):
        raise ValueError(f"Non-finite time values in: {file_path}")

    t_edges = np.empty(t_num.size + 1)
    if t_num.size > 1:
        t_edges[1:-1] = (t_num[:-1] + t_num[1:]) / 2
        t_edges[0] = t_num[0] - (t_num[1] - t_num[0]) / 2
        t_edges[-1] = t_num[-1] + (t_num[-1] - t_num[-2]) / 2
    else:
        t_edges[:] = [t_num[0] - 0.5, t_num[0] + 0.5]

    # Z grid
    Z = pivot.T.to_numpy()
    Z = np.ma.masked_where((~np.isfinite(Z)) | (Z <= 0), Z)

    return t_edges, d_edges, Z, scan_times, sizes


def merge_smps_data(nano_path, long_path):
    """
    Merge Nano and Long SMPS data into a single unified grid.

    Strategy:
    - Nano: ~2.5-30 nm
    - Long: ~10-400 nm
    - Overlap: 10-30 nm → average both instruments
    - Create unified time grid based on both datasets
    - Interpolate each dataset to common time grid
    - Merge size bins: Nano (below 10nm) + averaged overlap + Long (above 30nm)
    """

    # Load both datasets with their full ranges
    tE_nano, dE_nano, Z_nano, times_nano, sizes_nano = load_smps_for_pcolormesh(
        nano_path, dmin=2.5, dmax=40, drop_below_nm=2.5
    )
    tE_long, dE_long, Z_long, times_long, sizes_long = load_smps_for_pcolormesh(
        long_path, dmin=10, dmax=400, drop_below_nm=None
    )

    # Convert scan times to numeric for interpolation
    t_nano_num = mdates.date2num(times_nano.to_numpy())
    t_long_num = mdates.date2num(times_long.to_numpy())

    # Create unified time grid (use the union of both time points, then create regular grid)
    t_min = min(t_nano_num.min(), t_long_num.min())
    t_max = max(t_nano_num.max(), t_long_num.max())

    # Determine time resolution from the finer instrument
    dt_nano = np.median(np.diff(t_nano_num)) if len(t_nano_num) > 1 else 1 / 48
    dt_long = np.median(np.diff(t_long_num)) if len(t_long_num) > 1 else 1 / 48
    dt = min(dt_nano, dt_long)

    # Create unified time grid
    n_times = int((t_max - t_min) / dt) + 1
    t_unified = np.linspace(t_min, t_max, n_times)

    # Define size boundaries for merging
    # Nano only: < 10 nm
    # Overlap: 10-30 nm (average both)
    # Long only: > 30 nm

    nano_only_mask = sizes_nano < 10
    long_only_mask = sizes_long > 30

    # Overlap region size bins (from both instruments)
    nano_overlap_mask = (sizes_nano >= 10) & (sizes_nano <= 30)
    long_overlap_mask = (sizes_long >= 10) & (sizes_long <= 30)

    # Extract size bins for each region
    sizes_nano_only = sizes_nano[nano_only_mask]
    sizes_nano_overlap = sizes_nano[nano_overlap_mask]
    sizes_long_overlap = sizes_long[long_overlap_mask]
    sizes_long_only = sizes_long[long_only_mask]

    # Create unified size array (need to merge overlap bins intelligently)
    # Use all unique sizes from both instruments in overlap, sorted
    sizes_overlap_combined = np.unique(np.concatenate([sizes_nano_overlap, sizes_long_overlap]))
    sizes_unified = np.concatenate([sizes_nano_only, sizes_overlap_combined, sizes_long_only])
    sizes_unified = np.sort(sizes_unified)

    # Initialize merged Z array
    Z_merged = np.zeros((len(sizes_unified), len(t_unified)))

    # Interpolate Nano data to unified time grid for each size bin
    Z_nano_interp = np.full((len(sizes_nano), len(t_unified)), np.nan)
    for i, size in enumerate(sizes_nano):
        valid_mask = ~Z_nano.mask[i, :] if hasattr(Z_nano[i, :], 'mask') else np.isfinite(Z_nano[i, :])
        if valid_mask.sum() > 1:
            f = interp1d(t_nano_num[valid_mask], Z_nano[i, valid_mask],
                         kind='linear', bounds_error=False, fill_value=np.nan)
            Z_nano_interp[i, :] = f(t_unified)

    # Interpolate Long data to unified time grid for each size bin
    Z_long_interp = np.full((len(sizes_long), len(t_unified)), np.nan)
    for i, size in enumerate(sizes_long):
        valid_mask = ~Z_long.mask[i, :] if hasattr(Z_long[i, :], 'mask') else np.isfinite(Z_long[i, :])
        if valid_mask.sum() > 1:
            f = interp1d(t_long_num[valid_mask], Z_long[i, valid_mask],
                         kind='linear', bounds_error=False, fill_value=np.nan)
            Z_long_interp[i, :] = f(t_unified)

    # Fill merged array
    for i, size in enumerate(sizes_unified):
        if size < 10:
            # Nano only region
            nano_idx = np.where(sizes_nano == size)[0]
            if len(nano_idx) > 0:
                Z_merged[i, :] = Z_nano_interp[nano_idx[0], :]

        elif size > 30:
            # Long only region
            long_idx = np.where(sizes_long == size)[0]
            if len(long_idx) > 0:
                Z_merged[i, :] = Z_long_interp[long_idx[0], :]

        else:
            # Overlap region (10-30 nm): average both instruments where available
            nano_idx = np.where(sizes_nano == size)[0]
            long_idx = np.where(sizes_long == size)[0]

            if len(nano_idx) > 0 and len(long_idx) > 0:
                # Both have this size - average them
                nano_vals = Z_nano_interp[nano_idx[0], :]
                long_vals = Z_long_interp[long_idx[0], :]

                # Average where both are valid
                both_valid = np.isfinite(nano_vals) & np.isfinite(long_vals)
                nano_only_valid = np.isfinite(nano_vals) & ~np.isfinite(long_vals)
                long_only_valid = ~np.isfinite(nano_vals) & np.isfinite(long_vals)

                Z_merged[i, both_valid] = (nano_vals[both_valid] + long_vals[both_valid]) / 2
                Z_merged[i, nano_only_valid] = nano_vals[nano_only_valid]
                Z_merged[i, long_only_valid] = long_vals[long_only_valid]

            elif len(nano_idx) > 0:
                # Only nano has this size in overlap region
                Z_merged[i, :] = Z_nano_interp[nano_idx[0], :]

            elif len(long_idx) > 0:
                # Only long has this size in overlap region
                Z_merged[i, :] = Z_long_interp[long_idx[0], :]

    # Create edges for pcolormesh
    d_edges = np.empty(len(sizes_unified) + 1)
    d_edges[1:-1] = np.sqrt(sizes_unified[:-1] * sizes_unified[1:])
    d_edges[0] = sizes_unified[0] / np.sqrt(sizes_unified[1] / sizes_unified[0])
    d_edges[-1] = sizes_unified[-1] * np.sqrt(sizes_unified[-1] / sizes_unified[-2])

    t_edges = np.empty(len(t_unified) + 1)
    if len(t_unified) > 1:
        t_edges[1:-1] = (t_unified[:-1] + t_unified[1:]) / 2
        t_edges[0] = t_unified[0] - (t_unified[1] - t_unified[0]) / 2
        t_edges[-1] = t_unified[-1] + (t_unified[-1] - t_unified[-2]) / 2
    else:
        t_edges[:] = [t_unified[0] - 0.5, t_unified[0] + 0.5]

    # Mask invalid/zero values
    Z_merged = np.ma.masked_where((~np.isfinite(Z_merged)) | (Z_merged <= 0), Z_merged)

    return t_edges, d_edges, Z_merged


# -----------------------------
# Config
# -----------------------------
base_dir = r"C:/Users/zengz/Box/Jen Lab Data Archive/PSD Data 2024"
nano_fmt = "nanodma_{date_nodash}_000000.csv"
long_fmt = "longdma_{date_nodash}_000000.csv"

# Output folder for figures
output_dir = r"D:\Documents\PSD Plots"
os.makedirs(output_dir, exist_ok=True)

# Date range: Update as needed
date_strings = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-10-03", "2024-10-10")]

# Color scale
vmin, vmax = 1e3, 1e5
norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
cmap = cm.jet

# Fonts
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18
})


def nice_log_ticks(vmin, vmax):
    pmin = int(np.floor(np.log10(vmin)))
    pmax = int(np.ceil(np.log10(vmax)))
    return [10.0 ** p for p in range(pmin, pmax + 1)]


# -----------------------------
# Tide CSV (Boothbay Harbor, ME - Sep 2025)
# Expected columns (ISO dates):
# Date,High_AM,High_AM_ft,High_PM,High_PM_ft,Low_AM,Low_AM_ft,Low_PM,Low_PM_ft
# # -----------------------------
# TIDE_CSV_PATH = r"D:\Documents\PSD Plots\tides_sep_2025.csv"
# tides = pd.read_csv(TIDE_CSV_PATH, parse_dates=["Date"], low_memory=False)


def _dt_or_none(date_str, time_str):
    """Combine date and HH:MM (or NaN) into a pandas Timestamp or None."""
    if pd.isna(time_str):
        return None
    s = f"{date_str} {str(time_str)}"
    try:
        return pd.to_datetime(s)
    except Exception:
        return None


# def overlay_tides(ax, date_iso):
#     """
#     Draw vertical lines and markers for AM/PM High/Low tides on merged plot.
#     - Markers: ^ High, v Low (AM/PM colored)
#     """
#     row = tides.loc[tides["Date"] == pd.to_datetime(date_iso)]
#     if row.empty:
#         return
#
#     row = row.iloc[0]
#
#     # Parse event datetimes
#     events = [
#         ("AM High", _dt_or_none(date_iso, row.get("High_AM")), row.get("High_AM_ft"), "tab:blue", "-", "^"),
#         ("PM High", _dt_or_none(date_iso, row.get("High_PM")), row.get("High_PM_ft"), "tab:purple", "-", "^"),
#         ("AM Low", _dt_or_none(date_iso, row.get("Low_AM")), row.get("Low_AM_ft"), "tab:orange", "--", "v"),
#         ("PM Low", _dt_or_none(date_iso, row.get("Low_PM")), row.get("Low_PM_ft"), "tab:green", "--", "v"),
#     ]
#
#     # Marker height for merged plot (spans 2.5-400 nm)
#     y_high = 350
#     y_low = 3
#
#     handles = []
#     labels = []
#
#     for label, tdt, h_ft, color, ls, marker in events:
#         if tdt is None or pd.isna(h_ft):
#             continue
#         # vertical line
#         ax.axvline(tdt, color=color, linestyle=ls, linewidth=1.2, alpha=0.9)
#
#         # marker
#         if "High" in label:
#             ax.scatter(tdt, y_high, color=color, edgecolor="black", s=70, marker=marker, zorder=5)
#         else:
#             ax.scatter(tdt, y_low, color=color, edgecolor="black", s=70, marker=marker, zorder=5)
#
#         # legend label
#         legend_text = f"{label} ({h_ft:.1f} ft)"
#         handles.append(plt.Line2D([0], [0], color=color, linestyle=ls, marker=marker,
#                                   markerfacecolor=color, markeredgecolor="black", linewidth=1.5))
#         labels.append(legend_text)
#
#     if handles:
#         # Deduplicate labels while preserving order
#         uniq = []
#         seen = set()
#         for h, l in zip(handles, labels):
#             if l not in seen:
#                 uniq.append((h, l))
#                 seen.add(l)
#         ax.legend([h for h, _ in uniq], [l for _, l in uniq],
#                   loc="upper right", frameon=True, title="Tides", fontsize=10, title_fontsize=11)
#

# -----------------------------
# Plot per day & collect into PDF
# -----------------------------
pdf_path = os.path.join(output_dir, "banana_merged_10031010.pdf")
with PdfPages(pdf_path) as pdf:
    for ds in date_strings:
        date_nodash = ds.replace("-", "")
        day_dir = os.path.join(base_dir, ds)

        nano_path = os.path.join(day_dir, nano_fmt.format(date_nodash=date_nodash))
        long_path = os.path.join(day_dir, long_fmt.format(date_nodash=date_nodash))
        if not os.path.isfile(nano_path) or not os.path.isfile(long_path):
            print(f"[skip] Missing files for {ds}")
            continue

        try:
            # Merge the datasets
            tE_merged, dE_merged, Z_merged = merge_smps_data(nano_path, long_path)
        except Exception as e:
            print(f"[skip] {ds}: {e}")
            continue

        # Create single panel plot
        fig, ax = plt.subplots(1, 1, figsize=(9, 5), gridspec_kw={"right": 0.86})

        # Plot merged data
        pcm = ax.pcolormesh(tE_merged, dE_merged, Z_merged, norm=norm, cmap=cmap, shading="flat")
        ax.set_yscale('log')
        ax.set_ylabel('Diameter (nm)', fontsize=14)
        ax.set_xlabel('Local Time (US Eastern)', fontsize=14)
        ax.set_ylim(1, 500)
        # Set y-axis ticks to include 10^0
        y_ticks = [1, 10, 100]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([r'$10^0$', r'$10^1$', r'$10^2$'])

        # ax.set_title('Merged Nano + Long SMPS', fontsize=16)

        # Time axis formatting
        xmin, xmax = tE_merged[0], tE_merged[-1]
        ax.set_xlim(xmin, xmax)
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Overlay tide events
        # overlay_tides(ax, ds)

        # Colorbar on the right
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm, cax=cbar_ax, extend='both')
        cbar.set_label('dN/dlogDp', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        ticks = nice_log_ticks(vmin, vmax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([fr"$10^{int(np.log10(t))}$" for t in ticks])

        # fig.suptitle(f'Merged PSD Contour — {ds}', y=0.98, fontsize=18)
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        # Save PNG + append to combined PDF
        out_path = os.path.join(output_dir, f"banana_merged_{ds}.png")
        plt.savefig(out_path, dpi=200)
        pdf.savefig(fig)
        plt.close(fig)

        print(f"[ok] Saved {out_path} & added to PDF")

print(f"[done] Combined PDF saved at: {pdf_path}")