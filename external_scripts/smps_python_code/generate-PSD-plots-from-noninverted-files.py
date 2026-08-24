"""
Banana plots for two SMPS ranges per day (Long on top, Nano on bottom) + Tide overlays
- Date range: 2025-09-05 → 2025-09-20 (inclusive)
- Shared log color scale
- Colorbar to the right (outside)
- Larger fonts for readability
- Saves each day to D:\Documents\PSD Plots\banana_YYYY-MM-DD.png
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
from matplotlib.backends.backend_pdf import PdfPages  # for multi-page PDF
from datetime import datetime

# -----------------------------
# Loader function
# -----------------------------
def load_smps_for_pcolormesh(file_path, dmin=None, dmax=None, drop_below_nm=1.0):
    """Load one SMPS CSV and prepare pcolormesh inputs (t_edges, d_edges, Z, scan_times)."""
    df = pd.read_csv(file_path, low_memory=False)  # Fix the dtype warning
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

    return t_edges, d_edges, Z, scan_times

# -----------------------------
# Config
# -----------------------------
base_dir = r"C:/Users/zengz/Box/Jen Lab Data Archive/PSD"
nano_fmt = "nanodma_{date_nodash}_000000.csv"
long_fmt = "longdma_{date_nodash}_000000.csv"

# Output folder for figures
output_dir = r"D:\Documents\PSD Plots"
os.makedirs(output_dir, exist_ok=True)

# Date range: 9/5 → 9/20
date_strings = [d.strftime("%Y-%m-%d") for d in pd.date_range("2025-09-27", "2025-09-28")]

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
    return [10.0**p for p in range(pmin, pmax + 1)]


# -----------------------------
# Tide CSV (Boothbay Harbor, ME - Sep 2025)
# Expected columns (ISO dates):
# Date,High_AM,High_AM_ft,High_PM,High_PM_ft,Low_AM,Low_AM_ft,Low_PM,Low_PM_ft
# -----------------------------
TIDE_CSV_PATH = r"D:\Documents\PSD Plots\tides_sep_2025.csv"
tides = pd.read_csv(TIDE_CSV_PATH, parse_dates=["Date"], low_memory=False)

def _dt_or_none(date_str, time_str):
    """Combine date and HH:MM (or NaN) into a pandas Timestamp or None."""
    if pd.isna(time_str):
        return None
    s = f"{date_str} {str(time_str)}"
    try:
        return pd.to_datetime(s)
    except Exception:
        return None

def overlay_tides(ax_top, ax_bot, date_iso):
    """
    Draw vertical lines and markers for AM/PM High/Low tides.
    - Lines on both panels for alignment
    - Markers: ^ High, v Low (AM/PM colored)
    - Legend only on top axis
    """
    row = tides.loc[tides["Date"] == pd.to_datetime(date_iso)]
    if row.empty:
        return

    row = row.iloc[0]

    # Parse event datetimes
    events = [
        ("AM High", _dt_or_none(date_iso, row.get("High_AM")), row.get("High_AM_ft"), "tab:blue",  "-", "^"),
        ("PM High", _dt_or_none(date_iso, row.get("High_PM")), row.get("High_PM_ft"), "tab:purple","-", "^"),
        ("AM Low",  _dt_or_none(date_iso, row.get("Low_AM")),  row.get("Low_AM_ft"),  "tab:orange","--","v"),
        ("PM Low",  _dt_or_none(date_iso, row.get("Low_PM")),  row.get("Low_PM_ft"),  "tab:green", "--","v"),
    ]

    # Marker heights (log y)
    # Long panel ~10–400 nm; Nano panel ~1–40 nm
    y_high_top, y_low_top   = 350, 12
    y_high_bot, y_low_bot   = 35,  1.2

    handles = []
    labels  = []

    for label, tdt, h_ft, color, ls, marker in events:
        if tdt is None or pd.isna(h_ft):
            continue
        # vertical lines
        for ax in (ax_top, ax_bot):
            ax.axvline(tdt, color=color, linestyle=ls, linewidth=1.2, alpha=0.9)

        # markers (top & bottom)
        if "High" in label:
            ax_top.scatter(tdt, y_high_top, color=color, edgecolor="black", s=70, marker=marker, zorder=5)
            ax_bot.scatter(tdt, y_high_bot, color=color, edgecolor="black", s=70, marker=marker, zorder=5)
        else:
            ax_top.scatter(tdt, y_low_top,  color=color, edgecolor="black", s=70, marker=marker, zorder=5)
            ax_bot.scatter(tdt, y_low_bot,  color=color, edgecolor="black", s=70, marker=marker, zorder=5)

        # legend label (unique per event type)
        legend_text = f"{label} ({h_ft:.1f} ft)"
        handles.append(plt.Line2D([0],[0], color=color, linestyle=ls, marker=marker,
                                  markerfacecolor=color, markeredgecolor="black", linewidth=1.5))
        labels.append(legend_text)

    if handles:
        # Deduplicate labels while preserving order
        uniq = []
        seen = set()
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq.append((h, l))
                seen.add(l)
        ax_top.legend([h for h,_ in uniq], [l for _,l in uniq],
                      loc="upper right", frameon=True, title="Tides", fontsize=10, title_fontsize=11)


# -----------------------------
# Plot per day & collect into PDF
# -----------------------------
pdf_path = os.path.join(output_dir, "banana_09230930.pdf")
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
            # Slightly widened ranges for resilience
            tE_long, dE_long, Z_long, _ = load_smps_for_pcolormesh(
                long_path, dmin=10, dmax=400, drop_below_nm=None
            )
            tE_nano, dE_nano, Z_nano, _ = load_smps_for_pcolormesh(
                nano_path, dmin=1, dmax=40, drop_below_nm=1.0
            )
        except Exception as e:
            print(f"[skip] {ds}: {e}")
            continue

        xmin = min(tE_long[0], tE_nano[0])
        xmax = max(tE_long[-1], tE_nano[-1])

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(16, 9), sharex=True,
            gridspec_kw={"right": 0.86}
        )

        # Long SMPS (top)
        pcm1 = ax_top.pcolormesh(tE_long, dE_long, Z_long, norm=norm, cmap=cmap, shading="flat")
        ax_top.set_yscale('log')
        ax_top.set_ylabel('Diameter (nm)')
        ax_top.set_ylim(10, 400)
        ax_top.set_title('Long SMPS', fontsize=16)

        # Nano SMPS (bottom)
        pcm2 = ax_bot.pcolormesh(tE_nano, dE_nano, Z_nano, norm=norm, cmap=cmap, shading="flat")
        ax_bot.set_yscale('log')
        ax_bot.set_ylabel('Diameter (nm)')
        ax_bot.set_ylim(1, 40)
        ax_bot.set_xlabel('Time', fontsize=14)
        ax_bot.set_title('Nano SMPS', fontsize=16)

        # Time axis formatting
        for ax in (ax_top, ax_bot):
            ax.set_xlim(xmin, xmax)
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.setp(ax_bot.get_xticklabels(), rotation=45, ha='right')

        # ---- Tide overlay (AM/PM High/Low) ----
        overlay_tides(ax_top, ax_bot, ds)

        # Colorbar on the right
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(pcm2, cax=cbar_ax, extend='both')
        cbar.set_label('dN/dlogDp', fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        ticks = nice_log_ticks(vmin, vmax)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([fr"$10^{int(np.log10(t))}$" for t in ticks])

        fig.suptitle(f'PSD Contour — {ds}', y=0.98, fontsize=18)
        fig.tight_layout(rect=[0, 0, 0.85, 1])

        # Save PNG + append to combined PDF
        out_path = os.path.join(output_dir, f"banana_{ds}.png")
        plt.savefig(out_path, dpi=200)
        pdf.savefig(fig)
        plt.close(fig)

        print(f"[ok] Saved {out_path} & added to PDF")

print(f"[done] Combined PDF saved at: {pdf_path}")
