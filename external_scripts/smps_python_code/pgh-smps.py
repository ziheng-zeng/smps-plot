import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates

### ================== CONFIGURATION ================== ###
# Input SMPS data (Lawrenceville)
smps_folder = "D:/Documents/PhD-Research/SMPS data/data-all-time"
smps_pattern = os.path.join(smps_folder, "SMPS*.csv")

# Output folder for plots
out_folder = "D:/Documents/PhD-Research/SMPS Comparison/Figures/Lawrenceville_banana"
os.makedirs(out_folder, exist_ok=True)

# Date range (US/Eastern, inclusive)
start_date_str = "2024-10-30"
end_date_str   = "2024-10-31"

# Plot window length in hours: set to 24 or 48
PLOT_WINDOW_HOURS = 48  # 24 for daily, 48 for 2-day windows

# Turn plume overlay on/off
OVERLAY_PLUMES = True

# List of plume events (start, end) in local US/Eastern time (strings).
PLUME_EVENTS_STR = [
    # ("2025-03-10 19:00:00", "2025-03-11 09:00:00"),
    # ("2025-03-13 18:00:00", "2025-03-14 14:00:00"),
    # ("2025-03-18 00:00:00", "2025-03-18 12:00:00"),
    # ("2025-03-19 00:00:00", "2025-03-14 09:00:00"),
]
# ===================================================== ###


def get_bounds_and_dlogDp(mid_D):
    """Compute bin boundaries and dlogDp from bin midpoints."""
    log_mid = np.log10(mid_D)
    avg_diff = np.mean(np.diff(log_mid))
    D_bound = np.empty(len(mid_D) + 1)
    D_bound[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
    D_bound[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
    D_bound[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
    dlogDp = np.log10(D_bound[1:]) - np.log10(D_bound[:-1])
    return D_bound, dlogDp


def parse_plume_events(plume_events_str):
    """
    Parse list of (start_str, end_str) into tz-aware US/Eastern timestamps.
    """
    events = []
    for start_str, end_str in plume_events_str:
        start = pd.Timestamp(start_str, tz="US/Eastern")
        end = pd.Timestamp(end_str, tz="US/Eastern")
        events.append((start, end))
    return events


def get_plume_events_for_window(window_start, window_end, all_events):
    """
    Return plume events that intersect [window_start, window_end),
    converted to naive (no-tz) datetimes in US/Eastern for plotting.
    """
    selected = []
    for start, end in all_events:
        # Event intersects the window if it starts before window_end
        # and ends after window_start
        if (start < window_end) and (end > window_start):
            s_plot = start.tz_convert("US/Eastern").tz_localize(None)
            e_plot = end.tz_convert("US/Eastern").tz_localize(None)
            selected.append((s_plot, e_plot))
    return selected


def load_lawrenceville_smps(smps_pattern, start_date_str, end_date_str):
    """Load SMPS files, apply time filtering, STP correction and return arrays."""
    smps_files = glob.glob(smps_pattern)
    if not smps_files:
        raise FileNotFoundError(f"No SMPS files found with pattern: {smps_pattern}")

    smps_df = pd.concat([pd.read_csv(f, skiprows=52) for f in smps_files])

    # Strip column names to remove any accidental leading/trailing spaces
    smps_df.columns = smps_df.columns.str.strip()

    # Convert datetime column from UTC to US/Eastern
    smps_df['DateTime Sample Start'] = pd.to_datetime(
        smps_df['DateTime Sample Start'],
        format='%d/%m/%Y %H:%M:%S',
        utc=True
    )
    smps_df['DateTime Sample Start'] = smps_df['DateTime Sample Start'].dt.tz_convert("US/Eastern")

    # Filter for LAWRENCEVILLE date range
    start_date_smps = pd.Timestamp(start_date_str, tz="US/Eastern")
    end_date_smps   = pd.Timestamp(end_date_str, tz="US/Eastern") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    mask = (smps_df['DateTime Sample Start'] >= start_date_smps) & \
           (smps_df['DateTime Sample Start'] <= end_date_smps)
    smps_df = smps_df.loc[mask].copy()

    if smps_df.empty:
        raise ValueError("No SMPS data found within the specified date range.")

    # Set datetime as index AFTER filtering
    smps_df.set_index('DateTime Sample Start', inplace=True)

    # --- STP Correction Factor ---
    P = smps_df['Sheath Pressure (kPa)'].values  # kPa
    T = smps_df['Sheath Temp (C)'].values        # °C
    STP_factor = (101.35 / P) * ((273.15 + T) / 273.15)

    # Size bins (adapted from your original indexing)
    mid_D_smps = np.array([float(c) for c in smps_df.columns[41:425]])
    D_bound_smps, dlogDp_smps = get_bounds_and_dlogDp(mid_D_smps)

    # dN/dlogDp data (raw)
    dNdlogDp_smps = smps_df.iloc[:, 41:425].values

    # Apply STP correction per scan (broadcast across bins)
    dNdlogDp_smps = dNdlogDp_smps * STP_factor[:, None]

    # Time index
    time_smps = smps_df.index

    # Just in case, ensure shapes line up
    mid_D_smps = mid_D_smps.flatten()
    assert dNdlogDp_smps.shape[1] == mid_D_smps.shape[0], "Mismatch in bin count"

    return time_smps, mid_D_smps, dNdlogDp_smps


def plot_banana(time, dp, dNdlogDp, title, save_path=None,
                plume_lines=None, xlim_start=None, xlim_end=None):
    """
    Make a banana plot (time vs Dp with dN/dlogDp color) and optionally save.
    Optionally overlay plume start/end lines (list of (start, end) naive datetimes).
    xlim_start / xlim_end: naive datetimes defining the full window (24/48 h).
    """
    # Ensure we have a DatetimeIndex
    time_idx = pd.DatetimeIndex(time)

    # Convert to US/Eastern and then drop tz info so matplotlib treats them as "naive Eastern"
    if time_idx.tz is not None:
        time_plot = time_idx.tz_convert("US/Eastern").tz_localize(None)
    else:
        time_plot = time_idx

    T, D = np.meshgrid(mdates.date2num(time_plot), dp)
    Z = dNdlogDp.T
    Z[Z <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(
        T, D, Z,
        shading="nearest",
        antialiased=False,
        norm=LogNorm(vmin=1e3, vmax=1e5),
        cmap='turbo'
    )

    ax.set_yscale('log')
    ax.set_ylim(10, 800)
    ax.set_ylabel("Dp [nm]")
    ax.set_xlabel("Time (US/Eastern)")
    ax.set_title(title)

    plt.colorbar(pcm, ax=ax, label='dN/dlogDp [cm⁻³]')

    # Use the requested window bounds for x-limits so it's truly 24/48h wide
    if xlim_start is not None and xlim_end is not None:
        ax.set_xlim(xlim_start, xlim_end)
    else:
        ax.set_xlim(time_plot.min(), time_plot.max())

    # Overlay plume start/end lines, if provided
    if plume_lines:
        for s, e in plume_lines:
            ax.axvline(s, linestyle='--', linewidth=1)
            ax.axvline(e, linestyle='--', linewidth=1)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d\n%H:%M'))

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def main():
    # Load SMPS data for the chosen range
    time_smps, mid_D_smps, dNdlogDp_smps = load_lawrenceville_smps(
        smps_pattern, start_date_str, end_date_str
    )

    # Put data into a DataFrame for easier slicing
    df_dNd = pd.DataFrame(dNdlogDp_smps, index=time_smps, columns=mid_D_smps)

    # Build list of window start times at local midnight (tz-aware)
    start_day = pd.Timestamp(start_date_str, tz="US/Eastern").normalize()
    end_day   = pd.Timestamp(end_date_str, tz="US/Eastern").normalize()

    window_starts = []
    current = start_day
    while current <= end_day:
        window_starts.append(current)
        current += pd.Timedelta(days=1)

    # Parse plume events (tz-aware)
    all_plume_events = parse_plume_events(PLUME_EVENTS_STR) if OVERLAY_PLUMES and PLUME_EVENTS_STR else []

    # Make a plot for each window
    for window_start in window_starts:
        window_end = window_start + pd.Timedelta(hours=PLOT_WINDOW_HOURS)

        # Mask for data in this window (using tz-aware times)
        mask = (df_dNd.index >= window_start) & (df_dNd.index < window_end)
        df_win = df_dNd.loc[mask]

        if df_win.empty:
            continue

        time_win = df_win.index
        dNdlogDp_win = df_win.values

        # Naive bounds for plotting x-axis
        xlim_start = window_start.tz_convert("US/Eastern").tz_localize(None)
        xlim_end   = window_end.tz_convert("US/Eastern").tz_localize(None)

        # Title and filename
        window_label = f"{window_start.strftime('%Y-%m-%d_%H%M')}_to_{(window_end - pd.Timedelta(seconds=1)).strftime('%Y-%m-%d_%H%M')}"
        title = f"Lawrenceville SMPS – {PLOT_WINDOW_HOURS}-hr window starting {window_start.strftime('%Y-%m-%d %H:%M')}"
        fname = f"Lawrenceville_banana_{PLOT_WINDOW_HOURS}hr_{window_label}.png"
        save_path = os.path.join(out_folder, fname)

        # Get plume lines for this window (as naive datetimes)
        plume_lines = None
        if all_plume_events:
            plume_lines = get_plume_events_for_window(window_start, window_end, all_plume_events)
            print(f"{window_label}: overlaying {len(plume_lines)} plume event(s)")

        plot_banana(
            time_win,
            mid_D_smps,
            dNdlogDp_win,
            title,
            save_path=save_path,
            plume_lines=plume_lines,
            xlim_start=xlim_start,
            xlim_end=xlim_end,
        )
        print(f"Saved {PLOT_WINDOW_HOURS}-hr banana plot:\n  {save_path}")


if __name__ == "__main__":
    main()
