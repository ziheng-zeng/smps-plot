import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates

# ========== CONFIG ==========
spider_folder = r"C:/Users/zengz/Box/Jen Lab Data Archive/Bigelow 2025/Spider Data/inverted"
spider_pattern = os.path.join(spider_folder, "SpiderMAGIC_SN*_N_*.txt")

out_folder = r"D:/Documents/PhD-Research/SMPS Comparison/Figures/Bigelow_banana_daily"
os.makedirs(out_folder, exist_ok=True)

# Leave as None to plot every day available in the inverted Bigelow files.
# Use strings like "2025-09-12" to restrict the range.
start_date_str = None
end_date_str = None
overwrite_existing = False
plot_tides = False
ion_polarity = None

# Tide CSV (Boothbay Harbor, ME - Sep 2025)
# Expected columns:
#   Date,High_AM,High_AM_ft,High_PM,High_PM_ft,Low_AM,Low_AM_ft,Low_PM,Low_PM_ft
TIDE_CSV_PATH = r"D:\Documents\PSD Plots\tides_sep_2025.csv"

# ============================

def get_bounds_and_dlogDp(mid_D):
    log_mid = np.log10(mid_D)
    avg_diff = np.mean(np.diff(log_mid))
    D_bound = np.empty(len(mid_D) + 1)
    D_bound[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
    D_bound[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
    D_bound[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
    dlogDp = np.log10(D_bound[1:]) - np.log10(D_bound[:-1])
    return D_bound, dlogDp


def _is_size_bin_column(col):
    try:
        float(str(col))
        return True
    except (TypeError, ValueError):
        return False


def _filter_ion_polarity(spider_df, polarity):
    """Use SpiderMAGIC convention: V1 > 0 is negative ions, V1 < 0 is positive ions."""
    if polarity is None:
        return spider_df

    v1 = pd.to_numeric(spider_df["V1 (V)"], errors="coerce")
    if polarity == "negative":
        return spider_df[v1 > 0]
    if polarity == "positive":
        return spider_df[v1 < 0]
    raise ValueError("polarity must be None, 'positive', or 'negative'")


def _extract_dndlogdp(spider_df):
    dp_cols = [c for c in spider_df.columns if _is_size_bin_column(c)]
    dNdlogDp = spider_df[dp_cols].apply(pd.to_numeric, errors="coerce")
    dNdlogDp.columns = [float(c) for c in dNdlogDp.columns]
    dNdlogDp = dNdlogDp.T.groupby(level=0).mean().T
    dNdlogDp = dNdlogDp.reindex(sorted(dNdlogDp.columns), axis=1)
    dNdlogDp = dNdlogDp.sort_index()
    dNdlogDp = dNdlogDp.groupby(level=0).mean()
    return dNdlogDp


def load_spider_data(folder_pattern, start_date=None, end_date=None, polarity=None):
    """Load SpiderMAGIC inverted files, allowing size-bin grids to change."""
    spider_files = sorted(glob.glob(folder_pattern), key=os.path.getmtime)
    if len(spider_files) == 0:
        print(f"Warning: No files found matching {folder_pattern}")
        return None

    frames = []
    for f in spider_files:
        df = pd.read_csv(f, low_memory=False)
        df["Source file"] = os.path.basename(f)
        frames.append(df)

    spider_df = pd.concat(frames, ignore_index=True, sort=False)

    spider_df['Start datetime (PC)'] = pd.to_datetime(spider_df['Start datetime (PC)'])

    # Set index and localize
    spider_df.set_index('Start datetime (PC)', inplace=True)
    try:
        spider_df.index = spider_df.index.tz_localize('US/Eastern', ambiguous='infer')
    except Exception:
        spider_df.index = spider_df.index.tz_localize(
            'US/Eastern', ambiguous='NaT', nonexistent='shift_forward'
        )

    spider_df = spider_df[spider_df.index.notna()]

    if "Mode" in spider_df.columns:
        spider_df = spider_df[spider_df["Mode"].astype(str).str.lower().eq("scan")]
    spider_df = _filter_ion_polarity(spider_df, polarity)

    if start_date is not None or end_date is not None:
        spider_df = spider_df.loc[start_date:end_date]
    if len(spider_df) == 0:
        print(f"Warning: No Spider data in date range {start_date} to {end_date}")
        return None

    return _extract_dndlogdp(spider_df)


def load_tides(csv_path):
    """
    Load tide data and return a dict mapping date() -> list of naive
    datetime objects (in local time, US/Eastern-style) for each tide event.
    """
    if not os.path.exists(csv_path):
        print(f"Warning: tide CSV not found at {csv_path}")
        return {}

    tides = pd.read_csv(csv_path, parse_dates=["Date"], low_memory=False)
    # Normalize Date to midnight (and then use .date() for keys)
    tides["Date"] = tides["Date"].dt.normalize()

    time_cols = ["High_AM", "High_PM", "Low_AM", "Low_PM"]
    tides_by_date = {}

    for _, row in tides.iterrows():
        d = row["Date"].date()
        events = []

        # Helper: convert time string or skip
        def _maybe(dt_str):
            if pd.isna(dt_str):
                return None
            dt_str = str(dt_str).strip()
            if dt_str == "":
                return None
            return pd.to_datetime(f"{d} {dt_str}")

        # High tides
        t = _maybe(row.get("High_AM"))
        if t is not None:
            events.append(("high", t))
        t = _maybe(row.get("High_PM"))
        if t is not None:
            events.append(("high", t))

        # Low tides
        t = _maybe(row.get("Low_AM"))
        if t is not None:
            events.append(("low", t))
        t = _maybe(row.get("Low_PM"))
        if t is not None:
            events.append(("low", t))

        if events:
            tides_by_date[d] = events


    return tides_by_date


def plot_banana(time, dp, dNdlogDp, title, save_path=None, tide_times=None):
    """
    Banana plot with EXACT same style as your good Lawrenceville version:
    - convert tz-aware → naive Eastern for plotting
    - LogNorm 1e3–1e5
    - optional tide_times: list of datetime-like values where vertical dashed
      lines are drawn.
    """
    time_idx = pd.DatetimeIndex(time)

    # Same trick as Lawrenceville to keep labels in local Eastern
    if time_idx.tz is not None:
        time_plot = time_idx.tz_convert("US/Eastern").tz_localize(None)
    else:
        time_plot = time_idx

    plot_df = pd.DataFrame(dNdlogDp, index=time_plot, columns=dp)
    plot_df = plot_df.groupby(level=0).mean().sort_index()
    time_plot = pd.DatetimeIndex(plot_df.index)
    dp = plot_df.columns.to_numpy(dtype=float)
    dNdlogDp = plot_df.to_numpy()

    T, D = np.meshgrid(mdates.date2num(time_plot), dp)
    Z = dNdlogDp.T
    Z[Z <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))

    pcm = ax.pcolormesh(
        T,
        D,
        Z,
        shading='auto',
        norm=LogNorm(vmin=1e3, vmax=1e5),
        cmap='turbo'
    )

    ax.set_yscale('log')
    positive_dp = np.asarray(dp, dtype=float)
    positive_dp = positive_dp[positive_dp > 0]
    ax.set_ylim(positive_dp.min(), 400)
    ax.set_ylabel("Dp [nm]")
    ax.set_xlabel("Time (US/Eastern)")
    ax.set_title(title)

    plt.colorbar(pcm, ax=ax, label='dN/dlogDp [cm^-3]')

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d\n%H:%M'))
    ax.set_xlim(time_plot.min(), time_plot.max())

    # ---- overlay tide times as dashed vertical lines + legend ----
    legend_handles = []
    legend_labels = []
    already_added_high = False
    already_added_low = False

    if tide_times is not None and len(tide_times) > 0:
        for label, t in tide_times:
            t = pd.to_datetime(t)
            # Make naive Eastern
            if t.tzinfo is not None:
                t_plot = t.tz_convert("US/Eastern").tz_localize(None)
            else:
                t_plot = t

            if label == "high":
                ax.axvline(t_plot, linestyle="--", linewidth=1.1, color="blue", alpha=0.8)
                if not already_added_high:
                    legend_handles.append(
                        plt.Line2D([0], [0], color="blue", linestyle="--", linewidth=1.5)
                    )
                    legend_labels.append("High tide")
                    already_added_high = True

            elif label == "low":
                ax.axvline(t_plot, linestyle=":", linewidth=1.1, color="orange", alpha=0.8)
                if not already_added_low:
                    legend_handles.append(
                        plt.Line2D([0], [0], color="orange", linestyle=":", linewidth=1.5)
                    )
                    legend_labels.append("Low tide")
                    already_added_low = True

    if legend_handles:
        ax.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            frameon=True,
            fontsize=9,
            title="Tides",
            title_fontsize=10,
        )

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def main():
    start_dt = pd.Timestamp(start_date_str, tz="US/Eastern") if start_date_str else None
    end_dt = pd.Timestamp(end_date_str, tz="US/Eastern") if end_date_str else None

    # Load Spider data
    df_dNd = load_spider_data(spider_pattern, start_dt, end_dt, polarity=ion_polarity)
    if df_dNd is None:
        return

    if start_dt is None:
        start_dt = df_dNd.index.min().normalize()
    if end_dt is None:
        end_dt = df_dNd.index.max().normalize()

    # Load tide data
    tides_by_date = load_tides(TIDE_CSV_PATH) if plot_tides else {}

    # ------------ 1) full-range plot ------------
    start_date_label = start_dt.strftime("%Y-%m-%d")
    end_date_label = end_dt.strftime("%Y-%m-%d")
    full_path = os.path.join(
        out_folder,
        f"Bigelow_banana_{start_date_label}_to_{end_date_label}.png"
    )
    if len(df_dNd) <= 50000:
        plot_banana(
            df_dNd.index,
            df_dNd.columns.to_numpy(dtype=float),
            df_dNd.to_numpy(),
            f"Bigelow SpiderMAGIC {start_date_label} to {end_date_label}",
            save_path=full_path,
        )
        print("Saved full-range:", full_path)
    else:
        print(f"Skipped full-range plot for {len(df_dNd)} scans; daily plots will be saved.")

    # ------------ 2) daily plots ------------
    # unique days as tz-aware Timestamps
    unique_days = sorted({ts.normalize() for ts in df_dNd.index})
    # Filter to requested range (still tz-aware Timestamps)
    start_day_ts = start_dt.normalize()
    end_day_ts = end_dt.normalize()
    unique_days = [d for d in unique_days if (d >= start_day_ts) and (d <= end_day_ts)]

    for day in unique_days:
        mask = (df_dNd.index.normalize() == day)
        df_day = df_dNd.loc[mask]
        if df_day.empty:
            continue

        df_day = df_day.dropna(axis=1, how="all")
        time_day = df_day.index
        mid_D_day = df_day.columns.to_numpy(dtype=float)
        dNdlogDp_day = df_day.values

        # tide times for this day (key is Python date)
        tide_times_day = tides_by_date.get(day.date(), [])

        date_str = day.strftime("%Y-%m-%d")
        title = f"Bigelow SpiderMAGIC - {date_str}"
        fname = f"Bigelow_banana_{date_str}.png"
        save_path = os.path.join(out_folder, fname)
        if os.path.exists(save_path) and not overwrite_existing:
            print("Skipped existing daily:", save_path)
            continue

        plot_banana(
            time_day,
            mid_D_day,
            dNdlogDp_day,
            title,
            save_path=save_path,
            tide_times=tide_times_day
        )
        print("Saved daily:", save_path)


if __name__ == "__main__":
    main()
