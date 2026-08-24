import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import dates as mdates

# ========== CONFIG ========== #
kigali_file = (
    r"D:/Documents/2025/SMPS Comparison/Data/Kigali/"
    r"Kigali_SMPS_2024_03_25.txt"
)

out_folder = r"D:/Documents/2025/SMPS Comparison/Figures/Kigali_banana_daily"
os.makedirs(out_folder, exist_ok=True)

start_date_str = "2024-03-10"
end_date_str   = "2024-03-20"
# ============================ #


def get_bounds_and_dlogDp(mid_D):
    log_mid = np.log10(mid_D)
    avg_diff = np.mean(np.diff(log_mid))
    D_bound = np.empty(len(mid_D) + 1)
    D_bound[1:-1] = 10 ** ((log_mid[1:] + log_mid[:-1]) / 2)
    D_bound[0] = 10 ** (log_mid[0] - 0.5 * avg_diff)
    D_bound[-1] = 10 ** (log_mid[-1] + 0.5 * avg_diff)
    dlogDp = np.log10(D_bound[1:]) - np.log10(D_bound[:-1])
    return D_bound, dlogDp


def load_kigali_data(file_path, start_date, end_date):
    """Your original Kigali loader, simplified to what we need."""
    df = pd.read_csv(file_path, skiprows=17, encoding='latin-1',on_bad_lines="warn")

    # Process Time
    df['Date'] = df['Date'].astype(str).str.strip()
    df['Start Time'] = df['Start Time'].astype(str).str.strip()
    df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'],
                                format='%m/%d/%y %H:%M:%S', errors='coerce')
    df.dropna(subset=['Time'], inplace=True)

    columns = list(df.columns)
    df = df[[columns[-1]] + columns[:-1]]
    df.fillna(0, inplace=True)
    tsdf = df.set_index('Time')

    tsdf = tsdf.loc[start_date:end_date]
    if len(tsdf) == 0:
        print(f"Warning: No Kigali data in date range {start_date} to {end_date}")
        return None, None, None, None

    mid_D = np.array([float(x) for x in tsdf.columns[8:200]])
    dNdlogDp = tsdf.iloc[:, 8:200].to_numpy()
    _, dlogDp = get_bounds_and_dlogDp(mid_D)
    time = tsdf.index

    return mid_D, dlogDp, dNdlogDp, time


def plot_banana(time, dp, dNdlogDp, title, save_path=None):
    """Same style as your multisite banana plot."""
    time_idx = pd.DatetimeIndex(time)  # naive local (Kigali)
    T, D = np.meshgrid(mdates.date2num(time_idx), dp)
    Z = dNdlogDp.T
    Z[Z <= 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    pcm = ax.pcolormesh(
        T, D, Z,
        shading='auto',
        norm=LogNorm(vmin=1e3, vmax=1e5),
        cmap='turbo'
    )

    ax.set_yscale('log')
    ax.set_ylim(10, 400)
    ax.set_ylabel("Dp [nm]")
    ax.set_xlabel("Time (local)")
    ax.set_title(title)

    plt.colorbar(pcm, ax=ax, label='dN/dlogDp [cm⁻³]')

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d\n%H:%M'))
    ax.set_xlim(time_idx.min(), time_idx.max())

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()


def main():
    start_dt = pd.Timestamp(start_date_str)
    end_dt   = pd.Timestamp(end_date_str)

    mid_D, dlogDp, dNdlogDp, time = load_kigali_data(kigali_file, start_dt, end_dt)
    if mid_D is None:
        return

    # full-range
    full_title = f"Kigali SMPS {start_date_str} to {end_date_str}"
    full_path = os.path.join(out_folder, f"Kigali_banana_{start_date_str}_to_{end_date_str}.png")
    plot_banana(time, mid_D, dNdlogDp, full_title, save_path=full_path)
    print("Saved full-range:", full_path)

    # daily
    df_dNd = pd.DataFrame(dNdlogDp, index=time, columns=mid_D)
    unique_days = sorted({ts.normalize() for ts in df_dNd.index})

    start_day = start_dt.normalize()
    end_day   = end_dt.normalize()
    unique_days = [d for d in unique_days if (d >= start_day) and (d <= end_day)]

    for day in unique_days:
        mask = (df_dNd.index.normalize() == day)
        df_day = df_dNd.loc[mask]
        if df_day.empty:
            continue

        time_day = df_day.index
        dNdlogDp_day = df_day.values

        date_str = day.strftime("%Y-%m-%d")
        title = f"Kigali SMPS – {date_str}"
        fname = f"Kigali_banana_{date_str}.png"
        save_path = os.path.join(out_folder, fname)

        plot_banana(time_day, mid_D, dNdlogDp_day, title, save_path=save_path)
        print("Saved daily:", save_path)


if __name__ == "__main__":
    main()
