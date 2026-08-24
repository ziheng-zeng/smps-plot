#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 23:04:06 2025

@author: rishantilve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import datetime as dt
import glob
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as mdates
from datetime import timedelta

# --- Folder paths ---
spider_folder = "" #insert path to spider data file folder
weather_folder = "" #insert path to tempest data file folder
tide_file = "" #insert path to tide file
contour_plots = "" "" #insert path to contour folder (where generated images will be saved)

# --- Read tide data ---
tides = pd.read_csv(tide_file, sep='\s+', header=None, names=["Date", "Time", "HL"])
tides["Datetime"] = pd.to_datetime(tides["Date"] + " " + tides["Time"], format="%Y/%m/%d %H:%M")

# --- Helper to extract Dp columns ---
def get_dp_cols(df):
    dp_cols = []
    for col in df.columns:
        try:
            float(col)
            dp_cols.append(col)
        except:
            continue
    return dp_cols

# --- Helper to get banana plot data ---
def get_banana_data(df, dp_cols):
    time = df['Start datetime (PC)']
    Z = df[dp_cols].to_numpy().T
    Z = np.where(Z > 0, Z, np.nan)
    return time, Z

# --- Loop through each unique date in Spider folder ---
for date_str in sorted(set([f.split("_")[-2][:8] for f in os.listdir(spider_folder) if f.endswith(".txt")])):
    date = dt.datetime.strptime(date_str, "%Y%m%d").date()
    print(f"Processing {date}...")

    # --- Collect all Spider files for that date ---
    spider_files = sorted(glob.glob(os.path.join(spider_folder, f"*{date_str}*.txt")))
    if not spider_files:
        print(f"No Spider files for {date}")
        continue

    dfs = [pd.read_csv(f) for f in spider_files]
    df_full = pd.concat(dfs, ignore_index=True)
    df_full['Start datetime (PC)'] = pd.to_datetime(df_full['Start datetime (PC)'])

    dp_cols = get_dp_cols(df_full)
    dp = np.array([float(col) for col in dp_cols])

    #CHANGE POLARITY (positive vs. negative) with > or < below!!!!

    df_pos = df_full[df_full['V1 (V)'] > 0].copy()


    # --- Get contour data ---
    time_pos, Z_pos = get_banana_data(df_pos, dp_cols)

    # --- Handle data gaps larger than 5 minutes ---
    time_num = mdates.date2num(time_pos)
    Z_masked = Z_pos.copy()

    for i in range(1, len(time_pos)):
        gap = time_pos.iloc[i] - time_pos.iloc[i - 1]
        if gap > timedelta(minutes=5):
            Z_masked[:, i] = np.nan
            Z_masked[:, i - 1] = np.nan

    Z_pos = Z_masked

    # --- Load matching weather file ---
    weather_path = os.path.join(weather_folder, f"{date_str}-weather-data.csv")

    if not os.path.exists(weather_path):
        print(f"No weather file for {date} — continuing with empty weather plots")
        weather = pd.DataFrame(columns=[
            "Timestamp",
            "air_temperature",
            "solar_radiation",
            "wind_avg",
            "wind_direction",
            "precip"
        ])
    else:
        try:
            weather = pd.read_csv(
                weather_path,
                on_bad_lines='skip',
                engine='python'
            )
        except Exception as e:
            print(f"Error reading {weather_path}: {e}")
            weather = pd.DataFrame(columns=[
                "Timestamp",
                "air_temperature",
                "solar_radiation",
                "wind_avg",
                "wind_direction",
                "precip"
            ])

        weather = weather.loc[:, ~weather.columns.str.contains('^Unnamed')]

        if "Timestamp" not in weather.columns:
            print(f"Weather file for {date_str} missing 'Timestamp' column")
            weather = pd.DataFrame(columns=[
                "Timestamp",
                "air_temperature",
                "solar_radiation",
                "wind_avg",
                "wind_direction",
                "precip"
            ])
        else:
            weather["Timestamp"] = pd.to_datetime(weather["Timestamp"], errors='coerce')
            weather = weather.dropna(subset=["Timestamp"])

    # --- Filter tides for this date ---
    tide_today = tides[tides["Datetime"].dt.date == date]

    # --- Plot setup (NOW 4 PANELS) ---
    fig, axs = plt.subplots(
        4, 1,
        figsize=(16, 14),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1, 1, 1]}
    )

    # =========================
    # 1️⃣ Contour Plot
    # =========================
    pcm = axs[0].pcolormesh(
        time_pos,
        dp,
        Z_pos,
        shading='auto',
        cmap='turbo',
        norm=LogNorm(vmin=1e3, vmax=1e5)
    )

    axs[0].set_yscale('log')
    axs[0].set_ylabel('Dp [nm]', fontsize=20)
    axs[0].set_ylim(8, 300)
    axs[0].set_title(f"Positive Polarity - {date.strftime('%m/%d/%Y')}", fontsize=24)

    # --- Add tides ---
    for _, t in tide_today.iterrows():

        if t["HL"].strip().upper() == "H":
            color = "yellow"
        elif t["HL"].strip().upper() == "L":
            color = "orange"
        else:
            continue

        for ax in axs:
            ax.axvline(
                t["Datetime"],
                color=color,
                linestyle="--",
                lw=2,
                alpha=0.9
            )

    # --- Tide legend ---
    from matplotlib.lines import Line2D

    tide_legend = [
        Line2D([0], [0], color='yellow', lw=2, linestyle='--', label='High Tide'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Low Tide')
    ]

    axs[0].legend(handles=tide_legend, loc='upper right', fontsize=12)

    # --- Colorbar ---
    cax = inset_axes(
        axs[0],
        width="2%",
        height="100%",
        loc='center right',
        bbox_to_anchor=(0.07, 0, 1, 1),
        bbox_transform=axs[0].transAxes,
        borderpad=0
    )

    cbar = fig.colorbar(pcm, cax=cax)
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('dN/dlogDp (cm⁻³)', fontsize=16)

    # =========================
    # 2️⃣ Temperature & Solar
    # =========================
    axs[1].plot(
        weather["Timestamp"],
        weather["air_temperature"],
        color='tab:red',
        label='Temperature (°C)'
    )

    axs[1].set_ylabel('Temp (°C)', fontsize=16, color='tab:red')
    axs[1].tick_params(axis='y', labelcolor='tab:red')

    ax2b = axs[1].twinx()

    ax2b.plot(
        weather["Timestamp"],
        weather["solar_radiation"],
        color='tab:orange',
        label='Solar (W/m²)',
        alpha=0.7
    )

    ax2b.set_ylabel('Solar (W/m²)', fontsize=16, color='tab:orange')
    ax2b.tick_params(axis='y', labelcolor='tab:orange')

    axs[1].legend(loc='upper left', fontsize=12)
    ax2b.legend(loc='upper right', fontsize=12)

    # =========================
    # 3️⃣ Wind Speed
    # =========================
    axs[2].plot(
        weather["Timestamp"],
        weather["wind_avg"],
        color='tab:blue',
        label='Wind Speed (m/s)'
    )

    axs[2].set_ylabel('Wind (m/s)', fontsize=16, color='tab:blue')
    axs[2].tick_params(axis='y', labelcolor='tab:blue')

    if "wind_direction" in weather.columns and not weather["wind_direction"].isnull().all():

        step = max(1, len(weather)//30)

        times = weather["Timestamp"].iloc[::step]
        speeds = weather["wind_avg"].iloc[::step]
        dirs = weather["wind_direction"].iloc[::step]

        u = np.sin(np.deg2rad(dirs))
        v = np.cos(np.deg2rad(dirs))

        axs[2].quiver(
            times,
            speeds,
            u,
            v,
            scale=25,
            width=0.004,
            color='black',
            label='Wind Dir'
        )

    axs[2].legend(loc='upper right', fontsize=12)

    # =========================
    # 4️⃣ Precipitation
    # =========================
    axs[3].plot(
        weather["Timestamp"],
        weather["precip"],
        color='tab:green',
        linewidth=2,
        label='Precipitation'
    )

    axs[3].set_ylabel('Precip', fontsize=16, color='tab:green')
    axs[3].tick_params(axis='y', labelcolor='tab:green')

    axs[3].legend(loc='upper right', fontsize=12)

    # =========================
    # Shared x-axis
    # =========================
    start = dt.datetime.combine(date, dt.time(0, 0, 0))
    end = dt.datetime.combine(date, dt.time(23, 59, 59))

    axs[3].set_xlim(start, end)

    for ax in axs:
        ax.tick_params(labelsize=14)
        ax.grid(True, linestyle=':', alpha=0.6)

    axs[3].set_xlabel("Time", fontsize=18)

    fig.tight_layout(rect=[0, 0, 0.98, 1])




    # --- Show & Save plot ---
    outpath = os.path.join(contour_plots, f"positive_plot_{date_str}.png")
    plt.savefig(outpath, dpi=300)
    plt.show()
    plt.close(fig)
