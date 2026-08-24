import pandas as pd
import numpy as np
from datetime import timedelta
import os
from glob import glob

# -------------------------
# FOLDERS
# -------------------------
SPIDER_FOLDER = "" #insert spider data file folder path
WEATHER_FOLDER = "" #insert tempest file folder path
TIDES_FILE = "" #insert tides file path

# -------------------------
# PARAMETERS
# -------------------------
DP_MAX_NM = 10
DN_THRESHOLD = 10**4.5
MERGE_GAP_HOURS = 3
MAX_PERIOD_HOURS = 6

LOW_TIDE_WINDOW_HOURS = 1
WEATHER_BUFFER_MINUTES = 15  # Buffer to catch 10-min weather logs for short spikes

PRECIP_THRESHOLD = 0.0
SOLAR_THRESHOLD = 10

# -------------------------
# 1. LOAD TIDES
# -------------------------
try:
    tides = pd.read_csv(TIDES_FILE, sep="\t", header=None, names=["date", "time", "tide"], dtype=str)
    tides["datetime"] = pd.to_datetime(tides["date"] + " " + tides["time"], errors="coerce")
    tides = tides.dropna(subset=["datetime"])
    tides["tide"] = tides["tide"].str.strip().str.upper() # Clean "L" vs " L"
    low_tides = tides[tides["tide"] == "L"]["datetime"]
except Exception as e:
    print(f"Tide loading error: {e}")
    low_tides = pd.Series(dtype="datetime64[ns]")

# -------------------------
# 2. CACHE WEATHER DATA (Corrected Logic)
# -------------------------
weather_files = glob(os.path.join(WEATHER_FOLDER, "*.csv"))
weather_by_date = {}

for wf in weather_files:
    try:
        wdf = pd.read_csv(wf)
        # Standardize column names (lowercase, no spaces)
        wdf.columns = wdf.columns.str.lower().str.replace(' ', '_').str.strip()

        # Identify timestamp column
        ts_col = 'timestamp' if 'timestamp' in wdf.columns else wdf.columns[0]
        wdf["datetime"] = pd.to_datetime(wdf[ts_col], errors="coerce")
        wdf = wdf.dropna(subset=["datetime"])

        # Group by date to handle files spanning multiple days
        for date, group in wdf.groupby(wdf["datetime"].dt.date):
            if date not in weather_by_date:
                weather_by_date[date] = {"precip": pd.Series(dtype="datetime64[ns]"),
                                         "solar": pd.Series(dtype="datetime64[ns]")}

            p_times = group[group["precip"] > PRECIP_THRESHOLD]["datetime"]
            s_times = group[group["solar_radiation"] > SOLAR_THRESHOLD]["datetime"]

            weather_by_date[date]["precip"] = pd.concat([weather_by_date[date]["precip"], p_times])
            weather_by_date[date]["solar"] = pd.concat([weather_by_date[date]["solar"], s_times])

    except Exception as e:
        print(f"Skipping weather file {wf}: {e}")

# -------------------------
# 3. PROCESS FUNCTION
# -------------------------
def process_spider_file(file_path):
    df = pd.read_csv(file_path)
    df["datetime"] = pd.to_datetime(df["Start datetime (PC)"])
    if df["datetime"].isna().all(): return []

    # Get thresholds for the specific dates covered by this file
    unique_dates = df["datetime"].dt.date.unique()
    precip_times = pd.concat([weather_by_date.get(d, {}).get("precip", pd.Series(dtype="datetime64[ns]")) for d in unique_dates])
    solar_times = pd.concat([weather_by_date.get(d, {}).get("solar", pd.Series(dtype="datetime64[ns]")) for d in unique_dates])

    # Identify Dp columns
    dp_cols = [c for c in df.columns if c.replace('.','',1).isdigit()]
    small_dp_cols = [c for c in dp_cols if float(c) <= DP_MAX_NM]
    df["is_spike"] = (df[small_dp_cols] >= DN_THRESHOLD).any(axis=1)

    # Build spike periods
    periods = []
    in_period = False
    for i in range(len(df)):
        if df.loc[i, "is_spike"] and not in_period:
            start_time = df.loc[i, "datetime"]; in_period = True
        elif not df.loc[i, "is_spike"] and in_period:
            periods.append((start_time, df.loc[i-1, "datetime"])); in_period = False
    if in_period: periods.append((start_time, df.iloc[-1]["datetime"]))

    # Merge and Split logic
    merged = []
    for s, e in sorted(periods):
        if not merged or s - merged[-1][1] > timedelta(hours=MERGE_GAP_HOURS):
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    final_periods = []
    for s, e in merged:
        while e - s > timedelta(hours=MAX_PERIOD_HOURS):
            final_periods.append((s, s + timedelta(hours=MAX_PERIOD_HOURS)))
            s += timedelta(hours=MAX_PERIOD_HOURS)
        final_periods.append((s, e))


## this weather flagging still doesn't work quite well...


    # Flagging
    results = []
    w_buf = timedelta(minutes=WEATHER_BUFFER_MINUTES)
    for s, e in final_periods:
        res_precip = (not precip_times.empty) and ((precip_times >= s - w_buf) & (precip_times <= e + w_buf)).any()
        res_solar = (not solar_times.empty) and ((solar_times >= s - w_buf) & (solar_times <= e + w_buf)).any()
        res_tide = (not low_tides.empty) and ((low_tides >= s - timedelta(hours=1)) & (low_tides <= e + timedelta(hours=1))).any()

        results.append({
            "file": os.path.basename(file_path),
            "start_time": s, "end_time": e,
            "low_tide": res_tide,
            "precip": res_precip,
            "solar": res_solar
        })
    return results

# -------------------------
# RUN
# -------------------------
all_res = []
for f in glob(os.path.join(SPIDER_FOLDER, "*.txt")):
    all_res.extend(process_spider_file(f))

if all_res:
    results_df = pd.DataFrame(all_res)
    print(results_df.head())
    results_df.to_csv("spike_results.csv" , index=False) #may need to adjust the results file path/name
else:
    print("No data found.")