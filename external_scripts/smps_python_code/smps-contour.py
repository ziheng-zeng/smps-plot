import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm

### 1. Load SMPS file ###
path = ''
csv_files = glob.glob(path + '/SMPS*.csv')
df = pd.read_csv(csv_files[0], skiprows=52)

### 2. Parse and convert time to Eastern ###
df['DateTime Sample Start'] = pd.to_datetime(df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')
df['DateTime Sample Start'] = df['DateTime Sample Start'] + pd.DateOffset(hours=-4)

### 3. Extract dNdlogDp and diameter bins ###
mid_D = np.array([float(x) for x in df.columns[41:425]])
dNdlogDp = df.iloc[:, 41:425].values

# Calculate bin width
avg_diff = np.mean(np.diff(np.log10(mid_D)))
D_bound = np.empty(mid_D.size + 1)
D_bound[1:-1] = 10 ** (0.5 * (np.log10(mid_D[1:]) + np.log10(mid_D[:-1])))
D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)
D_low = D_bound[:-1]
D_high = D_bound[1:]
dlogDp = np.log10(D_high) - np.log10(D_low)

# Total number per scan (not used in plot below, but calculated)
N = np.nansum(dNdlogDp * dlogDp, axis=1)

### 4. Prepare for plotting ###
time_num = mdates.date2num(df['DateTime Sample Start'])
XX, YY = np.meshgrid(time_num, mid_D)
Z = dNdlogDp.T
Z = np.where(Z < 10, np.nan, Z)

### 5. Plot ###
fig, ax = plt.subplots(figsize=(12, 5))
norm = mcolors.LogNorm(vmin=1e3, vmax=1e5)
pcm = ax.pcolormesh(XX, YY, Z, shading='auto', cmap='jet', norm=norm)

ax.set_yscale('log')
ax.set_ylim(10, 1000)
ax.set_ylabel('Dp (nm)')
ax.set_xlabel('Time (US/Eastern)')
ax.set_title('SMPS Number Distribution (First File Only)')

# Format time axis
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
fig.autofmt_xdate()

# Colorbar
cbar = fig.colorbar(pcm, ax=ax, label=r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')

plt.tight_layout()
plt.show()
