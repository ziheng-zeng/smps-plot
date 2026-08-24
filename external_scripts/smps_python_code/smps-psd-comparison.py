import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib
import pytz
from mpl_toolkits.axes_grid1 import make_axes_locatable


### 1. Load SMPS data ###
path = 'D:/Documents/research-2024/PSD_SMPS/12-3' # change folder file path
csv_files = glob.glob(path + '/SMPS*.csv')
df_list = (pd.read_csv(files, skiprows=52) for files in csv_files)
df = pd.concat(df_list, ignore_index=True)
### 2. Calculate the lower bound and higher bound for each size bin ###
tsdf = df.set_index('DateTime Sample Start')
mid_D = np.array([float(x) for x in tsdf.columns[41:425]])
avg_diff = np.mean(np.diff(np.log10(mid_D)))
D_bound = np.full(mid_D.shape[0] + 1, np.nan)
for i in range(1, (len(D_bound) - 1)):
    D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i - 1])))
D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)
D_low = D_bound[0:-1]
D_high = D_bound[1:]
dlogDp = np.log10(D_high) - np.log10(D_low)
### 3. Calculate the total number of each scan ###
# calculate total number of each scan
artsdf = np.array(tsdf)
dNdlogDp = artsdf[:, 41:425]
dN = dNdlogDp * dlogDp
N = np.nansum(dN, axis=1)
### 5. Plot time series of number distribution  ###
# Convert time from UTC to Eastern Time
def adjust_timezone(dt_series, offset):
    # Adjusts 'dt_series' by 'offset' hours
    return dt_series + pd.DateOffset(hours=offset)

# Assuming 'df' has a datetime column named 'DateTime Sample Start' in UTC
df['DateTime Sample Start'] = pd.to_datetime(df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')
# Manually convert UTC to Eastern Time (assuming standard time, adjust as necessary for DST)
df['DateTime Sample Start'] = adjust_timezone(df['DateTime Sample Start'], -4)
Time = mdates.date2num(df['DateTime Sample Start'])  # This refers to the converted datetime

print(df['DateTime Sample Start'].min(), df['DateTime Sample Start'].max())
Y = mid_D.copy()  # Particle diameters
XX, YY = np.meshgrid(Time, Y)  # Create a meshgrid for Time (time) and Y (diameter)
Z = dNdlogDp.copy().T  # Transpose dNdlogDp to align with the meshgrid dimensions
# Logarithmic color scale setup
Z_numeric = np.asarray(Z, dtype=np.float64)  # Ensure Z is numeric for plotting
Z_numeric[Z_numeric == 0] = np.nanmin(Z_numeric[np.nonzero(Z_numeric)])
Z_masked_below_10 = np.ma.masked_less(Z_numeric, 10)



# norm = mcolors.LogNorm(vmin=1e3, vmax=1e5)
# cmap = cm.get_cmap('jet')
#
# fig, ax1 = plt.subplots(figsize=(16, 6))
# matplotlib.rcParams['timezone'] = 'US/Eastern'
# pcm = ax1.pcolormesh(XX, YY, Z_masked_below_10, shading='auto', cmap=cmap, norm=norm)
# cbar = fig.colorbar(pcm, ax=ax1, extend='neither', orientation = 'vertical', aspect=20, label=r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')
# ax1.xaxis_date()
# # Set y-axis to logarithmic scale
# ax1.set_yscale('log')
# ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
# ax1.set_ylim([10**1, 10**3])
#
# time_format = mdates.DateFormatter('%H:%M')  # Changed to display only hours and minutes
# ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Set major ticks to be every 4 hours
# ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Optional: Set minor ticks to be every hour
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Set formatter to display hours and minutes
# fig.autofmt_xdate()  # Auto format the date labels
# ax1.set_xlabel('Time (US/Eastern)')
# ax1.set_ylabel('Dp (nm)')
#
# plt.show()

# load PSD data
file_path = "D:/Documents/research-2024/PSD_SMPS/12-3-PSD/2024-08-23_SMPS.csv"
df2 = pd.read_csv(file_path)

# Assuming diameters and dNdlogDp are structured as described below

### First col is timestamp
### then first half of the remaining is diameter
### Other half is the inverted data
### for example the first row, it’s at 00:00, then diameter 3.946094 is corresponding to a dNdlogDp of 2382.401?
### no, need to cut off the first column of the diameters and inverted data
### So start from 1.339 nm diameter and that corresponds to 0 dNdlogDp?
### Then second is 1.46 corresponding to 0? Then going all the way like that ###

diameters2 = df2.iloc[0, 1:len(df2.columns)//2 + 1].values.astype(float)
dNdlogDp2 = df2.iloc[:, len(df2.columns)//2 + 1:] * 2.3025

# Replace zeros with NaN to facilitate log scale plotting
Z2 = dNdlogDp2.T.values  # Transpose to match the meshgrid dimensions
Z2[Z2 == 0] = np.nan  # Replace zeros with NaN

Time2 = pd.to_datetime(df2.iloc[:, 0])  # Assuming the first column is datetime
Time_numeric2 = mdates.date2num(Time2)

XX2, YY2 = np.meshgrid(Time_numeric2, diameters2)

print("XX shape:", XX.shape)
print("YY shape:", YY.shape)
print("Z shape:", Z.shape)

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(11, 6), sharex=True, sharey=True)

norm = mcolors.LogNorm(vmin=1e3, vmax=1e5)  # Shared normalization for color scales

# First dataset plot (Uncomment and use the prepared data)
pcm1 = ax1.pcolormesh(XX, YY, Z_masked_below_10, norm=norm, cmap='jet', shading='nearest')
ax1.set_title('Lawrenceville, Pittsburgh')
ax1.set_ylabel('Diameter (nm)')

# Second dataset plot
pcm2 = ax2.pcolormesh(XX2, YY2, Z2, norm=norm, cmap=cm.jet, shading='nearest')
ax2.set_title('Carnegie Mellon University, Pittsburgh')
ax2.set_xlabel('Time (US/Eastern)')
ax2.set_ylabel('Diameter (nm)')

ax1.xaxis_date()
# Set y-axis to logarithmic scale
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax1.set_ylim([10**0, 10**3])


time_format = mdates.DateFormatter('%H:%M')  # Changed to display only hours and minutes
ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Set major ticks to be every 4 hours
ax1.xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Optional: Set minor ticks to be every hour
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Set formatter to display hours and minutes

# Colorbar (shared for both subplots)
# fig.colorbar(pcm1, ax=[ax1, ax2], orientation='vertical', label=r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="2%", pad=0.1)
colorbar1 = fig.colorbar(pcm1, cax=cax1)
colorbar1.set_label(r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')

divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="2%", pad=0.1)
colorbar2 = fig.colorbar(pcm2, cax=cax2)
colorbar2.set_label(r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')

# Adding custom text at the bottom left
# ax1.text(0.01, 0.01, 'Aug 23', color='red', fontsize=12, transform=ax1.transAxes, verticalalignment='bottom')


fig.subplots_adjust(right=0.8)

fig.autofmt_xdate()
plt.tight_layout()
plt.show()