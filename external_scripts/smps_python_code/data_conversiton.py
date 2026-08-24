import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
from matplotlib import colormaps

# Load data from a file
file_path = 'D:/Documents/research-2024/SMPS data/SMPS_20240921.csv'
df = pd.read_csv(file_path, skiprows=52)

# Set the index to 'DateTime Sample Start' and clean column names
df.columns = df.columns.str.strip()  # Clean any leading/trailing spaces in column names
tsdf = df.set_index('DateTime Sample Start')

# Clean up the columns by removing the leading underscores before converting to float
cleaned_columns = [col.lstrip('_') for col in tsdf.columns[41:425]]  # Remove leading underscores
mid_D = np.array([float(x) for x in cleaned_columns])

# Calculate size bin bounds
avg_diff = np.mean(np.diff(np.log10(mid_D)))
D_bound = np.full(mid_D.shape[0] + 1, np.nan)
for i in range(1, len(D_bound) - 1):
    D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i - 1])))

D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)

D_low = D_bound[:-1]
D_high = D_bound[1:]
dlogDp = np.log10(D_high) - np.log10(D_low)

# Calculate total number concentration (N)
artsdf = np.array(tsdf)
dNdlogDp = artsdf[:, 41:425]
dN = dNdlogDp * dlogDp
N = np.nansum(dN, axis=1)

# Plotting the time series of dNdlogDp
# Convert 'DateTime Sample Start' to datetime format for plotting
Time = pd.to_datetime(df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')

# Convert datetime to matplotlib date numbers
X = mdates.date2num(Time)  # Time values
Y = mid_D.copy()  # Particle diameters
XX, YY = np.meshgrid(X, Y)  # Create a meshgrid for X (time) and Y (diameter)

# Transpose dNdlogDp for proper alignment with the meshgrid
Z = dNdlogDp.T

# Handle zeros and ensure values are compatible with LogNorm by replacing 0 with the minimum non-zero value
Z_numeric = np.asarray(Z, dtype=np.float64)  # Ensure numeric type
Z_numeric[Z_numeric == 0] = np.nanmin(Z_numeric[np.nonzero(Z_numeric)])

# Mask values below 10 to ensure uniform color for lower bound
Z_masked_below_10 = np.ma.masked_less(Z_numeric, 10)

# Use LogNorm for logarithmic color scaling
norm = mcolors.LogNorm(vmin=10, vmax=1e5)

# Set up the colormap (using 'jet' colormap)
cmap = colormaps['jet']

# Plotting the data
fig, ax1 = plt.subplots(figsize=(16, 6))
pcm = ax1.pcolormesh(XX, YY, Z_masked_below_10, shading='auto', cmap=cmap, norm=norm)

# Add a colorbar
cbar = fig.colorbar(pcm, ax=ax1, extend='neither', orientation='vertical', aspect=20, label=r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=15)
cbar.set_ticks([10 ** i for i in range(1, 6)])

# Format the x-axis as dates
ax1.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()  # Auto format the date labels
ax1.set_ylim(0, 300)

# Set labels
ax1.set_xlabel('Time (UTC)')
ax1.set_ylabel('Dp (nm)')

# Show the plot
plt.show()
