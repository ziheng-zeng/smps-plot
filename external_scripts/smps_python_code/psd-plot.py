import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib import cm
import matplotlib.ticker as ticker

# Load the data
file_path = "D:/Documents/research-2024/PSD_SMPS/12-3-PSD/2024-08-23_SMPS.csv"
df2 = pd.read_csv(file_path)

# Assuming diameters and dNdlogDp are structured as described
diameters = df2.iloc[0, 1:len(df2.columns)//2 + 1].values.astype(float)
dNdlogDp = df2.iloc[:, len(df2.columns)//2 + 1:] * 2.3025

# Replace zeros with NaN to facilitate log scale plotting
Z = dNdlogDp.T.values  # Transpose to match the meshgrid dimensions
Z[Z == 0] = np.nan  # Replace zeros with NaN

Time = pd.to_datetime(df2.iloc[:, 0])  # Assuming the first column is datetime
Time_numeric = mdates.date2num(Time)  # Convert datetime to Matplotlib's format

XX, YY = np.meshgrid(Time_numeric, diameters)

print("XX2 shape:", XX.shape)
print("YY2 shape:", YY.shape)
print("Z2 shape:", Z.shape)

# Create the plot
fig, ax = plt.subplots(figsize=(15, 7))
norm = mcolors.LogNorm(vmin=1e3, vmax=1e5)
pcm = ax.pcolormesh(XX, YY, Z, norm=norm, cmap=cm.jet)

# Add color bar
cbar = fig.colorbar(pcm, ax=ax, extend='both')
cbar.set_label('dNdlogDp (scaled)')

# Set the y-axis to log scale
ax.set_yscale('log')
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

# Set axis labels and title
ax.set_xlabel('Time')
ax.set_ylabel('Diameter (nm)')
ax.set_title('Time vs. Diameter Distribution')

# Format the x-axis with time labels
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

plt.show()