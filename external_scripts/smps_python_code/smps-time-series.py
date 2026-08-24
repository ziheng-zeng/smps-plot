import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib import colormaps



### 1. Load data ###
# get .csv files list from a folder
path = 'D:/Documents/research-2024/PSD_SMPS/12-3' # change file path
csv_files = glob.glob(path + '/SMPS*.csv')

# read each file in to DataFrame
# create a list of dataframes
df_list = (pd.read_csv(files, skiprows=52) for files in csv_files)

# concatenate all dataframes
df = pd.concat(df_list, ignore_index=True)

### 2. Calculate the lower bound and higher bound for each size bin ###
# change index to datetime to create a TimeSeries DataFrame
tsdf = df.set_index('DateTime Sample Start')

# adjust the line to handle strings with non-numeric characters before conversion to float
# using regular expressions to remove any leading non-numeric characters
mid_D = np.array([float(x) for x in tsdf.columns[41:425]])

# calculate the average difference of the logarithmic midpoints
avg_diff = np.mean(np.diff(np.log10(mid_D)))

# print the shape of mid_D to verify the results
# print(mid_D.shape)

# calculate the low bound and higher bound of each bin
D_bound = np.full(mid_D.shape[0] + 1, np.nan)
for i in range(1, (len(D_bound) - 1)):
    D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i - 1])))

D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)
# print(D_bound)

D_low = D_bound[0:-1]
D_high = D_bound[1:]
dlogDp = np.log10(D_high) - np.log10(D_low)

### 3. Calculate the total number/volume/mass/surface area (N, V, M, S) of each scan, and the lognormal volume/mass/surface area distribution (dVdlogDp, dMdlogDp, dSdlogDp) ###
# calculate total number of each scan
artsdf = np.array(tsdf)
dNdlogDp = artsdf[:, 41:425]
dN = dNdlogDp * dlogDp
N = np.nansum(dN, axis=1)

# calculate volume distribution and total volume of each scan
dVdlogDp = (np.pi / 6.) * (mid_D / 1e3) ** 3 * dNdlogDp  # um3/cm3
dV = dVdlogDp * dlogDp
V = np.nansum(dV, axis=1)

# calculate mass distribution and total mass of each scan by assuming a particle density
density = 1.4  # g/cm3
dMdlogDp = (density / 1e9) * (np.pi / 6.) * mid_D ** 3 * dNdlogDp  # ug/cm3
dM = dMdlogDp * dlogDp
M = np.nansum(dM, axis=1)  # ug/m3
# print(dM.shape)

# calculate surface area distribution and total surface area of each scan
# assuming mid_D is in nanometers (nm) and needs to be converted to cm for surface area calculations in cm^2
# convert mid_D from nm to cm (1 nm = 1e-7 cm)
mid_D_cm = mid_D * 1e-7  # Convert diameter from nanometers to centimeters
# calculate surface area distribution (dSdlogDp) in cm^2/cm^3
dSdlogDp = (np.pi) * mid_D_cm ** 2 * dNdlogDp  # Surface area distribution: cm^2/cm^3
# calculate the differential surface area (dS) in each bin and total surface area (S) of each scan
dS = dSdlogDp * dlogDp  # Differential surface area for each bin: cm^2/cm^3
S = np.nansum(dS, axis=1)  # Total surface area of each scan: cm^2/cm^3
# print(dS.shape)  # To verify the shape of the surface area distribution matrix

### 5. Plot time series of number distribution  ###
# plot time series of dNdlogDp
# Convert 'DateTime Sample Start' to datetime format and then to matplotlib date numbers
Time = pd.to_datetime(df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')
X = mdates.date2num(Time)  # This should refer to the converted datetime
Y = mid_D.copy()  # Particle diameters
XX, YY = np.meshgrid(X, Y)  # Create a meshgrid for X (time) and Y (diameter)
Z = dNdlogDp.copy().T  # Transpose dNdlogDp to align with the meshgrid dimensions
# Ensure Z is numeric and replace zeros with the smallest non-zero value for log scale compatibility
Z_numeric = np.asarray(Z, dtype=np.float64)  # Ensure Z is numeric for plotting
Z_numeric[Z_numeric == 0] = np.nanmin(Z_numeric[np.nonzero(Z_numeric)])

# Mask values below 10 to ensure they are plotted with the same color as 10
Z_masked_below_10 = np.ma.masked_less(Z_numeric, 10)

# Use LogNorm for logarithmic scaling starting from 10 to the maximum value
norm = mcolors.LogNorm(vmin=10, vmax=1e5)

# Use the 'jet' colormap
cmap = colormaps['jet']

fig, ax1 = plt.subplots(figsize=(16, 6))
pcm = ax1.pcolormesh(XX, YY, Z_masked_below_10, shading='auto', cmap=cmap, norm=norm)

cbar = fig.colorbar(pcm, ax=ax1, extend='neither', orientation = 'vertical', aspect=20, label=r'$\frac{dN}{dlogDp}$ (cts cm$^{-3}$)')
cbar.ax.tick_params(labelsize=15)
cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=15)
cbar.set_ticks([10**i for i in range(1, 6)])

# Set the x-axis as date
ax1.xaxis_date()
date_format = mdates.DateFormatter('%m/%d/%Y')
ax1.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()  # Auto format the date labels

# Set labels
ax1.set_xlabel('Time')
ax1.set_ylabel('Dp (nm)')

plt.show()

# ### 6. calculate 1-hour averaged number/volume/mass/surface area and plot time series of raw vs. 1-hour averaged###
# # Convert 'DateTime Sample Start' to datetime and set as index
# df['DateTime Sample Start'] = pd.to_datetime(df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')
# tsdf = df.set_index('DateTime Sample Start')
#
# # Function to convert arrays to DataFrame columns
# def to_df(dataframe, var, name_in_df):
#     dataframe[name_in_df] = var
#     dataframe[name_in_df] = dataframe[name_in_df].astype(float)
#
# # Assuming N, V, M, S are numpy arrays or lists with the correct values
# to_df(tsdf, N, 'N_total')
# to_df(tsdf, V, 'V_total')
# to_df(tsdf, M, 'M_total')
# to_df(tsdf, S, 'S_total')
#
# # Select only numeric columns for resampling
# tsdf_numeric = tsdf.select_dtypes(include=[np.number])
#
# # Calculate 1-hour average
# avg_plotdf = tsdf_numeric.resample('1h').mean()
#
# # Plotting function for time series
# def plot_t_series(ax, raw_df, avg_df, var, ylabel):
#     ax.plot(raw_df.index, raw_df[var], label='Raw', alpha=0.5)
#     ax.plot(avg_df.index, avg_df[var], linewidth=2.5, label='1 Hour Average')
#     date_fmt = DateFormatter('%m/%d/%Y')
#     ax.xaxis.set_major_formatter(date_fmt)
#     ax.figure.autofmt_xdate()  # Auto-format x-axis dates
#     ax.tick_params(axis='both', which='major', labelsize=15)
#     ax.set_ylabel(ylabel, fontsize=15)
#     ax.legend(loc='upper right', fontsize=12)
#
# # Plotting raw vs. 1-hour averaged time series
# fig, axs = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
#
# plot_t_series(axs[0], tsdf, avg_plotdf, 'N_total', 'Number (cts cm$^{-3}$)')
# plot_t_series(axs[1], tsdf, avg_plotdf, 'V_total', 'Volume (μm$^3$ cm$^{-3}$)')
# plot_t_series(axs[2], tsdf, avg_plotdf, 'M_total', 'Mass (μg m$^{-3}$)')
# plot_t_series(axs[3], tsdf, avg_plotdf, 'S_total', 'Surface Area (cm$^2$ cm$^{-3}$)')
#
# axs[-1].set_xlabel('Time', fontsize=15)
# plt.tight_layout()
# plt.show()