import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from matplotlib.colors import LogNorm
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib import colormaps

# Step 1: Load the CSV, skipping rows and setting the header
df = pd.read_csv("D:/Documents/research spring 24/SMPS data/rwanda/Kigali_SMPS_2024_01_30.1_no_raw_data.csv", skiprows=17)
save_dir = "D:/Documents/research spring 24/SMPS data/rwanda/day-plots"

# Get the datetime into one string
df['Date'] = df['Date'].astype(str)
df['Start Time'] = df['Start Time'].astype(str)
# Strip any leading/trailing whitespace
df['Date'] = df['Date'].str.strip()
df['Start Time'] = df['Start Time'].str.strip()
# Concatenate and convert to datetime, specifying the format
df['Time'] = pd.to_datetime(df['Date'] + ' ' + df['Start Time'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
# Extracting Dp values from the specified range (columns "I" to "GR") in row 18 of the original DataFrame
#dp_values = df.iloc[0, 8:200]  # Assuming the first non-header row (index 0 after skiprows=17) contains Dp values

# Move the last time column to the first position
columns = list(df.columns)
new_order = [columns[-1]] + columns[:-1]
df = df[new_order]
df.fillna(0, inplace=True)
tsdf = df.set_index('Time')
# print(tsdf.head())

# using regular expressions to remove any leading non-numeric characters
mid_D = np.array([float(x) for x in tsdf.columns[8:200]])

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


# Group data by day
grouped = tsdf.groupby(pd.Grouper(freq='D'))

for name, group in grouped:
    if group.empty:
        continue  # Skip empty groups

    ### 3. Calculate the total number/volume/mass/surface area (N, V, M, S) of each scan, and the lognormal volume/mass/surface area distribution (dVdlogDp, dMdlogDp, dSdlogDp) ###

    dNdlogDp = group.iloc[:, 8:200]
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

    # Proceed to plot for each day

    ### 5. Plot time series of number distribution  ###
    # plot time series of dNdlogDp
    # Convert 'DateTime Sample Start' to datetime format and then to matplotlib date numbers

    group_time = group.index  # Assuming 'Time' is already the index
    X = mdates.date2num(group_time.to_pydatetime())
    Y = mid_D.copy()  # Particle diameters
    XX, YY = np.meshgrid(X, Y)  # Create a meshgrid for X (time) and Y (diameter)
    Z = dNdlogDp.copy().T  # Transpose dNdlogDp to align with the meshgrid dimensions
    # Ensure Z is numeric and replace zeros with the smallest non-zero value for log scale compatibility
    Z_numeric = np.asarray(Z, dtype=np.float64)  # Ensure Z is numeric for plotting
    Z_numeric[Z_numeric == 0] = np.nanmin(Z_numeric[np.nonzero(Z_numeric)])

    # Mask values below 10 to ensure they are plotted with the same color as 0
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
    date_format = mdates.DateFormatter('%H:%M')
    ax1.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()

    # Set labels
    ax1.set_title(name.strftime('%Y-%m-%d'))  # Add a title with the date
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dp (nm)')

    # Define the full path with a custom filename
    # For example, using the date (name) to name the file
    filename = f"{name.strftime('%Y-%m-%d')}.png"
    full_path = os.path.join(save_dir, filename)
    # Save the figure to the specified directory with the custom filename
    plt.savefig(full_path)
    # Optionally, close the figure after saving to free up memory
    plt.close(fig)

# ### 6. calculate 1-hour averaged number/volume/mass/surface area and plot time series of raw vs. 1-hour averaged###
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

# ### 4. Plot lognormal number/volume/mass/surface area distribution for a specific scan ###
# time = 6321    # change this for specific scan
# print(tsdf.index[time])
# def plot_hist(ax,time_index,var,bound,ylabel):
#     ax.bar(bound[:-1],var[time_index,:],width=np.diff(bound),align='edge')
#     ax.set_xlim(10,1000)
#     ax.set_xscale('log')
#     ax.set_ylabel(ylabel, fontsize=15)
#     ax.tick_params(axis='both',which='major',labelsize=15)
#
# fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
#
# # assuming `time` is a valid index within your DataFrame's bounds
# # update these calls with the correct variables and labels
# plot_hist(ax1, time, dNdlogDp, D_bound, r'$\frac{dN}{d\log D_p}$ (cts cm$^{-3}$)')
# plot_hist(ax2, time, dVdlogDp, D_bound, r'$\frac{dV}{d\log D_p}$ (μm$^3$ cm$^{-3}$)')
# plot_hist(ax3, time, dMdlogDp, D_bound, r'$\frac{dM}{d\log D_p}$ (μg m$^{-3}$)')
# plot_hist(ax4, time, dSdlogDp, D_bound, r'$\frac{dS}{d\log D_p}$ (cm$^2$ cm$^{-3}$)')
#
# ax4.set_xlabel('Dp (nm)', fontsize=15)
# plt.tight_layout()
# plt.show()
#
