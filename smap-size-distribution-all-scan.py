import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

### 1. Load data ###
# get .csv files list from a folder
path = 'D:/Documents/Research spring 24/SMPS/data-practice'  # change file path
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
D_bound = np.full(mid_D.shape[0]+1, np.nan)
for i in range (1, (len(D_bound)-1)):
    D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i-1])))

D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5*avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5*avg_diff)
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
dVdlogDp = (np.pi/6.) * (mid_D/1e3) **3 * dNdlogDp  #um3/cm3
dV = dVdlogDp * dlogDp
V = np.nansum(dV, axis=1)

# calculate mass distribution and total mass of each scan by assuming a particle density
density = 1.4        #g/cm3
dMdlogDp = (density/1e9) * (np.pi/6.) * mid_D**3 * dNdlogDp        #ug/cm3
dM = dMdlogDp * dlogDp
M = np.nansum(dM, axis=1)     #ug/m3
# print(dM.shape)

# calculate surface area distribution and total surface area of each scan
# assuming mid_D is in nanometers (nm) and needs to be converted to cm for surface area calculations in cm^2
# convert mid_D from nm to cm (1 nm = 1e-7 cm)
mid_D_cm = mid_D * 1e-7  # Convert diameter from nanometers to centimeters
# calculate surface area distribution (dSdlogDp) in cm^2/cm^3
dSdlogDp = (np.pi) * mid_D_cm**2 * dNdlogDp  # Surface area distribution: cm^2/cm^3
# calculate the differential surface area (dS) in each bin and total surface area (S) of each scan
dS = dSdlogDp * dlogDp  # Differential surface area for each bin: cm^2/cm^3
S = np.nansum(dS, axis=1)  # Total surface area of each scan: cm^2/cm^3
# print(dS.shape)  # To verify the shape of the surface area distribution matrix


### 4. Plot lognormal number/volume/mass/surface area distribution for each scan ###
def plot_hist(ax, time_index, var, bound, ylabel, title):
    ax.bar(bound[:-1], var[time_index, :], width=np.diff(bound), align='edge')
    ax.set_xlim(10, 1000)
    ax.set_xscale('log')
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_title(title, fontsize=17)


# Assuming you have an array or list for each distribution type and D_bound is defined
# Example: dNdlogDp = np.array(...) where dNdlogDp[time_index, :] gives the distribution for a specific scan
# Similarly for dVdlogDp, dMdlogDp, dSdlogDp, and D_bound

# Iterate over each scan time in tsdf
for time_index, scan_time in enumerate(tsdf.index):
    fig, axs = plt.subplots(4, 1, figsize=(16, 20), sharex=True)
    plt.suptitle(f'Scan Time: {scan_time}', fontsize=20)

    # Update these calls with the correct variables and labels
    plot_hist(axs[0], time_index, dNdlogDp, D_bound, r'$\frac{dN}{d\log D_p}$ (cts cm$^{-3}$)',
              'Number Size Distribution')
    plot_hist(axs[1], time_index, dVdlogDp, D_bound, r'$\frac{dV}{d\log D_p}$ (μm$^3$ cm$^{-3}$)',
              'Volume Size Distribution')
    plot_hist(axs[2], time_index, dMdlogDp, D_bound, r'$\frac{dM}{d\log D_p}$ (μg m$^{-3}$)', 'Mass Size Distribution')
    plot_hist(axs[3], time_index, dSdlogDp, D_bound, r'$\frac{dS}{d\log D_p}$ (cm$^2$ cm$^{-3}$)',
              'Surface Area Size Distribution')

    axs[3].set_xlabel('Dp (nm)', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect to make space for the suptitle
    # plt.show()

    # Print the current working directory
    print(os.getcwd())

    # Change the current working directory
    os.chdir('D:/Documents/research spring 24/SMPS/jan size distribution plots')
    print(os.getcwd())

    # Assuming scan_time is a string like "10/01/2024 00:00:10"
    # Replace slashes and colons with dashes and underscores
    formatted_scan_time = scan_time.replace('/', '-').replace(' ', '_').replace(':', '-')

    # Now, saving a plot will use the new current working directory as the base path
    plt.savefig(f'plot_{formatted_scan_time}.png', dpi=300)