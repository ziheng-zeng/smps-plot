import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np

# Load ACSM data
acsm_df = pd.read_csv("D:/Documents/research spring 24/ACSM data/JAN_FEB_2024_1.csv")

# Load SMPS data
path = 'D:/Documents/Research spring 24/SMPS data/data-jan_feb' # change file path
csv_files = glob.glob(path + '/SMPS*.csv')
df_list = (pd.read_csv(files, skiprows=52) for files in csv_files)
smps_df = pd.concat(df_list, ignore_index=True)


# Sum the mass concentrations for ACSM
acsm_df['Total_Mass_Conc_ACSM'] = acsm_df[['Org_11100', 'SO4_11100', 'NO3_11100', 'NH4_11100', 'Chl_11100']].sum(axis=1)

# # Define a threshold for "close to zero" and filter out those values from ACSM data
# close_to_zero_threshold = 0.15  # Adjust this threshold as needed
# acsm_df = acsm_df[acsm_df['Total_Mass_Conc_ACSM'] > close_to_zero_threshold]

# Get SMPS mass concentrations
tsdf = smps_df.set_index('DateTime Sample Start')
mid_D = np.array([float(x) for x in tsdf.columns[41:425]])
# calculate the average difference of the logarithmic midpoints
avg_diff = np.mean(np.diff(np.log10(mid_D)))
# calculate the low bound and higher bound of each bin
D_bound = np.full(mid_D.shape[0] + 1, np.nan)
for i in range(1, (len(D_bound) - 1)):
    D_bound[i] = 10 ** (0.5 * (np.log10(mid_D[i]) + np.log10(mid_D[i - 1])))
D_bound[0] = 10 ** (np.log10(mid_D[0]) - 0.5 * avg_diff)
D_bound[-1] = 10 ** (np.log10(mid_D[-1]) + 0.5 * avg_diff)
D_low = D_bound[0:-1]
D_high = D_bound[1:]
dlogDp = np.log10(D_high) - np.log10(D_low)

artsdf = np.array(tsdf)
dNdlogDp = artsdf[:, 41:425]
dN = dNdlogDp * dlogDp
N = np.nansum(dN, axis=1)
# calculate mass distribution and total mass of each scan by assuming a particle density
density = 1.4  # g/cm3
dMdlogDp = (density / 1e9) * (np.pi / 6.) * mid_D ** 3 * dNdlogDp  # ug/cm3
dM = dMdlogDp * dlogDp
M = np.nansum(dM, axis=1)  # ug/m3

smps_df['Total_Mass_Conc_SMPS'] = pd.to_numeric(M, errors='coerce')

# Ensure datetime formats are consistent and set as index
acsm_df['t_base'] = pd.to_datetime(acsm_df['t_base'])
smps_df['DateTime Sample Start'] = pd.to_datetime(smps_df['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S')
smps_df.rename(columns={'DateTime Sample Start': 't_base'}, inplace=True)

# Select numeric columns only for resampling
numeric_cols_acsm = acsm_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_smps = smps_df.select_dtypes(include=[np.number]).columns.tolist()

# Resample data to hourly means, using only numeric columns
acsm_df_numeric = acsm_df.set_index('t_base')[numeric_cols_acsm].resample('1h').mean().reset_index()
smps_df_numeric = smps_df.set_index('t_base')[numeric_cols_smps].resample('1h').mean().reset_index()

# Merge the datasets on the 't_base' timestamp column
merged_df = pd.merge(acsm_df_numeric, smps_df_numeric, on='t_base', suffixes=('_ACSM', '_SMPS'))

# Find overlapping dates between the two datasets
overlapping_dates = acsm_df['t_base'].isin(smps_df['t_base'])
print("Overlapping dates:", acsm_df['t_base'][overlapping_dates].unique())

# Check if merged_df is empty
if merged_df.empty:
    print("No overlapping data found. Please check the datasets' date ranges.")
else:
    # Assuming 'x' and 'y' are your data series or arrays for SMPS and ACSM, respectively.
    x = merged_df['Total_Mass_Conc_SMPS']
    y = merged_df['Total_Mass_Conc_ACSM']
    # Create a mask that will be True for indices where neither x nor y is NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    # Apply the mask to both series to remove any pairs where either is NaN
    x_masked = x[mask]
    y_masked = y[mask]

    # Now you can safely perform linear regression on the cleaned data
    slope, intercept, r_value, p_value, std_err = linregress(x_masked, y_masked)

    # Find the timestamp where the ACSM value is close to zero
    # We choose 0.1 as an arbitrary threshold for "close to zero"
    close_to_zero_threshold = 0.15
    timestamps_close_to_zero = merged_df[mask & (y_masked < close_to_zero_threshold)]['t_base']
    print("Timestamps where ACSM is close to zero:", timestamps_close_to_zero)

    # Plot the data and the regression line
    plt.scatter(x_masked, y_masked, alpha=0.5)
    plt.plot(x_masked, intercept + slope * x_masked, 'r', label='fitted line')
    plt.annotate(f'r$^{2}$={r_value ** 2:.2f}; slope={slope:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    # Add labels and title
    plt.xlabel('SMPS (µg/m$^{3}$)')
    plt.ylabel('Total Chem Resolved PM (µg/m$^{3}$)')
    plt.title('Comparison of ACSM and SMPS Data')
    # Show plot
    plt.show()