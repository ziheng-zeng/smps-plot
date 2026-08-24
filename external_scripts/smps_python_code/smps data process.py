import glob
import pandas as pd

# get .csv files list from a folder
path = 'D:/Documents/Research spring 24/SMPS data/data-all-time' # change file path
csv_files = glob.glob(path + '/SMPS*.csv')

# read each file in to DataFrame
# create a list of dataframes
df_list = (pd.read_csv(files, skiprows=52) for files in csv_files)

# concatenate all dataframes
data = pd.concat(df_list, ignore_index=True)
data.to_csv('smps-all-time.csv')



# Load the dataset focusing only on the 'DateTime Sample Start' column
file_path = 'D:/Documents/Research spring 24/SMPS python code/smps-all-time.csv'
data = pd.read_csv(file_path, usecols=['DateTime Sample Start'], low_memory=False)

# Display the first few rows to inspect the format
print(data.head(10))

# Convert the 'DateTime Sample Start' column to datetime format with more flexibility
data['DateTime Sample Start'] = pd.to_datetime(data['DateTime Sample Start'], format='%d/%m/%Y %H:%M:%S', errors='coerce', dayfirst=True)

# Drop any rows with NaT values resulting from conversion errors
data = data.dropna(subset=['DateTime Sample Start'])

# Analysis by hour
data['Hour'] = data['DateTime Sample Start'].dt.floor('h')
unique_hours = data['Hour'].nunique()
total_hours_in_year = 8760

# Analysis by day
data['Day'] = data['DateTime Sample Start'].dt.date
unique_days = data['Day'].nunique()
total_days_in_year = 365

# Output the results
results = {
    'unique_hours': unique_hours,
    'total_hours_in_year': total_hours_in_year,
    'unique_days': unique_days,
    'total_days_in_year': total_days_in_year,
    'hours_collected_percentage': (unique_hours / total_hours_in_year) * 100,
    'days_collected_percentage': (unique_days / total_days_in_year) * 100
}

# Display the results
print(f"Collected {results['unique_hours']} hours of data out of the possible {results['total_hours_in_year']} hours in the year.")
print(f"Collected data on {results['unique_days']} days out of the possible {results['total_days_in_year']} days in the year.")
print(f"Percentage of hours collected: {results['hours_collected_percentage']:.2f}%")
print(f"Percentage of days collected: {results['days_collected_percentage']:.2f}%")