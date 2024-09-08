import pandas as pd
import xarray as xr
import numpy as np

# Open the dataset
data_ts = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/EGLN/thunder_2014_2022.nc')
data_t_up = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_t_era_land.nc')
data_t_sfc = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_t_sfc_era_land.nc')
data_td_up = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_td_era_land.nc')
data_td_sfc = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_td_sfc_era_land.nc')
data_rh_up = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_rh_era_land.nc')
data_uwind_up = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_uwind_era_land.nc')
data_uwind_sfc = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_uwind_sfc_era_land.nc')
data_vwind_up = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_vwind_era_land.nc')
data_vwind_sfc = xr.open_dataset('/run/media/nos/HDPH-UT/climatologydata/ENV/data_vwind_sfc_era_land.nc')

# Function to initialize an array with NaN values
def land(inputdata):
    initial = inputdata.astype(float).copy(deep=True)  # Convert data type to float
    initial[:] = np.nan
    return initial

land_coord = pd.read_csv('/run/media/nos/HDPH-UT/SCRIPT/ENV/coord.csv')

# Create a subset of the data only for the land areas in Indonesia
data_ts_land = land(data_ts.ts_day)

for index, row in land_coord.iterrows():
    latitude, longitude = row['Latitude'], row['Longitude']
    data_ts_land.loc[:, latitude, longitude] = data_ts.ts_day.sel(latitude=latitude, longitude=longitude)

# List to store the results
grid_thunderstorm_data = []

# Iterate over all grids
for lat in data_ts_land.latitude.values:
    for lon in data_ts_land.longitude.values:
        days = data_ts_land.sel(latitude=lat, longitude=lon).dropna(dim="time")
        thunderstorm_date = days.where(days == 1, drop=True).time.values
        thunderstorm_date_list = thunderstorm_date.tolist()
        
        # Store thunderstorm days data in the list
        grid_thunderstorm_data.append({
            "latitude": lat,
            "longitude": lon,
            "thunderstorm_date": thunderstorm_date_list if thunderstorm_date_list else np.nan
        })

# Convert to Pandas DataFrame
df = pd.DataFrame(grid_thunderstorm_data)

# Function to convert timestamps to datetime
def convert_to_datetime(thunderstorm_date):
    if isinstance(thunderstorm_date, list):
        return [pd.to_datetime(day).to_numpy().astype('datetime64[ns]') for day in thunderstorm_date]
    return thunderstorm_date

# Apply the function to the thunderstorm_days column
df['thunderstorm_date'] = df['thunderstorm_date'].apply(convert_to_datetime)

# Separate TS and no TS days
thunder_dates = df.dropna(subset=['thunderstorm_date'])

# Function to calculate the average meteorological data
def calculate_averages(thunder_dates_group, data_t_up, data_td_up, data_uwind_up, data_vwind_up, data_rh_up):
    subset_results_temp = []
    subset_results_dew = []
    subset_results_uwind = []
    subset_results_vwind = []
    subset_results_rh = []

    for index, row in thunder_dates_group.iterrows():
        dates_list = pd.to_datetime(row['thunderstorm_date']).tolist()

        subset_temp = data_t_up.sel(
            time=xr.DataArray(dates_list, dims="time"),
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        subset_dew = data_td_up.sel(
            time=xr.DataArray(dates_list, dims="time"),
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        subset_uwind = data_uwind_up.sel(
            time=xr.DataArray(dates_list, dims="time"),
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        subset_vwind = data_vwind_up.sel(
            time=xr.DataArray(dates_list, dims="time"),
            latitude=row['latitude'],
            longitude=row['longitude']
        )
        subset_rh = data_rh_up.sel(
            time=xr.DataArray(dates_list, dims="time"),
            latitude=row['latitude'],
            longitude=row['longitude']
        )

        subset_results_temp.append(subset_temp)
        subset_results_dew.append(subset_dew)
        subset_results_uwind.append(subset_uwind)
        subset_results_vwind.append(subset_vwind)
        subset_results_rh.append(subset_rh)

    data_t_up_subset = xr.concat(subset_results_temp, dim='new_dim')
    data_td_up_subset = xr.concat(subset_results_dew, dim='new_dim')
    data_uwind_up_subset = xr.concat(subset_results_uwind, dim='new_dim')
    data_vwind_up_subset = xr.concat(subset_results_vwind, dim='new_dim')
    data_rh_up_subset = xr.concat(subset_results_rh, dim='new_dim')

    average_data_t_up = data_t_up_subset.mean(dim=['new_dim', 'time'], skipna=True)
    average_data_td_up = data_td_up_subset.mean(dim=['new_dim', 'time'], skipna=True)
    average_data_uwind_up = data_uwind_up_subset.mean(dim=['new_dim', 'time'], skipna=True)
    average_data_vwind_up = data_vwind_up_subset.mean(dim=['new_dim', 'time'], skipna=True)
    average_data_rh_up = data_rh_up_subset.mean(dim=['new_dim', 'time'], skipna=True)

    return average_data_t_up, average_data_td_up, average_data_uwind_up, average_data_vwind_up, average_data_rh_up

# Calculate averages for TS days
average_data_t_up_ts, average_data_td_up_ts, average_data_uwind_up_ts, average_data_vwind_up_ts, average_data_rh_up_ts = calculate_averages(thunder_dates, data_t_up, data_td_up, data_uwind_up, data_vwind_up, data_rh_up)

# Save the results to NetCDF files
average_data_t_up_ts.to_netcdf('/run/media/nos/HDPH-UT/climatologydata/ENV/average_data_t_up_ts.nc')
average_data_td_up_ts.to_netcdf('/run/media/nos/HDPH-UT/climatologydata/ENV/average_data_td_up_ts.nc')
average_data_uwind_up_ts.to_netcdf('/run/media/nos/HDPH-UT/climatologydata/ENV/average_data_uwind_up_ts.nc')
average_data_vwind_up_ts.to_netcdf('/run/media/nos/HDPH-UT/climatologydata/ENV/average_data_vwind_up_ts.nc')
average_data_rh_up_ts.to_netcdf('/run/media/nos/HDPH-UT/climatologydata/ENV/average_data_rh_up_ts.nc')
