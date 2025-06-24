import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from isodisreg import idr
import os


# get data ----------------------------------------------------

obs_paths = {
    'era5':          'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr',
    'ifs_analysis':  'gs://weatherbench2/datasets/hres_t0/2016-2022-6h-64x32_equiangular_conservative.zarr',
}

forecast_paths = {
    'hres':                    'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr',
    'pangu':                   'gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr',
    'graphcast':               'gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr',
    'pangu_operational':       'gs://weatherbench2/datasets/pangu_hres_init/2020_0012_64x32_equiangular_conservative.zarr',
    'graphcast_operational':   'gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours-64x32_equiangular_conservative.zarr',
}

# local data path (needs to be chosen)
save_path = 'data/'

# defining variables of interest
variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_wind_speed',
    ]

lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
    ]

time_range = slice('2020-01-01','2020-12-31')

for name, path in obs_paths.items():
    local_zarr = save_path + f'{name}_64x32.zarr'
    if os.path.exists(local_zarr):
        print(f"{local_zarr} already exists. Skipping...")
        continue  # Skip loading if the file already exists
    ds = xr.open_zarr(
        store=path, 
        storage_options={'token': 'anon'},
        decode_timedelta=True
        ).sel(time=slice('2020-01-01','2021-01-10'))[variables].drop_encoding() # drop encoding to avoid overlap errors
    
    ds = ds.sel(time=ds.time.dt.hour.isin([0, 12])) # only 00 and 12 UTC times are relevant since we only have full day lead times

    # rechunk
    ds = ds.chunk({'time': 1, 'latitude': 64, 'longitude': 32}) ## ???

    with ProgressBar(): # not necessary but shows progress
        ds.to_zarr(save_path + f'{name}_64x32.zarr', mode='w', consolidated=True) # make sure save_path ends in /


for name, path in forecast_paths.items():
    local_zarr = save_path + f'{name}_64x32.zarr'
    if os.path.exists(local_zarr):
        print(f"{local_zarr} already exists. Skipping...")
        continue  # Skip loading if the file already exists
    ds = xr.open_zarr(
        store=path, 
        storage_options={'token': 'anon'},
        decode_timedelta=True
        ).sel(time=time_range, prediction_timedelta=lead_times)[variables].drop_encoding()

    # rechunk
    ds = ds.chunk({'time': 1, 'latitude': 64, 'longitude': 32})

    with ProgressBar():
        ds.to_zarr(save_path + f'{name}_64x32.zarr', mode='w', consolidated=True)      


