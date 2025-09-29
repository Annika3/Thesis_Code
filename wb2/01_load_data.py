# load_data.py
import numpy as np
import xarray as xr
import pandas as pd
from dask.diagnostics import ProgressBar
from isodisreg import idr  # kept for downstream compatibility
import os

# --------------------------------------------------------------------
# Constants and paths
# --------------------------------------------------------------------

# Remote object storage locations (WeatherBench2)
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

# Local target directory for Zarr outputs (co-located with this script)
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "data")
os.makedirs(save_dir, exist_ok=True)
save_path = save_dir

# Variables of interest for obs/forecast datasets
variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_wind_speed',
]

# Lead times used for both model forecasts and climatology rolling
lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D'),
]

# Time range used for 2020 evaluation
time_range = slice('2020-01-01', '2020-12-31')


# --------------------------------------------------------------------
# Load functions
# --------------------------------------------------------------------

def load_classic_zarr_data() -> None:
    """
    Load reanalysis/analysis and forecast datasets from WB2.

    The datasets are subset to the specified variables and time range, restricted
    to 00/12 UTC cycles for consistency with full-day lead times, rechunked for
    I/O efficiency, and stored as consolidated Zarr.
    """
    # Reanalysis / analysis
    for name, path in obs_paths.items():
        local_zarr = os.path.join(save_path, f'{name}_64x32.zarr')
        if os.path.exists(local_zarr):
            print(f"{local_zarr} already exists. Skipping...")
            continue  # Skip if already materialized locally

        ds = (
            xr.open_zarr(store=path, storage_options={'token': 'anon'}, decode_timedelta=True)
              .sel(time=slice('2020-01-01', '2021-01-10'))[variables]
              .drop_encoding()  # avoid encoding conflicts on write
        )

        # Restrict to 00 and 12 UTC cycles to align with daily lead times
        ds = ds.sel(time=ds.time.dt.hour.isin([0, 12]))

        # Rechunk: one timestep per chunk, full spatial slab per chunk
        ds = ds.chunk({'time': 1, 'latitude': 64, 'longitude': 32})

        with ProgressBar():  # prints progress of the write
            ds.to_zarr(local_zarr, mode='w', consolidated=True)

    # Forecast models
    for name, path in forecast_paths.items():
        local_zarr = os.path.join(save_path, f'{name}_64x32.zarr')
        if os.path.exists(local_zarr):
            print(f"{local_zarr} already exists. Skipping...")
            continue

        ds = (
            xr.open_zarr(store=path, storage_options={'token': 'anon'}, decode_timedelta=True)
              .sel(time=time_range, prediction_timedelta=lead_times)[variables]
              .drop_encoding()
        )

        ds = ds.chunk({'time': 1, 'latitude': 64, 'longitude': 32})

        with ProgressBar():
            ds.to_zarr(local_zarr, mode='w', consolidated=True)


def load_era5_climatology() -> None:
    """
    Load the ERA5 hourly climatology, reshape to a 12-hourly time series,
    and construct lead-time climatology forecasts by rolling along time.

    Note: the hourly climatology source is on a 240×121 grid (with poles).
    If a unified grid with 64×32 products is required downstream, explicit
    regridding needs to be applied after saving.
    """
    # Load hourly climatology and subset to 00/12 UTC for 12-hourly cadence
    era5_climatology = (
        xr.open_zarr(
            store='gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr',
            storage_options={"token": "anon"}
        )
        .sel(hour=[0, 12])[variables]
        .load()  # materialize to compute the stacking/rolling deterministically
    )
    print('loading complete')

    # Build 12-hourly index for 2020
    time_index = pd.date_range(start='2020-01-01 00:00', end='2020-12-31 12:00', freq='12h')

    # Collapse (dayofyear, hour) -> time to obtain a time-like series
    era5_climatology = (
        era5_climatology
            .stack(datetime=('dayofyear', 'hour'))
            .assign_coords(time=('datetime', time_index))
            .swap_dims({'datetime': 'time'})
            .drop_vars(['datetime', 'dayofyear', 'hour'])
    )

    # Base temporal resolution (expected 12 hours)
    delta = era5_climatology.time.diff("time").isel(time=0)

    # Roll forward for each lead time and label with prediction_timedelta
    rolled_list = []
    for td in lead_times:
        shift_steps = int(td / delta)
        shifted = era5_climatology.roll(time=-shift_steps, roll_coords=False)
        shifted = shifted.expand_dims(prediction_timedelta=[td])
        rolled_list.append(shifted)
    print('rolling complete')

    # Concatenate across lead times and persist
    era5_climatology_forecasts = xr.concat(rolled_list, dim="prediction_timedelta")
    output_path = os.path.join(save_path, 'era5_climatology_forecasts.zarr')
    era5_climatology_forecasts.to_zarr(output_path, mode='w', consolidated=True)
    print(f"Saved ERA5 climatology forecasts to {output_path}")


def load(
    load_classic_data: bool = False,
    build_era5_climatology: bool = False,
) -> None:
    """
    Orchestrate data preparation tasks.

    Parameters
    ----------
    load_classic_data : bool
        If True, loads ERA5/IFS analysis and model forecasts and stores them locally.
    build_era5_climatology : bool
        If True, builds and stores ERA5 climatology-based lead-time forecasts.
    """
    if load_classic_data:
        load_classic_zarr_data()

    if build_era5_climatology:
        load_era5_climatology()


if __name__ == "__main__":
   
    load(
        load_classic_data=False,
        build_era5_climatology=False,
    )
