import os
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from isodisreg import idr

from scores import tw_crps as original_tw_crps, qw_crps as original_qw_crps

# Configuration
data_dir = os.path.dirname(__file__)
SCORE_DATA_DIR = os.path.join(data_dir, 'score_data')
os.makedirs(SCORE_DATA_DIR, exist_ok=True)

# Threshold and quantile levels
T_QUANTILES = [0.9, 0.95, 0.99]
Q_VALUES   = [0.9, 0.95, 0.99]

# Observation sources and corresponding models
OBS_SOURCES = {
    'ifs': {
        'path': 'data/ifs_analysis_64x32.zarr',
        'models': ['pangu_operational', 'graphcast_operational', 'hres'],
    },
    'era5': {
        'path': 'data/era5_64x32.zarr',
        'models': ['graphcast', 'pangu', 'hres'],
    }
}

# Variables and lead times
VARIABLES = [
    '10m_wind_speed',
    '2m_temperature',
    'mean_sea_level_pressure',
]
LEAD_TIMES = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D'),
]
# Evaluation period
TIME_RANGE = slice('2020-01-01', '2020-12-31')


def compute_metrics_raw(model, lat, lon, lead_time, var, obs_ds, forecast_ds):
    """
    Compute raw per-time scores for a single gridpoint.
    Returns list of dicts with:
      model, variable, latitude, longitude,
      prediction_timedelta, time, crps,
      tw_crps_<t>, qw_crps_<q> for each threshold/quantile.
    """
    # Select forecast and verify times
    preds = (forecast_ds[model]
             .sel(prediction_timedelta=lead_time, latitude=lat, longitude=lon, method='nearest')
             .sel(time=TIME_RANGE)[var]
            )
    obs = (obs_ds
           .sel(latitude=lat, longitude=lon)
           .sel(time=preds.time + lead_time)[var]
          )

    times = preds.time.values
    pred_arr = preds.values
    obs_arr = obs.values
    if pred_arr.size == 0 or obs_arr.size == 0:
        return []

    # Fit IDR and get probabilistic forecast
    fitted = idr(obs_arr, pd.DataFrame({'x': pred_arr}))
    prob_pred = fitted.predict(pd.DataFrame({'x': pred_arr}))

    # Raw CRPS per time
    crps_scores = prob_pred.crps(obs_arr)

    # Compute thresholds from observations
    thresholds = {t: float(np.nanquantile(obs_arr, t)) for t in T_QUANTILES}

    # Compute weighted CRPS using original functions
    tw_scores = {t: original_tw_crps(prob_pred, obs_arr, thresholds[t]) for t in T_QUANTILES}
    qw_scores = {q: original_qw_crps(prob_pred, obs_arr, q)           for q in Q_VALUES}

    # Build records
    recs = []
    lead_days = int(lead_time / np.timedelta64(1, 'D'))
    for i, ts in enumerate(times):
        rec = {
            'model': model,
            'variable': var,
            'latitude': float(lat),
            'longitude': float(lon),
            'prediction_timedelta': lead_days,
            'time': pd.to_datetime(ts),
            'crps': float(crps_scores[i]),
        }
        # attach threshold-weighted
        for t, arr in tw_scores.items():
            rec[f'tw_crps_{t}'] = float(arr[i])
        # attach quantile-weighted
        for q, arr in qw_scores.items():
            rec[f'qw_crps_{q}'] = float(arr[i])
        recs.append(rec)
    return recs


def main(n_jobs: int = 8):
    all_records = []

    for obs_name, info in OBS_SOURCES.items():
        print(f"Loading observations: {obs_name}")
        obs_ds = xr.open_zarr(info['path'], decode_timedelta=True)
        forecast_ds = {}
        for mdl in info['models']:
            path = os.path.join(data_dir, 'data', f"{mdl}_64x32.zarr")
            print(f"  Loading forecast: {mdl}")
            forecast_ds[mdl] = xr.open_zarr(path, decode_timedelta=True)

        for var in VARIABLES:
            for lead in LEAD_TIMES:
                lead_d = int(lead / np.timedelta64(1, 'D'))
                print(f"Computing raw scores: {obs_name}, {var}, lead={lead_d}d")
                lats = obs_ds.latitude.values
                lons = obs_ds.longitude.values
                tasks = [ (mdl, lat, lon, lead, var, obs_ds, forecast_ds)
                          for lat in lats for lon in lons for mdl in info['models'] ]
                results = Parallel(n_jobs=n_jobs, verbose=5)(
                    delayed(compute_metrics_raw)(*t) for t in tasks
                )
                for sub in results:
                    all_records.extend(sub)

    # Assemble and save
    df = pd.DataFrame(all_records)
    idx = ['time', 'latitude', 'longitude', 'prediction_timedelta', 'model', 'variable']
    ds = df.set_index(idx).to_xarray()
    out_zarr = os.path.join(SCORE_DATA_DIR, 'raw_scores_for_permutation.zarr')
    ds.to_zarr(out_zarr, mode='w')
    print(f"Saved raw scores to {out_zarr}")


if __name__ == '__main__':
    main(n_jobs=6)
