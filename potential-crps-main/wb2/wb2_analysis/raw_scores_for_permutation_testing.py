import os
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from isodisreg import idr

print(">>> Running:", __file__)
print(">>> Working dir:", os.getcwd())


## METRIC FUNCTIONS ----------------------------------------------------

def tw_crps(self, obs, t):
    
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    def get_points(pred):
        return np.array(pred.points)
    def get_cdf(pred):
        return np.array(pred.ecdf)
    def modify_points(cdf):
        return np.hstack([cdf[0], np.diff(cdf)])

    def tw_crps0(y, p, w, x, t):
        x = np.maximum(x, t)
        y = np.maximum(y, t)
        return 2 * np.sum(w * ((y < x).astype(float) - p + 0.5 * w) * (x - y))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(modify_points, p))

    T = [t] * len(y)   
    return list(map(tw_crps0, y, p, w, x, T))



def qw_crps0(y, w, x, q):
        c_cum = np.cumsum(w)
        c_cum_prev = np.hstack(([0], c_cum[:-1]))
        c_cum_star = np.maximum(c_cum, q)
        c_cum_prev_star = np.maximum(c_cum_prev, q)
        indicator = (x >= y).astype(float)
        terms = indicator * (c_cum_star - c_cum_prev_star) - 0.5 * (c_cum_star**2 - c_cum_prev_star**2)
        return 2 * np.sum(terms * (x - y))

def qw_crps(self, obs, q=0.9):
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    def get_points(pred):
        return np.array(pred.points)
    def get_cdf(pred):
        return np.array(pred.ecdf)
    def get_weights(cdf):
        return np.hstack([cdf[0], np.diff(cdf)])

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(get_weights, p))
    Q = [q] * len(y)
    return list(map(qw_crps0, y, w, x, Q))


# Configuration
data_dir = "/mnt/c/Users/monam/potential-crps-main/data" #Folder with Zarr Files/ Folders
score_data_dir_helper = os.path.dirname(__file__)
SCORE_DATA_DIR = os.path.join(score_data_dir_helper, 'score_data')
os.makedirs(SCORE_DATA_DIR, exist_ok=True)

# Threshold and quantile levels
T_QUANTILES = [0.9, 0.95, 0.99]
Q_QUANTILES   = [0.9, 0.95, 0.99]

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
    Compute time-resolved scores for a single grid point and lead time:
      - PC (pointwise CRPS values)
      - tw_pc_<q> for each threshold quantile in T_QUANTILES
      - qw_pc_<q> for each quantile in Q_QUANTILES
    Returns: list of dicts (one record per time step)
    """
    # Select forecasts for the given model, variable, grid point, and lead time
    preds = (
        forecast_ds[model]
        .sel(prediction_timedelta=lead_time, latitude=lat, longitude=lon, method='nearest')
        .sel(time=TIME_RANGE)[var]
    )

    # Select observations shifted by the lead time
    obs = (
        obs_ds
        .sel(latitude=lat, longitude=lon)
        .sel(time=preds.time + lead_time)[var]
    )

    # Extract numpy arrays for faster operations
    times = preds.time.values
    pred_arr = np.asarray(preds.values)
    obs_arr  = np.asarray(obs.values)

    # Return empty list if no data
    if pred_arr.size == 0 or obs_arr.size == 0:
        return []

    # Mask out any missing values
    finite = np.isfinite(pred_arr) & np.isfinite(obs_arr)
    if not np.all(finite):
        pred_arr = pred_arr[finite]
        obs_arr  = obs_arr[finite]
        times    = times[finite]
        if pred_arr.size == 0:
            return []

    # Fit IDR model and generate probabilistic predictions
    fitted = idr(obs_arr, pd.DataFrame({'x': pred_arr}))
    prob_pred = fitted.predict(pd.DataFrame({'x': pred_arr}))

    # --- PC per time step (here: CRPS values, not aggregated) ---
    crps_series = np.asarray(prob_pred.crps(obs_arr))  # shape (T,)

    # --- Threshold-weighted PC per time step ---
    tw_series = {}
    for q in T_QUANTILES:  # e.g., [0.9, 0.95]
        t_q = float(np.nanquantile(obs_arr, q))
        type(prob_pred).tw_crps = tw_crps  # monkey-patch method into object
        tw_series[q] = np.asarray(prob_pred.tw_crps(obs_arr, t_q))

    # --- Quantile-weighted PC per time step ---
    qw_series = {}
    for q in Q_QUANTILES:  # can be same as T_QUANTILES
        type(prob_pred).qw_crps = qw_crps  # monkey-patch method into object
        qw_series[q] = np.asarray(prob_pred.qw_crps(obs_arr, q=q))

    # --- Build list of records ---
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
            'pc': float(crps_series[i]),
        }
        # Add threshold-weighted PC values
        for q, arr in tw_series.items():
            rec[f'tw_pc_{q}'] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
        # Add quantile-weighted PC values
        for q, arr in qw_series.items():
            rec[f'qw_pc_{q}'] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
        recs.append(rec)

    return recs



def main(n_jobs: int = 6):
    all_records = []

    for obs_name, info in OBS_SOURCES.items():
        print(f"Loading observations: {obs_name}")
        obs_ds = xr.open_zarr(info['path'], decode_timedelta=True)
        forecast_ds = {}
        for mdl in info['models']:
            path = os.path.join(data_dir, f"{mdl}_64x32.zarr")
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
