import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from isodisreg import idr
import os
import pickle
from joblib import Parallel, delayed

# get data  ----------------------------------------------------  

# model_names = ['graphcast', 'pangu', 'hres'] #for ERA5 comparison
# model_names = [ 'pangu_operational', 'graphcast_operational', 'hres'] #for IFS comparison
"""
forecast_ds = {}
for model in model_names:
    forecast_path = f'data/{model}_64x32.zarr'
    if not os.path.exists(forecast_path):
        print(f"Forecast file {forecast_path} not found, skipping!")
        continue
    forecast_ds[model] = xr.open_zarr(forecast_path, decode_timedelta=True)


forecasts = xr.open_zarr(
        store='data/graphcast_64x32.zarr',
        decode_timedelta=True
        )


observations = xr.open_zarr(
        # store='data/era5_64x32.zarr', # ERA5 
        store ='data/ifs_analysis_64x32.zarr', #IFS 
        decode_timedelta=True
        )
"""
'''
# check for shape/ names of variables/ dataset
ds = xr.open_zarr('data/era5_64x32.zarr', consolidated=True)
print(ds)
print(ds['2m_temperature'])
print(ds['2m_temperature'].shape)
print(ds['2m_temperature'].isel(time=0)) # First time slice

'''

# definiton of error measures (to be linked to simulation study?) ----------------------------------------------------

# Classic PC(S)
def pc(pred, y):
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    # "predict" the estimated cdfs again, so that we can use the crps function
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))
    mean_crps = np.mean(prob_pred.crps(y))
    return mean_crps

def pcs(pred, y):
    # crps of the climatological forecast
    pc_ref = np.mean(np.abs(np.tile(y, (len(y), 1)) - np.tile(y, (len(y), 1)).transpose())) / 2

    return (pc_ref - pc(pred, y)) / pc_ref

# threshold weighted
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

def tw_pc(pred, y, t): 
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))

    type(prob_pred).tw_crps = tw_crps # monkey-patch the tw_crps method into the prediction object

    tw_crps_scores = prob_pred.tw_crps(y, t)
    mean_tw_crps = np.mean(tw_crps_scores)
    return mean_tw_crps

def tw_pcs(pred, y, t):

    y_thresh = np.maximum(y, t)
    pc_ref = np.mean(np.abs(np.tile(y_thresh, (len(y_thresh), 1)) - np.tile(y_thresh, (len(y_thresh), 1)).transpose())) / 2

    pc_model = tw_pc(pred, y, t)

    pcs = (pc_ref - pc_model) / pc_ref 
   
    return pcs

# quantile weighted
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

    def qw_crps0(y, p, w, x, q):
        c_cum = np.cumsum(w)
        c_cum_prev = np.hstack(([0], c_cum[:-1]))
        c_cum_star = np.maximum(c_cum, q)
        c_cum_prev_star = np.maximum(c_cum_prev, q)
        indicator = (x >= y).astype(float)
        terms = indicator * (c_cum_star - c_cum_prev_star) - 0.5 * (c_cum_star**2 - c_cum_prev_star**2)
        return 2 * np.sum(terms * (x - y))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(get_weights, p))
    Q = [q] * len(y)
    return list(map(qw_crps0, y, p, w, x, Q))

def qw_pc(pred, y, q=0.9):
    
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))
    type(prob_pred).qw_crps = qw_crps
    qwcrps_scores = prob_pred.qw_crps(y, q=q)
    return np.mean(qwcrps_scores)


def qw_pcs(pred, y, q):

    pc_model = qw_pc(pred, y, q)
    
    def climatological_qw_pc(y, q):
        """
        Compute PCRPS using a climatological forecast: fit IDR on y, predict same pooled ECDF for all y.
        (This avoids overfitting)
        """
        x_dummy = np.zeros_like(y)  # All predictors the same -> 1 pooled distribution
        fitted_idr = idr(y, pd.DataFrame({'x': x_dummy}))
        prob_pred = fitted_idr.predict(pd.DataFrame({'x': x_dummy}))

        type(prob_pred).qw_crps = qw_crps # monkey-patch 

        qw_crps_scores = prob_pred.qw_crps(y, q)
        mean_qw_crps = np.mean(qw_crps_scores)
        return mean_qw_crps
    pc_ref = climatological_qw_pc(y, q)
    pcs = (pc_ref - pc_model) / pc_ref   
    return pcs




# (re-) define variables to loop over (to be linked to load_data file)   ----------------------------------------------------  

# lats = observations.latitude.values
# lons = observations.longitude.values


lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
    ]

variables = [
    '2m_temperature',
    'mean_sea_level_pressure',
    '10m_wind_speed',
    ]
total_vars = len(variables)
var_counter = 0
# lead_time = np.timedelta64(3, 'D')
time_range = slice('2020-01-01','2020-12-31')

# var = 'mean_sea_level_pressure' # fixed variable for now, can be looped over later

q = 0.9

# ----------------------------------------------------
#  Metric helper
# ----------------------------------------------------

def compute_metrics(model, lat, lon, lead_time, time_range, var, q, forecast_ds, observations):
    try:
        preds = forecast_ds[model].sel(
            prediction_timedelta=lead_time,
            latitude=lat,
            longitude=lon,
            method="nearest",
        ).sel(time=time_range)[var].load()

        obs = observations.sel(
            latitude=lat,
            longitude=lon,
            time=preds.time + lead_time,
        )[var].load()

        pred_arr = preds.values
        obs_arr = obs.values
        if pred_arr.size == 0 or obs_arr.size == 0:
            raise ValueError("Empty series")

        t = np.nanquantile(obs_arr, 0.9)

        return {
            "model": model,
            "lat": float(lat),
            "lon": float(lon),
            "variable": var,
            "lead_time": int(lead_time / np.timedelta64(1, "D")),
            "pc": pc(pred_arr, obs_arr),
            "pcs": pcs(pred_arr, obs_arr),
            "tw_pc": tw_pc(pred_arr, obs_arr, t),
            "tw_pcs": tw_pcs(pred_arr, obs_arr, t),
            "qw_pc": qw_pc(pred_arr, obs_arr, q),
            "qw_pcs": qw_pcs(pred_arr, obs_arr, q),
        }
    except Exception as e:
        print(f"Failed metric at ({lat},{lon}) for {model}/{var}: {e}")
        return {k: np.nan for k in [
            "model", "lat", "lon", "variable", "lead_time", "pc", "pcs",
            "tw_pc", "tw_pcs", "qw_pc", "qw_pcs"]}

# ----------------------------------------------------
#  CONFIGURATION  (structure wie zuvor – aber nur IFS)
# ----------------------------------------------------

obs_sources = {
    "ifs": {
        "path": "data/ifs_analysis_64x32.zarr",
        "models": ["pangu_operational", "graphcast_operational", "hres"],
    }
}

variables = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_wind_speed",
]

lead_time = np.timedelta64(7, "D")
time_range = slice("2020-01-01", "2020-12-31")
q = 0.9
n_jobs = 5

# ----------------------------------------------------
#  MAIN WORK – gleiche Struktur wie zuvor (ohne __main__‑Guard)
# ----------------------------------------------------

for obs_name, obs_info in obs_sources.items():
    print(f"\nProcessing {obs_name.upper()} observations…")
    obs_path = obs_info["path"]
    model_names = obs_info["models"]

    # Load observations
    observations = xr.open_zarr(obs_path, decode_timedelta=True)

    # Load forecasts
    forecast_ds = {}
    for mdl in model_names:
        f_path = f"data/{mdl}_64x32.zarr"
        if not os.path.exists(f_path):
            print(f"⚠️  Missing forecast file {f_path} – skip {mdl}")
            continue
        forecast_ds[mdl] = xr.open_zarr(f_path, decode_timedelta=True)

    # Grid coordinates
    lats = observations.latitude.values
    lons = observations.longitude.values

    # Directory for results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "score_data")
    os.makedirs(save_dir, exist_ok=True)

    # ---- iterate over variables -------------------------------------------
    for var in variables:
        print(f"  → variable: {var}")
        tasks = []
        for lat in lats:
            for lon in lons:
                for mdl in model_names:
                    if mdl not in forecast_ds:
                        continue
                    tasks.append((mdl, lat, lon, lead_time, time_range, var, q, forecast_ds, observations))

        # Parallel computation
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(compute_metrics)(*t) for t in tasks
        )

        # Save
        df = pd.DataFrame(results)
        lead_days = int(lead_time / np.timedelta64(1, "D"))
        csv_name = f"{var}_lead{lead_days}d_{obs_name}.csv"
        pkl_name = csv_name.replace(".csv", ".pkl")
        csv_path = os.path.join(save_dir, csv_name)
        pkl_path = os.path.join(save_dir, pkl_name)
        df.to_csv(csv_path, index=False)
        with open(pkl_path, "wb") as f:
            pickle.dump(df, f)
        print(f"    • saved → {csv_path}")


"""
results = Parallel(n_jobs=n_jobs, backend='loky', verbose=10)(
    delayed(compute_metrics)(*task) for task in tasks
)
"""
## Saving/ Showing results ----------------------------------------------------
"""""
df = pd.DataFrame(results)
print(df)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, 'score_data')

obs_name = 'ifs'

csv_name = f'{var}_lead{int(lead_time / np.timedelta64(1, "D"))}d_{obs_name}.csv'
pkl_name = f'{var}_lead{int(lead_time / np.timedelta64(1, "D"))}d_{obs_name}.pkl'

csv_path = os.path.join(save_dir, csv_name)
pkl_path = os.path.join(save_dir, pkl_name)

df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

with open(pkl_path, 'wb') as f:
    pickle.dump(df, f)
print(f"Pickle saved to {pkl_path}")

"""

'''
preds_point = forecasts.sel(prediction_timedelta=lead_time, latitude=ka_lat, longitude=ka_lon, method='nearest').sel(time=time_range)[var].load()
valid_time = preds_point.time + lead_time
obs_point = observations.sel(latitude=preds_point.latitude, longitude=preds_point.longitude, time=valid_time)[var].load()

fitted_idr = idr(obs_point, preds_point.to_dataframe()[[var]])
easyuq_preds_point = fitted_idr.predict(preds_point.to_dataframe()[[var]], digits=8)
crps = easyuq_preds_point.crps(obs_point)
pc = np.mean(crps)
print(pc)

'''