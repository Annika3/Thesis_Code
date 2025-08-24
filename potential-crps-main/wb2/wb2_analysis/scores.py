import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from isodisreg import idr
import os
import pickle
from joblib import Parallel, delayed

from metric_functions import (
    pc, pcs,
    tw_crps, tw_crps_small, tw_pc, tw_pc_small, tw_pcs, tw_pcs_small,
    qw_crps, qw_crps_small, qw_pc, qw_pc_small, qw_pcs, qw_pcs_small,
    tw_pc_gaussian_cdf, tw_pcs_gaussian_cdf
)

# Constants  ----------------------------------------------------  

lead_times = [
    np.timedelta64(1, 'D'),
    np.timedelta64(3, 'D'),
    np.timedelta64(5, 'D'),
    np.timedelta64(7, 'D'),
    np.timedelta64(10, 'D')
    ]


obs_sources = {
    "era5": {
        "path": "data/era5_64x32.zarr", 
        "models": ["graphcast", "pangu", "hres"], 
    }, 
    "ifs": {
        "path": "data/ifs_analysis_64x32.zarr",
        "models": ["pangu_operational", "graphcast_operational", "hres"],
    }
}

variables = [
    "10m_wind_speed",
    "2m_temperature",
    "mean_sea_level_pressure"    
]

time_range = slice("2020-01-01", "2020-12-31")

n_jobs = 30

t_quantiles = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]
q_values = [0.01, 0.05, 0.1, 0.9, 0.95, 0.99]


# --------------------------------------------------------------------
# Pointwise metric computation helper
# --------------------------------------------------------------------

def _compute_metrics_point(model, lat, lon, lead_time, time_range, var,
                           t_quantiles, q_values, forecast_ds, observations):
    """
    Compute all metrics for one (model, lat, lon, lead_time, var) tuple.
    Returns a list of dict rows (long format with TW and QW blocks).
    """
    try:
        # Load forecast series and matching observations
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

        # Fit IDR once and predict probabilistic forecasts
        fitted_idr = idr(obs_arr, pd.DataFrame({"x": pred_arr}))
        prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred_arr}), digits=12)

        # Base CRPS and PCS
        pc_val = pc(prob_pred, obs_arr)
        pcs_val = pcs(pc_val, obs_arr)

        base = {
            "model": model,
            "lat": float(lat),
            "lon": float(lon),
            "variable": var,
            "lead_time": int(lead_time / np.timedelta64(1, "D")),
            "pc": pc_val,
            "pcs": pcs_val,
        }

        rows = []

        # Threshold-weighted metrics
        for t_quantile in t_quantiles:
            t = np.nanquantile(obs_arr, t_quantile)
            if t_quantile < 0.5:
                tw_pc_val = tw_pc_small(prob_pred, obs_arr, t)
                tw_pcs_val = tw_pcs_small(tw_pc_val, obs_arr, t)
                tw_tail = "lower"
            else:
                tw_pc_val = tw_pc(prob_pred, obs_arr, t)
                tw_pcs_val = tw_pcs(tw_pc_val, obs_arr, t)
                tw_tail = "upper"

            rows.append({
                **base,
                "t_quantile": t_quantile,
                "q_value": np.nan,
                "tw_pc": tw_pc_val,
                "tw_pcs": tw_pcs_val,
                "tw_tail": tw_tail,
                "qw_pc": np.nan,
                "qw_pcs": np.nan,
                "qw_tail": np.nan,
            })

        # Quantile-weighted metrics
        for q_value in q_values:
            if q_value < 0.5:
                qw_pc_val = qw_pc_small(prob_pred, obs_arr, q_value)
                qw_pcs_val = qw_pcs_small(qw_pc_val, obs_arr, q_value)
                qw_tail = "lower"
            else:
                qw_pc_val = qw_pc(prob_pred, obs_arr, q_value)
                qw_pcs_val = qw_pcs(qw_pc_val, obs_arr, q_value)
                qw_tail = "upper"

            rows.append({
                **base,
                "t_quantile": np.nan,
                "q_value": q_value,
                "tw_pc": np.nan,
                "tw_pcs": np.nan,
                "tw_tail": np.nan,
                "qw_pc": qw_pc_val,
                "qw_pcs": qw_pcs_val,
                "qw_tail": qw_tail,
            })

        return rows

    except Exception as e:
        # Return NaN rows for all (t,q) combinations to preserve rectangular schema
        print(f"Failed metric at ({lat},{lon}) for {model}/{var}: {e}")
        return [{
            "model": model,
            "lat": float(lat),
            "lon": float(lon),
            "variable": var,
            "lead_time": int(lead_time / np.timedelta64(1, "D")),
            "t_quantile": tq,
            "q_value": qv,
            "pc": np.nan,
            "pcs": np.nan,
            "tw_pc": np.nan,
            "tw_pcs": np.nan,
            "tw_tail": np.nan,
            "qw_pc": np.nan,
            "qw_pcs": np.nan,
            "qw_tail": np.nan,
        } for tq in t_quantiles for qv in q_values]

# --------------------------------------------------------------------
# Public orchestration function (classic + TW/QW). Expandable later.
# --------------------------------------------------------------------

def compute_classic_tw_qw_scores(
    *,
    lead_times,
    obs_sources,
    variables,
    time_range,
    t_quantiles,
    q_values,
    n_jobs=1,
    output_dir="score_data"
):
    """
    Run classic PC/PCS plus TW- and QW-weighted metrics over all requested
    observation sources, models, variables, lead times, and grid points.

    Parameters
    ----------
    lead_times : list[np.timedelta64]
        List of lead times to evaluate.
    obs_sources : dict
        Mapping of observation source name -> {"path": <zarr path>, "models": [model names]}.
    variables : list[str]
        Variable names to evaluate.
    time_range : slice or pandas-like indexer
        Time selection for evaluation window.
    t_quantiles : list[float]
        Quantiles in (0,1) used to derive thresholds for TW metrics.
    q_values : list[float]
        Quantile levels in (0,1) used for QW metrics.
    n_jobs : int, default 1
        Parallel jobs for joblib. Use "loky" backend.
    output_dir : str, default "score_data"
        Directory where CSV/PKL files are written.

    Returns
    -------
    dict
        Mapping {(obs_name, var, lead_days): pandas.DataFrame} of results.
    """
    # Resolve output directory (relative to this file for reproducibility)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, output_dir)
    os.makedirs(save_dir, exist_ok=True)

    results_map = {}

    for obs_name, obs_info in obs_sources.items():
        print(f"\nProcessing {obs_name.upper()} observations…")
        obs_path = obs_info["path"]
        model_names = obs_info["models"]

        # Load observations
        observations = xr.open_zarr(obs_path, decode_timedelta=True)

        # Load forecasts (skip missing and warn)
        forecast_ds = {}
        for mdl in model_names:
            f_path = f"data/{mdl}_64x32.zarr"
            if not os.path.exists(f_path):
                print(f"⚠️  Missing forecast file {f_path} – skip {mdl}")
                continue
            forecast_ds[mdl] = xr.open_zarr(f_path, decode_timedelta=True)

        # If all models are missing, continue
        if not forecast_ds:
            print("No forecast datasets available for this observation source.")
            continue

        # Grid coordinates
        lats = observations.latitude.values
        lons = observations.longitude.values

        for var in variables:
            print(f"  → variable: {var}")
            for lead_time in lead_times:
                print(f"    → lead_time: {int(lead_time / np.timedelta64(1, 'D'))} days")

                # Build task list
                tasks = []
                for lat in lats:
                    for lon in lons:
                        for mdl in model_names:
                            if mdl not in forecast_ds:
                                continue
                            tasks.append((
                                mdl, lat, lon, lead_time, time_range, var,
                                t_quantiles, q_values, forecast_ds, observations
                            ))

                # Parallel computation
                rows_nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                    delayed(_compute_metrics_point)(*t) for t in tasks
                )

                # Flatten and frame
                flat_rows = [row for rows in rows_nested for row in rows]
                df = pd.DataFrame(flat_rows)

                # Save per variable + lead time + obs source
                lead_days = int(lead_time / np.timedelta64(1, "D"))
                csv_name = f"{var}_lead{lead_days}d_{obs_name}.csv"
                pkl_name = csv_name.replace(".csv", ".pkl")
                csv_path = os.path.join(save_dir, csv_name)
                pkl_path = os.path.join(save_dir, pkl_name)

                df.to_csv(csv_path, index=False)
                with open(pkl_path, "wb") as f:
                    pickle.dump(df, f)

                print(f"    • saved → {csv_path}")

                results_map[(obs_name, var, lead_days)] = df

    return results_map


import numpy as np
import pandas as pd
import xarray as xr
import os
import pickle
from joblib import Parallel, delayed
from isodisreg import idr

# Import scoring helpers from your local module
from metric_functions import (
    tw_pc, tw_pcs, tw_pc_small, tw_pcs_small,          # baseline threshold-weighted scores
    tw_pc_gaussian_cdf, tw_pcs_gaussian_cdf            # Gaussian-CDF TW scores
)

# --------------------------------------------------------------------
# Gaussian-CDF TW-only runner with baseline comparison
# --------------------------------------------------------------------

import numpy as np
import pandas as pd
import xarray as xr
import os
import pickle
from joblib import Parallel, delayed
from isodisreg import idr

# Scoring helpers (Gaussian-CDF TW only)
from metric_functions import (
    tw_pc_gaussian_cdf, tw_pcs_gaussian_cdf
)

# --------------------------------------------------------------------
# Gaussian-CDF TW-only runner that MERGES baseline TW from score_data
# --------------------------------------------------------------------

def compute_gaussian_tw_scores(
    *,
    lead_times,
    obs_sources,
    variables,
    time_range,
    t_quantiles,
    n_jobs=20
    ,
    baseline_dir="score_data",          # where your existing CSVs live
    output_dir="score_data",   # separate output directory
    gauss_sigma=0.1,                    # see sigma_is_fraction below
    sigma_is_fraction=True,
    fallback_sigma=1.0
):
    """
    Compute Gaussian-CDF threshold-weighted PC/PCS using mu=t (per t_quantile),
    merge baseline TW metrics from existing CSVs (no baseline recomputation),
    and save results to separate files.

    Baseline file naming assumed to be: f"{var}_lead{lead_days}d_{obs_name}.csv" in baseline_dir.
    Keys used for merge: model, lat, lon, variable, lead_time, t_quantile.

    Parameters
    ----------
    lead_times : list[np.timedelta64]
    obs_sources : dict  -> {obs_name: {"path": <zarr>, "models": [..]}}
    variables : list[str]
    time_range : slice or pandas-like indexer
    t_quantiles : list[float] in (0,1)
    n_jobs : int
    baseline_dir : str
    output_dir : str
    gauss_sigma : float
        If sigma_is_fraction=True, interpreted as fraction of local std(y).
        Else, used as absolute sigma.
    sigma_is_fraction : bool
    fallback_sigma : float
        Absolute sigma used if local std is zero or not finite.

    Returns
    -------
    dict[(obs_name, var, lead_days)] -> pandas.DataFrame
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Cache for baseline DataFrames by (obs_name, var, lead_days)
    baseline_cache = {}

    def _load_baseline(obs_name, var, lead_days):
        """
        Load and index the baseline TW metrics for one (obs_name, var, lead_days).
        Returns a dict keyed by (model, lat, lon, t_quantile) -> dict with baseline fields.
        """
        key = (obs_name, var, lead_days)
        if key in baseline_cache:
            return baseline_cache[key]

        fname = f"{var}_lead{lead_days}d_{obs_name}.csv"
        fpath = os.path.join(script_dir, baseline_dir, fname)
        if not os.path.exists(fpath):
            print(f"⚠️  Baseline file not found: {fpath}")
            baseline_cache[key] = {}
            return baseline_cache[key]

        dfb = pd.read_csv(fpath)
        # Keep only TW rows (t_quantile present) and relevant columns
        dfb = dfb.copy()
        if "t_quantile" in dfb.columns:
            dfb = dfb[dfb["t_quantile"].notna()]
        else:
            dfb = dfb.iloc[0:0]

        # Ensure consistent types for merge keys
        for col in ("lat", "lon", "t_quantile"):
            if col in dfb.columns:
                dfb[col] = dfb[col].astype(float)
        if "lead_time" in dfb.columns:
            dfb["lead_time"] = dfb["lead_time"].astype(int)

        cols_keep = [
            "model", "lat", "lon", "variable", "lead_time", "t_quantile",
            "tw_pc", "tw_pcs", "tw_tail"
        ]
        for c in cols_keep:
            if c not in dfb.columns:
                dfb[c] = np.nan

        # Build a fast lookup dict
        lookup = {}
        for _, r in dfb[cols_keep].iterrows():
            k = (str(r["model"]), float(r["lat"]), float(r["lon"]), float(r["t_quantile"]))
            lookup[k] = {
                "baseline_tw_pc": float(r["tw_pc"]) if pd.notna(r["tw_pc"]) else np.nan,
                "baseline_tw_pcs": float(r["tw_pcs"]) if pd.notna(r["tw_pcs"]) else np.nan,
                "baseline_tail": r["tw_tail"] if isinstance(r["tw_tail"], str) else np.nan,
            }

        baseline_cache[key] = lookup
        return lookup

    def _point_job(model, lat, lon, lead_time, var, observations, forecast_ds, baseline_lookup):
        """
        Compute Gaussian-CDF TW for one gridpoint and merge baseline from lookup.
        """
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

            # IDR fit for Gaussian TW (required)
            fitted_idr = idr(obs_arr, pd.DataFrame({"x": pred_arr}))
            prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred_arr}), digits=12)

            # Sigma handling
            local_std = float(np.nanstd(obs_arr))
            if not np.isfinite(local_std) or local_std <= 0:
                local_std = np.nan

            base = {
                "model": model,
                "lat": float(lat),
                "lon": float(lon),
                "variable": var,
                "lead_time": int(lead_time / np.timedelta64(1, "D")),
            }

            rows = []
            for tq in t_quantiles:
                t = float(np.nanquantile(obs_arr, tq))

                # Choose sigma
                if sigma_is_fraction:
                    if np.isfinite(local_std):
                        sigma_use = float(max(gauss_sigma * local_std, np.finfo(float).eps))
                    else:
                        sigma_use = float(fallback_sigma)
                else:
                    sigma_use = float(max(gauss_sigma, np.finfo(float).eps))

                # Gaussian-CDF TW using mu = t
                gauss_twpc = tw_pc_gaussian_cdf(prob_pred, obs_arr, mu=t, sigma=sigma_use)
                gauss_twpcs = tw_pcs_gaussian_cdf(gauss_twpc, obs_arr, mu=t, sigma=sigma_use)

                # Merge baseline from preloaded lookup
                bkey = (str(model), float(lat), float(lon), float(tq))
                b = baseline_lookup.get(bkey, None)
                if b is not None:
                    baseline_tw_pc = b["baseline_tw_pc"]
                    baseline_tw_pcs = b["baseline_tw_pcs"]
                    baseline_tail = b["baseline_tail"]
                else:
                    baseline_tw_pc = np.nan
                    baseline_tw_pcs = np.nan
                    baseline_tail = np.nan

                # Differences
                delta_pc = gauss_twpc - baseline_tw_pc if pd.notna(baseline_tw_pc) else np.nan
                delta_pcs = gauss_twpcs - baseline_tw_pcs if pd.notna(baseline_tw_pcs) else np.nan
                rel_pc = (delta_pc / baseline_tw_pc) if (pd.notna(delta_pc) and baseline_tw_pc not in (0, np.nan)) else np.nan
                rel_pcs = (delta_pcs / baseline_tw_pcs) if (pd.notna(delta_pcs) and baseline_tw_pcs not in (0, np.nan)) else np.nan

                rows.append({
                    **base,
                    "t_quantile": float(tq),
                    "t_value": t,
                    "gauss_mu": t,
                    "gauss_sigma": sigma_use,
                    "gauss_sigma_is_fraction": bool(sigma_is_fraction),
                    "gauss_tw_pc": gauss_twpc,
                    "gauss_tw_pcs": gauss_twpcs,
                    # merged baseline (read from CSV)
                    "baseline_tail": baseline_tail,
                    "baseline_tw_pc": baseline_tw_pc,
                    "baseline_tw_pcs": baseline_tw_pcs,
                    # differences
                    "delta_tw_pc": delta_pc,
                    "delta_tw_pcs": delta_pcs,
                    "rel_delta_tw_pc": rel_pc,
                    "rel_delta_tw_pcs": rel_pcs,
                })

            return rows

        except Exception as e:
            print(f"Gaussian-TW (merge) failed at ({lat},{lon}) for {model}/{var}: {e}")
            nan_rows = []
            for tq in t_quantiles:
                nan_rows.append({
                    "model": model,
                    "lat": float(lat),
                    "lon": float(lon),
                    "variable": var,
                    "lead_time": int(lead_time / np.timedelta64(1, "D")),
                    "t_quantile": float(tq),
                    "t_value": np.nan,
                    "gauss_mu": np.nan,
                    "gauss_sigma": np.nan,
                    "gauss_sigma_is_fraction": bool(sigma_is_fraction),
                    "gauss_tw_pc": np.nan,
                    "gauss_tw_pcs": np.nan,
                    "baseline_tail": np.nan,
                    "baseline_tw_pc": np.nan,
                    "baseline_tw_pcs": np.nan,
                    "delta_tw_pc": np.nan,
                    "delta_tw_pcs": np.nan,
                    "rel_delta_tw_pc": np.nan,
                    "rel_delta_tw_pcs": np.nan,
                })
            return nan_rows

    results_map = {}

    for obs_name, obs_info in obs_sources.items():
        print(f"\n[Gaussian TW] Processing {obs_name.upper()} observations…")
        obs_path = obs_info["path"]
        model_names = obs_info["models"]

        observations = xr.open_zarr(obs_path, decode_timedelta=True)

        # Load available model datasets
        forecast_ds = {}
        for mdl in model_names:
            f_path = f"data/{mdl}_64x32.zarr"
            if not os.path.exists(f_path):
                print(f"⚠️  Missing forecast file {f_path} – skip {mdl}")
                continue
            forecast_ds[mdl] = xr.open_zarr(f_path, decode_timedelta=True)

        if not forecast_ds:
            print("No forecast datasets available for this observation source.")
            continue

        lats = observations.latitude.values
        lons = observations.longitude.values

        for var in variables:
            for lead_time in lead_times:
                lead_days = int(lead_time / np.timedelta64(1, "D"))
                print(f"  → variable: {var} | lead: {lead_days}d")

                # Preload baseline lookup for this (obs_name, var, lead_days)
                baseline_lookup = _load_baseline(obs_name, var, lead_days)

                tasks = []
                for lat in lats:
                    for lon in lons:
                        for mdl in model_names:
                            if mdl not in forecast_ds:
                                continue
                            tasks.append((mdl, lat, lon, lead_time, var, observations, forecast_ds, baseline_lookup))

                rows_nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                    delayed(_point_job)(*t) for t in tasks
                )

                flat_rows = [r for rows in rows_nested for r in rows]
                df = pd.DataFrame(flat_rows)

                csv_name = f"{var}_lead{lead_days}d_{obs_name}_gaussianTW.csv"
                pkl_name = csv_name.replace(".csv", ".pkl")
                csv_path = os.path.join(out_dir, csv_name)
                pkl_path = os.path.join(out_dir, pkl_name)

                df.to_csv(csv_path, index=False)
                with open(pkl_path, "wb") as f:
                    pickle.dump(df, f)

                print(f"    • saved (Gaussian TW merged) → {csv_path}")
                results_map[(obs_name, var, lead_days)] = df

    return results_map


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------



if __name__ == "__main__":

    """
    compute_classic_tw_qw_scores(
        lead_times=lead_times,
        obs_sources=obs_sources,
        variables=variables,
        time_range=time_range,
        t_quantiles=t_quantiles,
        q_values=q_values,
        n_jobs=n_jobs,
        output_dir="score_data",
    )
    """
    
    compute_gaussian_tw_scores(
        lead_times=lead_times,
        obs_sources=obs_sources,
        variables=variables,
        time_range=time_range,
        t_quantiles=[0.9, 0.95, 0.99],
        n_jobs=20,
        output_dir="score_data",
        gauss_sigma=0.1,           # 10% of local std(y)
        sigma_is_fraction=True,     # interpret gauss_sigma as a fraction
        fallback_sigma=1.0          # used if local std is zero / not finite
    )