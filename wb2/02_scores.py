#scores.py

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import glob
from dask.diagnostics import ProgressBar
from isodisreg import idr
import os
import re
import pickle
from joblib import Parallel, delayed
from weatherbench2.metrics import _spatial_average

from metric_functions import (
    pc, pcs,
    tw_crps, tw_crps_small, tw_pc, tw_pc_small, tw_pcs, tw_pcs_small,
    qw_crps, qw_crps_small, qw_pc, qw_pc_small, qw_pcs, qw_pcs_small,
    tw_pc_gaussian_cdf, tw_pcs_gaussian_cdf
)

# --------------------------------------------------------------------
# Constants and defaults
# --------------------------------------------------------------------

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

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
score_data_dir = os.path.join(script_dir, "score_data")
os.makedirs(score_data_dir, exist_ok=True)



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
        # Return NaN rows for all (t,q) combinations 
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
    
    
def _sigma_tag(x: float) -> str:
    
    s = f"{x:.6g}"
    return s.replace(".", "p")


# --------------------------------------------------------------------
# Score computation functions (without init time column)
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
    # Resolve output directory (relative to this file)
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
                print(f"Missing forecast file {f_path} – skip {mdl}")
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


def compute_gaussian_tw_scores(
    *,
    lead_times,
    obs_sources,
    variables,
    time_range,
    t_quantiles,
    n_jobs=20
    ,
    baseline_dir="score_data",          # where existing CSVs live
    output_dir="score_data",            # separate output directory
    gauss_sigma=0.1,                    
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
            print(f" Baseline file not found: {fpath}")
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

            # IDR fit for Gaussian TW
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
                print(f"Missing forecast file {f_path} – skip {mdl}")
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

                mode = "frac" if sigma_is_fraction else "abs"
                sig_tag = _sigma_tag(gauss_sigma)
                fb_tag  = _sigma_tag(fallback_sigma)

                csv_name = f"{var}_lead{lead_days}d_{obs_name}_gaussianTW_{mode}_sig{sig_tag}_fb{fb_tag}.csv"
                pkl_name = csv_name.replace(".csv", ".pkl")
                csv_path = os.path.join(out_dir, csv_name)
                pkl_path = os.path.join(out_dir, pkl_name)

                df.to_csv(csv_path, index=False)
                with open(pkl_path, "wb") as f:
                    pickle.dump(df, f)

                print(f"    • saved (Gaussian TW merged) → {csv_path}")
                results_map[(obs_name, var, lead_days)] = df

    return results_map


def _wb2_weighted_mean_from_rows(
    df_region: pd.DataFrame,
    metric: str,
    *,
    exclude_zeros: bool = False,
    zero_tol: float = 0.0
) -> float:
    """
    WB2-consistent spatial mean for one region/group:
    1) collapse duplicates at identical (lat, lon) via mean
    2) make a 2D lat×lon grid
    3) call WeatherBench2's _spatial_average on that grid (area/cos-lat weighting).

    Parameters
    ----------
    exclude_zeros : bool
        If True, rows with |metric| <= zero_tol are discarded prior to averaging.
    zero_tol : float
        Absolute tolerance for treating values as zero (default exact zero).
    """
    if df_region.empty:
        return float("nan")

    d = df_region[["lat", "lon", metric]].copy()
    if exclude_zeros:
        m = ~np.isclose(d[metric].to_numpy(dtype=float), 0.0, atol=zero_tol, rtol=0.0)
        d = d[m]
        if d.empty:
            return float("nan")

    grid = d.pivot_table(index="lat", columns="lon", values=metric, aggfunc="mean")
    if grid.empty:
        return float("nan")

    da = xr.DataArray(
        grid.to_numpy(dtype=float),
        coords={"latitude": grid.index.to_numpy(dtype=float),
                "longitude": grid.columns.to_numpy(dtype=float)},
        dims=("latitude", "longitude"),
    )
    return float(_spatial_average(da, region=None, skipna=True).item())


# ----------------------------- Region masks ------------------------------------

def _normalize_lons(lons: np.ndarray) -> np.ndarray:
    """Normalize longitudes to [-180, 180] range for robust box checks."""
    l = np.asarray(lons, dtype=float)
    l = ((l + 180.0) % 360.0) - 180.0
    return l

def _make_2d(lat: np.ndarray, lon: np.ndarray):
    """Return 2D meshgrids from 1D latitude and longitude centers."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lat2d, lon2d

def _in_box(lat2d: np.ndarray, lon2d: np.ndarray,
            latmin: float, latmax: float, lonmin: float, lonmax: float) -> np.ndarray:
    """Boolean mask for a lat-lon box; supports dateline crossing by union of intervals."""
    if lonmin <= lonmax:
        cond_lon = (lon2d >= lonmin) & (lon2d <= lonmax)
    else:
        # Example: 145 to -130 crosses the dateline
        cond_lon = (lon2d >= lonmin) | (lon2d <= lonmax)
    return (lat2d >= latmin) & (lat2d <= latmax) & cond_lon

def build_region_masks(latitudes: np.ndarray, longitudes: np.ndarray) -> dict[str, np.ndarray]:
    """
    Build ECMWF/WeatherBench scorecard region masks on the provided grid.
    Returns dict[name] -> bool mask of shape [nlat, nlon].
    """
    lats = np.asarray(latitudes, dtype=float)
    lons = _normalize_lons(np.asarray(longitudes, dtype=float))
    lat2d, lon2d = _make_2d(lats, lons)

    # Broad latitude-defined regions
    masks = {
        "Tropics":          (lat2d >= -20.0) & (lat2d <=  20.0),
        "Extratropics":     (np.abs(lat2d) >= 20.0),
        "NH_extratropics":  (lat2d >=  20.0),
        "SH_extratropics":  (lat2d <= -20.0),
        "Arctic":           (lat2d >=  60.0),
        "Antarctic":        (lat2d <= -60.0),
    }

    # Box-defined subregions (lon in [-180, 180] after normalization)
    masks.update({
        "Europe":        _in_box(lat2d, lon2d, 35.0, 75.0,  -12.5,   42.5),
        "North_America": _in_box(lat2d, lon2d, 25.0, 60.0, -120.0,  -75.0),
        "North_Atlantic":_in_box(lat2d, lon2d, 25.0, 60.0,  -70.0,  -20.0),
        "North_Pacific": _in_box(lat2d, lon2d, 25.0, 60.0,  145.0, -130.0),  
        "East_Asia":     _in_box(lat2d, lon2d, 25.0, 60.0,  102.5,  150.0),
        "AusNZ":         _in_box(lat2d, lon2d,-45.0,-12.5,  120.0,  175.0),
    })
    return masks



# ----------------------------- Thresholds (unweighted) -------------------------

def compute_unweighted_regional_thresholds(
    observations: xr.Dataset,
    var: str,
    time_range,
    t_quantiles: list[float],
    region_masks: dict[str, np.ndarray],
    lead_time: np.timedelta64,                 
) -> dict[tuple[str, float], float]:
    """
    Compute pooled, unweighted regional quantile thresholds using the verifying
    window { y(t + lead_time) : t in time_range }.
    """
    # Build verifying-time slice by shifting the evaluation window by lead_time
    # (handles open/closed ends gracefully; xarray will clip to available data)
    start = time_range.start if isinstance(time_range, slice) else None
    stop  = time_range.stop  if isinstance(time_range, slice) else None

    # Use dataset bounds if start/stop are None
    tmin = observations.time.min().values if start is None else np.datetime64(pd.to_datetime(start))
    tmax = observations.time.max().values if stop  is None else np.datetime64(pd.to_datetime(stop))

    verif_slice = slice(tmin + lead_time, tmax + lead_time)

    # Select verifying observations for this variable
    arr = observations[var].sel(time=verif_slice).load()  # DataArray [time, latitude, longitude]

    thresholds = {}
    for name, mask2d in region_masks.items():
        mask_da = xr.DataArray(
            mask2d,
            coords={"latitude": observations.latitude, "longitude": observations.longitude},
            dims=("latitude", "longitude"),
        )
        vals = arr.where(mask_da).values
        x = vals[np.isfinite(vals)]
        for tq in t_quantiles:
            thresholds[(name, float(tq))] = float(np.quantile(x, tq)) if x.size else np.nan
    return thresholds


def _point_job_tw_regional(
    model: str,
    lat: float,
    lon: float,
    lead_time: np.timedelta64,
    var: str,
    region_name: str,
    t_quantiles: list[float],
    thresholds: dict[tuple[str, float], float],
    observations: xr.Dataset,
    forecast_ds: dict[str, xr.Dataset],
    time_range
):
    """
    Compute TW metrics at one gridpoint using the regional threshold(s).
    Returns a list of rows (one per t_quantile).
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
            raise ValueError("empty series")

        # Fit IDR once for this point
        fitted = idr(obs_arr, pd.DataFrame({"x": pred_arr}))
        prob_pred = fitted.predict(pd.DataFrame({"x": pred_arr}), digits=12)

        base = {
            "model": model,
            "lat": float(lat),
            "lon": float(lon),
            "variable": var,
            "lead_time": int(lead_time / np.timedelta64(1, "D")),
            "region": region_name,
        }

        rows = []
        for tq in t_quantiles:
            t = thresholds.get((region_name, float(tq)), np.nan)
            if not np.isfinite(t):
                # Still emit a row for schema consistency
                rows.append({**base,
                    "t_quantile": float(tq),
                    "t_value": np.nan,
                    "tw_tail": np.nan,
                    "tw_pc": np.nan,
                    "tw_pcs": np.nan,
                })
                continue

            if tq < 0.5:
                tw_val = tw_pc_small(prob_pred, obs_arr, t)
                tw_skill = tw_pcs_small(tw_val, obs_arr, t)
                tail = "lower"
            else:
                tw_val = tw_pc(prob_pred, obs_arr, t)
                tw_skill = tw_pcs(tw_val, obs_arr, t)
                tail = "upper"

            rows.append({**base,
                "t_quantile": float(tq),
                "t_value": float(t),
                "tw_tail": tail,
                "tw_pc": float(tw_val),
                "tw_pcs": float(tw_skill),
            })
        return rows

    except Exception as e:
        # Emit NaN rows for all tq on failure
        rows = []
        for tq in t_quantiles:
            rows.append({
                "model": model, "lat": float(lat), "lon": float(lon),
                "variable": var, "lead_time": int(lead_time / np.timedelta64(1, "D")),
                "region": region_name,
                "t_quantile": float(tq), "t_value": np.nan,
                "tw_tail": np.nan, "tw_pc": np.nan, "tw_pcs": np.nan
            })
        return rows

def compute_tw_scores_with_regional_thresholds(
    *,
    lead_times: list[np.timedelta64],
    obs_sources: dict,
    variables: list[str],
    time_range,
    t_quantiles: list[float],
    n_jobs: int = 20,
    output_dir: str = "score_data",
    regions_subset: list[str] | None = None,
    aggregate_csv_name: str = "regional_t_aggregates.csv",
):
    """
    Compute TW metrics using regional pooled thresholds.
    Saves per-(obs,var,lead,region) CSVs with suffix '_regional_t.csv'
    and a final CSV with WB2 area-weighted regional means across grid points.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, output_dir)
    os.makedirs(save_dir, exist_ok=True)

    all_point_rows = []  # collect rows for aggregation

    for obs_name, obs_info in obs_sources.items():
        print(f"\n[Regional-TW] Processing {obs_name.upper()} observations…")
        obs_path = obs_info["path"]
        model_names = obs_info["models"]

        observations = xr.open_zarr(obs_path, decode_timedelta=True)
        forecast_ds = {}
        for mdl in model_names:
            f_path = f"data/{mdl}_64x32.zarr"
            if not os.path.exists(f_path):
                print(f"Missing forecast file {f_path} – skip {mdl}")
                continue
            forecast_ds[mdl] = xr.open_zarr(f_path, decode_timedelta=True)

        if not forecast_ds:
            print("  No forecast datasets available; skipping this obs source.")
            continue

        lats = observations.latitude.values
        lons = observations.longitude.values

        # Build region masks and optionally restrict to a subset
        region_masks = build_region_masks(lats, lons)
        if regions_subset is not None:
            region_masks = {k: v for k, v in region_masks.items() if k in regions_subset}
            if not region_masks:
                print("  No regions after filtering; skipping.")
                continue

        for var in variables:
            print(f"  → variable: {var}")
            for lead_time in lead_times:
                lead_days = int(lead_time / np.timedelta64(1, "D"))
                print(f"    → lead_time: {lead_days}d")

                # thresholds on verifying window {y(t+τ): t ∈ time_range}
                thresholds = compute_unweighted_regional_thresholds(
                    observations=observations,
                    var=var,
                    time_range=time_range,
                    t_quantiles=t_quantiles,
                    region_masks=region_masks,
                    lead_time=lead_time,      
                )

                for region_name, mask2d in region_masks.items():
                    lat_idx, lon_idx = np.where(mask2d)
                    if lat_idx.size == 0:
                        # Save an empty CSV to keep file layout consistent
                        empty_df = pd.DataFrame(columns=[
                            "model","lat","lon","variable","lead_time","region",
                            "t_quantile","t_value","tw_tail","tw_pc","tw_pcs"
                        ])
                        csv_name = f"{var}_lead{lead_days}d_{obs_name}_{region_name}_regional_t.csv"
                        empty_df.to_csv(os.path.join(save_dir, csv_name), index=False)
                        continue

                    tasks = []
                    for i, j in zip(lat_idx, lon_idx):
                        lat = float(lats[i]); lon = float(lons[j])
                        for mdl in model_names:
                            if mdl not in forecast_ds:
                                continue
                            tasks.append((
                                mdl, lat, lon, lead_time, var,
                                region_name, t_quantiles, thresholds,
                                observations, forecast_ds, time_range
                            ))

                    rows_nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
                        delayed(_point_job_tw_regional)(*t) for t in tasks
                    )
                    rows = [r for rows_i in rows_nested for r in rows_i]
                    df = pd.DataFrame(rows)

                    # Save per-region CSV for this (obs,var,lead,region)
                    csv_name = f"{var}_lead{lead_days}d_{obs_name}_{region_name}_regional_t.csv"
                    df.to_csv(os.path.join(save_dir, csv_name), index=False)

                    # Stash for aggregation
                    df["obs_source"] = obs_name
                    all_point_rows.append(df)

    # ---------------- WB2 area-weighted regional means via _spatial_average -----

    if not all_point_rows:
        agg_empty = pd.DataFrame(columns=[
            "obs_source","region","variable","lead_time","model","t_quantile","tw_tail",
            "tw_pc_mean_wb2","tw_pcs_mean_wb2","n_points"
        ])
        agg_empty.to_csv(os.path.join(save_dir, aggregate_csv_name), index=False)
        print(f"\n[Regional-TW] Wrote empty aggregates → {os.path.join(save_dir, aggregate_csv_name)}")
        return

    big = pd.concat(all_point_rows, ignore_index=True)

    # Keep only columns needed for aggregation
    big = big[[
        "obs_source","region","variable","lead_time","model","t_quantile","tw_tail",
        "lat","lon","tw_pc","tw_pcs"
    ]].copy()

    def _agg_one_group(d: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "tw_pc_mean_wb2":  _wb2_weighted_mean_from_rows(d, "tw_pc"),
            "tw_pcs_mean_wb2": _wb2_weighted_mean_from_rows(d, "tw_pcs"),
            "n_points":        int(len(d))
        })

    gkeys = ["obs_source","region","variable","lead_time","model","t_quantile","tw_tail"]
    agg = big.groupby(gkeys, dropna=False).apply(_agg_one_group).reset_index()

    agg_path = os.path.join(save_dir, aggregate_csv_name)
    agg.to_csv(agg_path, index=False)
    print(f"\n[Regional-TW] Wrote area-weighted aggregates → {agg_path}")


# --------------------------------------------------------------------
# Score computation functions ("raw scores" with init time column needed for block permutation testing)
# --------------------------------------------------------------------
def _compute_metrics_raw_point(
    model: str,
    lat: float,
    lon: float,
    lead_time: np.timedelta64,
    var: str,
    *,
    obs_ds: xr.Dataset,
    forecast_ds: dict[str, xr.Dataset],
    time_range,
    t_quantiles: list[float],
    q_values: list[float],
):
    """
    Compute time-resolved scores at one grid point:
      - PC (CRPS) per time step
      - TW-PC_q for q in t_quantiles (lower/upper handled by q)
      - QW-PC_q for q in q_values

    Returns
    -------
    list[dict]
        One record per verifying time step.
    """
    # Forecast series at init times within the evaluation window
    preds = (
        forecast_ds[model]
        .sel(prediction_timedelta=lead_time,
             latitude=lat, longitude=lon, method="nearest")
        .sel(time=time_range)[var]
    )
    # Matching verifying observations at time + lead
    obs = (
        obs_ds
        .sel(latitude=lat, longitude=lon, method="nearest")
        .sel(time=preds.time + lead_time)[var]
    )

    times = preds.time.values
    x = np.asarray(preds.values)
    y = np.asarray(obs.values)
    if x.size == 0 or y.size == 0:
        return []

    m = np.isfinite(x) & np.isfinite(y)
    if not np.all(m):
        x = x[m]; y = y[m]; times = times[m]
        if x.size == 0:
            return []

    # IDR fit and probabilistic forecast at this grid point
    fitted = idr(y, pd.DataFrame({"x": x}))
    prob_pred = fitted.predict(pd.DataFrame({"x": x}), digits=12)

    # Attach TW/QW CRPS methods from your metric_functions
    type(prob_pred).tw_crps_small = tw_crps_small
    type(prob_pred).tw_crps = tw_crps
    type(prob_pred).qw_crps_small = qw_crps_small
    type(prob_pred).qw_crps = qw_crps

    crps_vals = np.asarray(prob_pred.crps(y), dtype=float)

    # Precompute TW/QW series (vectorized over time)
    tw_series = {}
    for q in t_quantiles:
        t_val = float(np.nanquantile(y, q))
        if q < 0.5:
            tw_series[q] = np.asarray(prob_pred.tw_crps_small(y, t_val), dtype=float)
        else:
            tw_series[q] = np.asarray(prob_pred.tw_crps(y, t_val), dtype=float)

    qw_series = {}
    for q in q_values:
        if q < 0.5:
            qw_series[q] = np.asarray(prob_pred.qw_crps_small(y, q=q), dtype=float)
        else:
            qw_series[q] = np.asarray(prob_pred.qw_crps(y, q=q), dtype=float)

    lead_days = int(lead_time / np.timedelta64(1, "D"))
    out = []
    for i, ts in enumerate(times):
        rec = {
            "model": model,
            "variable": var,
            "latitude": float(lat),
            "longitude": float(lon),
            "prediction_timedelta": lead_days,
            "time": pd.to_datetime(ts),
            "pc": float(crps_vals[i]),
        }
        for q, arr in tw_series.items():
            rec[f"tw_pc_{q}"] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
        for q, arr in qw_series.items():
            rec[f"qw_pc_{q}"] = float(arr[i]) if np.isfinite(arr[i]) else np.nan
        out.append(rec)
    return out

def compute_raw_gridpoint_time_series(
    *,
    lead_times,
    obs_sources,
    variables,
    time_range,
    t_quantiles,
    q_values,
    n_jobs: int = 20,
    output_dir: str = "score_data",
):
    """
    Compute raw, time-resolved gridpoint scores and write one Zarr per
    observation source to `output_dir` as:
        raw_scores_for_permutation_{obs_name}.zarr

    The Zarr schema follows a long-format conversion via pandas → xarray.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    for obs_name, info in obs_sources.items():
        print(f"\n[Raw gridpoint] {obs_name.upper()}")
        obs_ds = xr.open_zarr(info["path"], decode_timedelta=True)

        # available forecast datasets for this obs source
        forecast_ds = {}
        for mdl in info["models"]:
            fpath = os.path.join("data", f"{mdl}_64x32.zarr")
            if not os.path.exists(fpath):
                print(f"  Missing forecast file {fpath} – skip {mdl}")
                continue
            forecast_ds[mdl] = xr.open_zarr(fpath, decode_timedelta=True)
        if not forecast_ds:
            print("  No forecasts available; skipping.")
            continue

        lats = obs_ds.latitude.values
        lons = obs_ds.longitude.values

        all_records: list[dict] = []
        for var in variables:
            for lt in lead_times:
                ld = int(lt / np.timedelta64(1, "D"))
                print(f"  var={var} | lead={ld}d")

                tasks = [
                    (mdl, float(lat), float(lon), lt, var)
                    for lat in lats for lon in lons for mdl in info["models"]
                    if mdl in forecast_ds
                ]

                def _job(args):
                    mdl, lat, lon, lt_, v_ = args
                    return _compute_metrics_raw_point(
                        mdl, lat, lon, lt_, v_,
                        obs_ds=obs_ds,
                        forecast_ds=forecast_ds,
                        time_range=time_range,
                        t_quantiles=t_quantiles,
                        q_values=q_values,
                    )

                nested = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
                    delayed(_job)(a) for a in tasks
                )
                for rows in nested:
                    all_records.extend(rows)

        df = pd.DataFrame(all_records)
        if df.empty:
            print("  No records; skipping save.")
            continue

        idx = ["time", "latitude", "longitude", "prediction_timedelta", "model", "variable"]
        ds = df.set_index(idx).to_xarray()

        out_zarr = os.path.join(out_dir, f"raw_scores_for_permutation_{obs_name}.zarr")
        ds.to_zarr(out_zarr, mode="w")
        print(f"  • saved → {out_zarr}")

def compute_raw_regional_time_series_coslat(
    *,
    obs_sources,
    variables,
    lead_times,
    time_range,
    t_quantiles,
    n_jobs: int = 20,
    output_dir: str = "score_data",  
    write_point_csvs: bool = True,
) -> list[str]:
    """
    Compute raw, time-resolved point-level scores from pointwise IDR using
    region-wise thresholds (cos(latitude)-weighted) on the verifying window.

    Output (if write_point_csvs=True):
      pointraw_{obs}_{var}_lead{D}d_{region}.csv

    Returns
    -------
    list[str]
        List of written point-level CSV paths (empty if write_point_csvs=False).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- region mask builder (as in regional TW section) ----------
    def _normalize_lons(lons: np.ndarray) -> np.ndarray:
        return ((np.asarray(lons, dtype=float) + 180.0) % 360.0) - 180.0

    def _make_2d(lat: np.ndarray, lon: np.ndarray):
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d, lon2d

    def _in_box(lat2d, lon2d, latmin, latmax, lonmin, lonmax):
        cond_lon = (lon2d >= lonmin) & (lon2d <= lonmax) if lonmin <= lonmax else ((lon2d >= lonmin) | (lon2d <= lonmax))
        return (lat2d >= latmin) & (lat2d <= latmax) & cond_lon

    def _build_region_masks(latitudes, longitudes):
        lats = np.asarray(latitudes, dtype=float)
        lons = _normalize_lons(np.asarray(longitudes, dtype=float))
        lat2d, lon2d = _make_2d(lats, lons)
        masks = {
            "Tropics":         (lat2d >= -20.0) & (lat2d <= 20.0),
            "Extratropics":    (np.abs(lat2d) >= 20.0),
            "NH_extratropics": (lat2d >= 20.0),
            "SH_extratropics": (lat2d <= -20.0),
            "Arctic":          (lat2d >= 60.0),
            "Antarctic":       (lat2d <= -60.0),
        }
        masks.update({
            "Europe":        _in_box(lat2d, lon2d, 35.0, 75.0,  -12.5,  42.5),
            "North_America": _in_box(lat2d, lon2d, 25.0, 60.0, -120.0, -75.0),
            "North_Atlantic":_in_box(lat2d, lon2d, 25.0, 60.0,  -70.0, -20.0),
            "North_Pacific": _in_box(lat2d, lon2d, 25.0, 60.0,  145.0, -130.0),
            "East_Asia":     _in_box(lat2d, lon2d, 25.0, 60.0,  102.5, 150.0),
            "AusNZ":         _in_box(lat2d, lon2d,-45.0,-12.5, 120.0, 175.0),
            "Global":        np.ones_like(lat2d, dtype=bool),
        })
        return masks

    # ---------------- cos(lat)-weighted quantile on verifying window -----------
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not np.any(m):
            return np.nan
        v = values[m]; w = weights[m]
        idx = np.argsort(v); v = v[idx]; w = w[idx]
        cw = np.cumsum(w) / np.sum(w)
        j = np.searchsorted(cw, q, side="left")
        return float(v[np.clip(j, 0, v.size - 1)])

    def _region_thresholds_coslat(da: xr.DataArray, mask2d: np.ndarray,
                                  qs: list[float], lead: np.timedelta64, trange):
        # verifying window { y(t+lead) : t in trange }
        start = trange.start if isinstance(trange, slice) else None
        stop  = trange.stop  if isinstance(trange, slice) else None
        tmin = da.time.min().values if start is None else np.datetime64(pd.to_datetime(start))
        tmax = da.time.max().values if stop  is None else np.datetime64(pd.to_datetime(stop))
        verif = slice(tmin + lead, tmax + lead)

        arr = da.sel(time=verif).transpose("time", "latitude", "longitude")
        mask_da = xr.DataArray(mask2d,
                               coords={"latitude": arr.latitude, "longitude": arr.longitude},
                               dims=("latitude","longitude"))
        vals = arr.where(mask_da).values
        if vals.size == 0:
            return {float(q): np.nan for q in qs}

        T, Ny, Nx = vals.shape
        v = vals.reshape(T, Ny*Nx).ravel()
        cosw = np.cos(np.deg2rad(arr.latitude.values))[:, None]
        W2 = np.broadcast_to(cosw, (Ny, Nx)).reshape(Ny*Nx)
        ww = np.tile(W2, T)
        return {float(q): _weighted_quantile(v, ww, float(q)) for q in qs}

    # ---------------- main loop (no aggregation) --------------------------------
    point_csv_paths: list[str] = []

    for obs_name, info in obs_sources.items():
        print(f"\n[Raw regional] {obs_name.upper()}")
        obs_ds = xr.open_zarr(info["path"], decode_timedelta=True)

        # load available forecast datasets
        fc = {}
        for mdl in info["models"]:
            fpath = os.path.join("data", f"{mdl}_64x32.zarr")
            if os.path.exists(fpath):
                fc[mdl] = xr.open_zarr(fpath, decode_timedelta=True)
            else:
                print(f"  Missing forecast {fpath} – skip {mdl}")
        if not fc:
            print("  No forecasts available; skipping.")
            continue

        lats = obs_ds.latitude.values
        lons = obs_ds.longitude.values
        region_masks = _build_region_masks(lats, lons)

        for var in variables:
            obs_var = obs_ds[var].sel(time=time_range)
            for lt in lead_times:
                ld = int(lt / np.timedelta64(1, "D"))
                lt_td = np.timedelta64(ld, "D")

                preds_by_model = {
                    m: fc[m][var].sel(prediction_timedelta=lt_td).sel(time=time_range)
                    for m in fc.keys()
                }
                thr_by_region = {
                    r: _region_thresholds_coslat(obs_var, mask, t_quantiles, lt_td, time_range)
                    for r, mask in region_masks.items()
                }

                # iterate regions (compute & optionally write point-level CSVs only)
                for region_name, mask2d in region_masks.items():
                    lat_idx, lon_idx = np.where(mask2d)
                    if lat_idx.size == 0:
                        continue

                    def _point_job(mdl: str, lat: float, lon: float) -> pd.DataFrame:
                        preds = preds_by_model[mdl].sel(latitude=lat, longitude=lon, method="nearest")
                        init_times = pd.to_datetime(preds.time.values)
                        verif_times = init_times + pd.to_timedelta(ld, unit="D")

                        obs_point = (obs_var.sel(latitude=lat, longitude=lon, method="nearest")
                                           .reindex(time=verif_times, method="nearest",
                                                    tolerance=np.timedelta64(12, "h")))

                        x = np.asarray(preds.values)
                        y = np.asarray(obs_point.values)
                        msk = np.isfinite(x) & np.isfinite(y)
                        if not msk.any():
                            return pd.DataFrame()

                        x = x[msk]; y = y[msk]
                        it = pd.to_datetime(init_times[msk]).tz_localize(None)
                        vt = pd.to_datetime(verif_times[msk]).tz_localize(None)

                        fitted = idr(y, pd.DataFrame({"x": x}))
                        prob_pred = fitted.predict(pd.DataFrame({"x": x}), digits=12)
                        # attach TW routines
                        type(prob_pred).tw_crps_small = tw_crps_small
                        type(prob_pred).tw_crps = tw_crps

                        df = pd.DataFrame({
                            "obs_source": obs_name,
                            "region": region_name,
                            "variable": var,
                            "lead_time_d": ld,
                            "model": mdl,
                            "lat": float(lat),
                            "lon": float(lon),
                            "init_time": it,
                            "verif_time": vt,
                            "pc": np.asarray(prob_pred.crps(y), dtype=float),
                        })

                        # TW using region thresholds
                        for q in t_quantiles:
                            qf = float(q)
                            t_val = thr_by_region[region_name].get(qf, np.nan)
                            col_pc  = f"tw_pc_{qf}"
                            col_pcs = f"tw_pcs_{qf}"
                            if not np.isfinite(t_val):
                                df[col_pc] = np.nan
                                df[col_pcs] = np.nan
                                continue
                            if qf < 0.5:
                                tw_vals = np.asarray(prob_pred.tw_crps_small(y, t_val), dtype=float)
                                pcs_vals = np.asarray(tw_pcs_small(tw_vals, y, t_val), dtype=float)
                            else:
                                tw_vals = np.asarray(prob_pred.tw_crps(y, t_val), dtype=float)
                                pcs_vals = np.asarray(tw_pcs(tw_vals, y, t_val), dtype=float)
                            df[col_pc]  = tw_vals
                            df[col_pcs] = pcs_vals
                        return df

                    tasks = [(mdl, float(lats[i]), float(lons[j]))
                             for i, j in zip(lat_idx, lon_idx) for mdl in preds_by_model.keys()]
                    parts = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
                        delayed(_point_job)(*t) for t in tasks
                    )
                    if not parts:
                        continue

                    pts_df = pd.concat(
                        [p for p in parts if p is not None and not p.empty],
                        ignore_index=True
                    ) if any([p is not None for p in parts]) else pd.DataFrame()

                    if write_point_csvs and not pts_df.empty:
                        pcsv = os.path.join(out_dir, f"pointraw_{obs_name}_{var}_lead{ld}d_{region_name}.csv")
                        pts_df.to_csv(pcsv, index=False)
                        point_csv_paths.append(pcsv)

    return point_csv_paths


def build_regional_timeseries(
    *,
    score_dir: str,
    obs_source: str,  # {"ifs","era5"}
    out_csv: str,     # e.g. os.path.join(score_dir, f"regional_timeseries_coslat_{obs_source}.csv")
) -> str:
    """
    Build a long, time-resolved regional series from pointraw CSVs by applying
    WB2 cosine-latitude spatial means per timestamp.

    Input files expected (written by compute_raw_regional_time_series_coslat):
      pointraw_{obs_source}_{variable}_lead{D}d_{region}.csv

    Output columns:
      ['obs_source','region','variable','lead_time_d','model','time',
       'pc', 'tw_pc_*', 'tw_pcs_*', ...]
    """

    pattern = os.path.join(score_dir, f"pointraw_{obs_source}_*.csv")
    files = sorted(glob.glob(pattern)) or sorted(glob.glob(os.path.join(score_dir, "pointraw_*.csv")))
    if not files:
        raise FileNotFoundError(f"No pointraw CSVs found in {score_dir} for obs={obs_source}.")

    rows: list[dict] = []
    for f in files:
        dfp = pd.read_csv(f, parse_dates=["init_time","verif_time"])
        if dfp.empty:
            continue

        # Normalize column names if needed
        if "latitude" in dfp.columns:  dfp = dfp.rename(columns={"latitude":"lat"})
        if "longitude" in dfp.columns: dfp = dfp.rename(columns={"longitude":"lon"})

        need = {"obs_source","region","variable","lead_time_d","model","lat","lon","verif_time"}
        if not need.issubset(dfp.columns):
            continue

        metric_cols = ["pc"] + [c for c in dfp.columns if c.startswith(("tw_pc_","tw_pcs_"))]

        # Spatial mean per (obs, region, variable, lead, model, verifying time)
        grouped = dfp.groupby(
            ["obs_source","region","variable","lead_time_d","model","verif_time"],
            sort=False,
            dropna=False,
        )
        for (obs, reg, var, lead, mdl, vt), g in grouped:
            if obs != obs_source:
                continue
            base = {
                "obs_source": obs,
                "region": reg,
                "variable": var,
                "lead_time_d": int(lead),
                "model": mdl,
                "time": pd.to_datetime(vt),
            }
            for mcol in metric_cols:
                sub = g[["lat","lon",mcol]].dropna()
                base[mcol] = (
                    np.nan if sub.empty
                    else _wb2_weighted_mean_from_rows(sub.rename(columns={mcol: mcol}), mcol)
                )
            rows.append(base)

    out = pd.DataFrame(rows)
    out = out[out["obs_source"] == obs_source].copy()
    out.sort_values(["region","variable","lead_time_d","model","time"], inplace=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv




# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------

def run_scores(
    compute_classic: bool = False,
    compute_gaussian: bool = False,
    compute_regional: bool = False,
    compute_raw_gridpoint: bool = False,         
    compute_raw_regional: bool = False,          
):
    """
    Run different score computations depending on the specified flags.

    Parameters
    ----------
    compute_classic : bool
        If True, computes classical TW/QW scores (and base PC/PCS).
    compute_gaussian : bool
        If True, computes Gaussian-CDF TW scores merged with baseline TW.
    compute_regional : bool
        If True, computes TW scores with region-specific pooled thresholds.
    compute_raw_gridpoint : bool
        If True, computes raw, time-resolved gridpoint scores to Zarr.
    compute_raw_regional : bool
        If True, computes raw, cos-lat weighted regional time series to CSV.
    """
    if compute_classic:
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

    if compute_gaussian:
        compute_gaussian_tw_scores(
            lead_times=lead_times,
            obs_sources=obs_sources,
            variables=variables,
            time_range=time_range,
            t_quantiles=[0.9, 0.95, 0.99],
            n_jobs=20,
            output_dir="score_data",
            gauss_sigma=np.finfo(float).eps,
            sigma_is_fraction=True,
            fallback_sigma=1.0,
        )

    if compute_regional:
        compute_tw_scores_with_regional_thresholds(
            lead_times=lead_times,
            obs_sources=obs_sources,
            variables=variables,
            time_range=time_range,
            t_quantiles=t_quantiles,
            n_jobs=30,
            output_dir="score_data",
            regions_subset=None,
        )

    if compute_raw_gridpoint:
        compute_raw_gridpoint_time_series(
            lead_times=lead_times,
            obs_sources=obs_sources,
            variables=variables,
            time_range=time_range,
            t_quantiles=t_quantiles,
            q_values=q_values,
            n_jobs=n_jobs,
            output_dir="score_data",
        )

    if compute_raw_regional:
        compute_raw_regional_time_series_coslat(
            obs_sources=obs_sources,
            variables=variables,
            lead_times=lead_times,
            time_range=time_range,
            t_quantiles=t_quantiles,
            n_jobs=n_jobs,
            output_dir="score_data",
            write_point_csvs=True,
        )
         # build regional time-series caches for each observation source
        script_dir = os.path.dirname(os.path.abspath(__file__))
        score_dir = os.path.join(script_dir, "score_data")
        for obs_name in obs_sources.keys():  # e.g. "era5", "ifs"
            out_csv = os.path.join(score_dir, f"regional_timeseries_coslat_{obs_name}.csv")
            try:
                built = build_regional_timeseries(
                    score_dir=score_dir,
                    obs_source=obs_name,
                    out_csv=out_csv,
                )
                print(f"  • built regional timeseries cache → {built}")
            except FileNotFoundError as e:
                print(f"  • skip cache for {obs_name}: {e}")

if __name__ == "__main__":
    run_scores(
        compute_classic=False,
        compute_gaussian=False,
        compute_regional=False,

        # "raw scores" with init time column needed for block permutation testing:
        compute_raw_gridpoint=False,   # writes Zarr per obs → score_data/raw_scores_for_permutation_{obs}.zarr
        compute_raw_regional=False,    # writes CSVs        → score_data/regional_*_{obs}.csv
    )

    
    """
    rebuild_regional_t_aggregates_from_disk(
    score_dir=os.path.join(os.path.dirname(__file__), "score_data"),
    aggregate_csv_name="regional_t_aggregates.csv",
    # optional filters, e.g.:
    # obs_filter={"ifs"},
    # variables_filter={"mean_sea_level_pressure","2m_temperature","10m_wind_speed"},
    # lead_times_filter={1,3,5,7,10},
    # regions_filter={"Europe","North_America","Tropics","Arctic"},
    # models_filter={"hres","graphcast","pangu","graphcast_operational","pangu_operational"},
    )
    """


def rebuild_regional_t_aggregates_from_disk(
    *,
    score_dir=os.path.join(os.path.dirname(__file__), "score_data"),
    aggregate_csv_name: str = "regional_t_aggregates.csv",
    obs_filter: set[str] | None = None,          # e.g. {"ifs","era5"}
    variables_filter: set[str] | None = None,    # e.g. {"2m_temperature", ...}
    lead_times_filter: set[int] | None = None,   # e.g. {1,3,5,7,10}
    regions_filter: set[str] | None = None,      # e.g. {"Europe","Tropics",...}
    models_filter: set[str] | None = None        # e.g. {"hres","graphcast","pangu","graphcast_operational",...}
) -> str:
    """
    Rebuild the regional summary CSV purely from per-region files already on disk
    (files named: {var}_lead{D}d_{obs}_{region}_regional_t.csv).

    Output columns match the old aggregator:
      ["obs_source","region","variable","lead_time","model","t_quantile","tw_tail",
       "tw_pc_mean_wb2","tw_pcs_mean_wb2","n_points"]
    """
    # Find all per-region files
    try:
        files = [f for f in os.listdir(score_dir) if f.endswith("_regional_t.csv")]
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory not found: {score_dir!r}")

    # Pattern: var may contain underscores; obs has no underscores; region may contain underscores.
    pat = re.compile(r"(?P<var>.+)_lead(?P<lead>\d+)d_(?P<obs>[^_]+)_(?P<region>.+)_regional_t\.csv$")

    out_rows: list[dict] = []

    for fname in files:
        m = pat.match(fname)
        if not m:
            # Skip files that don't follow naming convention
            continue

        var    = m.group("var")
        lead   = int(m.group("lead"))
        obs    = m.group("obs")
        region = m.group("region")
        if (obs_filter is not None and obs not in obs_filter):             continue
        if (variables_filter is not None and var not in variables_filter): continue
        if (lead_times_filter is not None and lead not in lead_times_filter): continue
        if (regions_filter is not None and region not in regions_filter):   continue

        path = os.path.join(score_dir, fname)
        df = pd.read_csv(path)
        if df.empty:
            continue

        # Defensive: ensure needed columns exist
        need = {"model","lat","lon","t_quantile","tw_tail","tw_pc","tw_pcs"}
        missing = need.difference(df.columns)
        if missing:
            # Skip malformed files gracefully
            continue

        # Optional model filtering
        if models_filter is not None:
            df = df[df["model"].isin(models_filter)]
            if df.empty:
                continue

        # Compute area-weighted regional means per (model, t_quantile)
        for (model, tq), sub in df.groupby(["model","t_quantile"], dropna=False):
            # tw_tail should be constant within each level; pick the most frequent value
            tail = sub["tw_tail"].mode(dropna=False)
            tw_tail_val = (tail.iloc[0] if len(tail) else np.nan)

            # Number of unique grid points contributing
            n_points = int(sub[["lat","lon"]].drop_duplicates().shape[0])

            tw_pc_mean  = _wb2_weighted_mean_from_rows(sub[["lat","lon","tw_pc"]].dropna(subset=["tw_pc"]),   "tw_pc")
            tw_pcs_mean = _wb2_weighted_mean_from_rows(sub[["lat","lon","tw_pcs"]].dropna(subset=["tw_pcs"]), "tw_pcs")

            out_rows.append({
                "obs_source":         obs,
                "region":             region,
                "variable":           var,
                "lead_time":          lead,
                "model":              model,
                "t_quantile":         float(tq) if pd.notna(tq) else np.nan,
                "tw_tail":            tw_tail_val,
                "tw_pc_mean_wb2":     tw_pc_mean,
                "tw_pcs_mean_wb2":    tw_pcs_mean,
                "n_points":           n_points,
            })

    # Build DataFrame and write output
    agg = pd.DataFrame(out_rows, columns=[
        "obs_source","region","variable","lead_time","model","t_quantile","tw_tail",
        "tw_pc_mean_wb2","tw_pcs_mean_wb2","n_points"
    ])
    out_path = os.path.join(score_dir, aggregate_csv_name)
    agg.to_csv(out_path, index=False)
    print(f"[regional-aggregates] wrote → {out_path}")
    return out_path
