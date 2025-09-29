import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib.patches import Patch
from isodisreg import idr
from weatherbench2.metrics import _spatial_average

from metric_functions import (
    pc, pcs,
    tw_crps, tw_crps_small, tw_pc, tw_pc_small, tw_pcs, tw_pcs_small,
    qw_crps, qw_crps_small, qw_pc, qw_pc_small, qw_pcs, qw_pcs_small,
    tw_pc_gaussian_cdf, tw_pcs_gaussian_cdf,
    pc_ref_formula_small, qw_pc0_small
)


# --- Library defaults (edit only if changing library behaviour) ------------
TIME_RANGE = slice("2020-01-01", "2020-12-31")

OBS_SOURCES = {
    "era5": {"path": "data/era5_64x32.zarr",            "models": ["graphcast", "pangu", "hres"]},
    "ifs":  {"path": "data/ifs_analysis_64x32.zarr",     "models": ["pangu_operational", "graphcast_operational", "hres"]},
}
MODEL_ORDER = ["graphcast", "pangu", "hres"]


VARIABLES = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]

VAR_DISPLAY = {
    "mean_sea_level_pressure": "MSLP",
    "2m_temperature":          "T2M",
    "10m_wind_speed":          "WS10",
}
VAR_UNITS = {
    "mean_sea_level_pressure": "Pa",
    "2m_temperature":          "K",
    "10m_wind_speed":          "m s⁻¹",
}

Y_PAD_FRAC = 0.04  # symmetric y-padding used in plots

# Paths/compute defaults
HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "plots")
os.makedirs(SAVE_DIR, exist_ok=True)

# Parallelism defaults
N_JOBS  = 5
BACKEND = "loky"   # "loky" | "threading" | "multiprocessing"

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def display_model_name(mdl: str, obs_name: str) -> str:
    key = mdl.strip().lower()
    obs = obs_name.strip().lower()
    if key in {"graphcast", "graphcast_operational"}:
        return "GC-IFS" if obs == "ifs" else "GC-ERA5"
    if key in {"pangu", "pangu_operational"}:
        return "PW-IFS" if obs == "ifs" else "PW-ERA5"
    if key == "hres":
        return "HRES"
    return mdl

MODEL_ORDER = ["graphcast", "pangu", "hres"]

def score_family_and_metric(score: str):
    s = score.strip().upper()
    if s not in {"TW_PCS", "QW_PCS", "TW_PC", "QW_PC"}:
        raise ValueError(f"Unsupported score '{score}'. Use one of TW_PCS, QW_PCS, TW_PC, QW_PC.")
    family = "TW" if s.startswith("TW_") else "QW"
    metric = "PCS" if s.endswith("_PCS") else "PC"
    return family, metric

def csv_filename_for(var: str, lead_days: int, obs_name: str, score: str) -> str:
    # Keep original naming (separate files for PC and PCS).
    score_token = score.lower().replace("_", "")
    return f"sensitivity_{score_token}_localq_{var}_lead{lead_days}d_{obs_name}.csv"

def _csv_filename_for_ref_small(var: str, lead_days: int, obs_name: str, family: str) -> str:
    """
    File name for lower-tail reference PC^(0) curves.
    family ∈ {"TW","QW"}.
    """
    tok = f"{family.lower()}_pc0_small_ref_localq"
    return f"sensitivity_{tok}_{var}_lead{lead_days}d_{obs_name}.csv"
def csv_filename_for_small(var: str, lead_days: int, obs_name: str, score: str) -> str:
    """CSV naming for lower-tail ('small') sensitivity curves."""
    score_token = score.lower().replace("_", "")
    return f"sensitivity_small_{score_token}_localq_{var}_lead{lead_days}d_{obs_name}.csv"


def ensure_csv_exists_small(score_upper: str, var: str, lead_days: int):
    """
    Ensure the specific *_small CSV exists for the requested metric (PC/PCS) and family (TW/QW).
    If missing, build both PC and PCS for that family/variable/lead in one go.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_name = csv_filename_for_small(var, lead_days, OBS_NAME, score_upper)
    csv_path = os.path.join(SAVE_DIR, csv_name)
    if not os.path.exists(csv_path):
        family, _ = score_family_and_metric(score_upper)
        if family == "TW":
            _build_tw_small_csv(var, lead_days)
        else:
            _build_qw_small_csv(var, lead_days)



def _build_ref_pc0_small_csv(var: str, lead_days: int, family: str):
    """
    Build spatially weighted mean lower-tail reference PC^(0):
      - TW small: pc_ref_formula_small(y, t) with t = quantile(y, q) for q ∈ Q_GRID
      - QW small: qw_pc0_small(y, q) for q ∈ Q_GRID
    Uses OBS only and latitude-weighted spatial averaging.
    """
    observations, _, _ = _open_data()
    lats = observations.latitude.values
    lons = observations.longitude.values
    y_da = observations.sel(time=slice(TIME_RANGE.start, TIME_RANGE.stop))[var].load()

    def _point_curve(lat: float, lon: float):
        try:
            y = y_da.sel(latitude=lat, longitude=lon, method="nearest").values
            if y.size == 0 or np.all(~np.isfinite(y)):
                raise ValueError("empty/NaN series")
            vals = []
            if family == "TW":
                thr = np.nanquantile(y, Q_GRID_SMALL)
                for t in thr:
                    vals.append(float(pc_ref_formula_small(y, float(t))))
            else:
                for q in Q_GRID_SMALL:
                    vals.append(float(qw_pc0_small(y, float(q))))
            return True, float(lat), float(lon), np.asarray(vals, dtype=float)
        except Exception:
            return False, float(lat), float(lon), np.full(len(Q_GRID), np.nan, dtype=float)

    jobs = [(float(la), float(lo)) for la in lats for lo in lons]
    results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(
        delayed(_point_curve)(*a) for a in jobs
    )

    nl, nm = len(lats), len(lons)
    cube = np.full((nl, nm, len(Q_GRID)), np.nan)
    lat_to_i = {float(la): i for i, la in enumerate(lats)}
    lon_to_j = {float(lo): j for j, lo in enumerate(lons)}
    for ok, la, lo, arr in results:
        i, j = lat_to_i.get(la), lon_to_j.get(lo)
        if i is None or j is None:
            continue
        cube[i, j, :] = arr

    ref_mean = np.asarray([_spatial_weighted_mean(cube[:, :, k], lats, lons) for k in range(len(Q_GRID))], dtype=float)
    df = pd.DataFrame({
        "obs_source": OBS_NAME,
        "variable": var,
        "lead_time_days": lead_days,
        "quantile": Q_GRID_SMALL.astype(float),
        f"{family.lower()}_pc0_small_ref": ref_mean.astype(float),
    }).sort_values("quantile").reset_index(drop=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, _csv_filename_for_ref_small(var, lead_days, OBS_NAME, family))
    df.to_csv(out_path, index=False)
    print(f"[build][{family}-small] wrote reference PC^(0) -> {out_path}")


def _ensure_ref_pc0_small_csv(var: str, lead_days: int, family: str):
    """Ensure small reference CSV exists for (var, lead_days, family)."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, _csv_filename_for_ref_small(var, lead_days, OBS_NAME, family))
    if not os.path.exists(path):
        _build_ref_pc0_small_csv(var, lead_days, family)
    return path


def xaxis_label_for(score: str) -> str:
    s = score.strip().upper()
    if s.startswith("TW_"):
        return "Quantile for local threshold"
    else:  # QW_*
        return "Quantile"

def _spatial_weighted_mean(grid_vals: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> float:
    da = xr.DataArray(grid_vals, coords={"latitude": lats, "longitude": lons}, dims=("latitude", "longitude"))
    return float(_spatial_average(da, region=None, skipna=True).item())

def _csv_filename_for_ref(var: str, lead_days: int, obs_name: str, family: str) -> str:
    # family in {"TW","QW"}
    tok = f"{family.lower()}_pc0_ref_localq"
    return f"sensitivity_{tok}_{var}_lead{lead_days}d_{obs_name}.csv"

def _build_ref_pc0_csv(var: str, lead_days: int, family: str):
    """
    Build spatially weighted mean reference curves PC^(0) using explicit functions:
      - TW: _tw_pc0(y, t) with t = local quantile threshold of y, per q in Q_GRID
      - QW: qw_pc0(y, q)   (empirical definition you provided)
    These are computed from OBS only (no model dependency), then latitude-weighted averaged.
    """
    observations, _, _ = _open_data()
    lats = observations.latitude.values
    lons = observations.longitude.values

    y_da = observations.sel(time=slice(TIME_RANGE.start, TIME_RANGE.stop))[var].load()

    def _point_curve(lat: float, lon: float):
        try:
            y = y_da.sel(latitude=lat, longitude=lon, method="nearest").values
            if y.size == 0 or np.all(~np.isfinite(y)):
                raise ValueError("empty/NaN series")
            vals = []
            if family == "TW":
                thr = np.nanquantile(y, Q_GRID)
                for t in thr:
                    vals.append(_tw_pc0(y, float(t)))
            else:  # "QW"
                for q in Q_GRID:
                    vals.append(qw_pc0(y, float(q)))
            return True, float(lat), float(lon), np.asarray(vals, dtype=float)
        except Exception:
            return False, float(lat), float(lon), np.full(len(Q_GRID), np.nan, dtype=float)

    jobs = [(float(la), float(lo)) for la in lats for lo in lons]
    results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(
        delayed(_point_curve)(*a) for a in jobs
    )

    nl, nm = len(lats), len(lons)
    cube = np.full((nl, nm, len(Q_GRID)), np.nan)
    lat_to_i = {float(la): i for i, la in enumerate(lats)}
    lon_to_j = {float(lo): j for j, lo in enumerate(lons)}
    for ok, la, lo, arr in results:
        i, j = lat_to_i.get(la), lon_to_j.get(lo)
        if i is None or j is None:
            continue
        cube[i, j, :] = arr

    # spatial weighted mean per quantile
    ref_mean = []
    for k in range(len(Q_GRID)):
        ref_mean.append(_spatial_weighted_mean(cube[:, :, k], lats, lons))
    ref_mean = np.asarray(ref_mean, dtype=float)

    df = pd.DataFrame({
        "obs_source": OBS_NAME,
        "variable": var,
        "lead_time_days": lead_days,
        "quantile": Q_GRID.astype(float),
        f"{family.lower()}_pc0_ref": ref_mean.astype(float),
    }).sort_values("quantile").reset_index(drop=True)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, _csv_filename_for_ref(var, lead_days, OBS_NAME, family))
    df.to_csv(out_path, index=False)
    print(f"[build][{family}] wrote reference PC^(0) -> {out_path}")

def _ensure_ref_pc0_csv(var: str, lead_days: int, family: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, _csv_filename_for_ref(var, lead_days, OBS_NAME, family))
    if not os.path.exists(path):
        _build_ref_pc0_csv(var, lead_days, family)
    return path

# --------------------------- Metric helpers ---------------------------
# TW CRPS/PC/PCS

def _tw_crps(self, obs, t):
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1: raise ValueError("obs must be 1-D")
    if np.isnan(np.sum(y)): raise ValueError("obs contains NaN")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs length must be 1 or equal to number of predictions")

    def get_points(pred): return np.array(pred.points)
    def get_cdf(pred):    return np.array(pred.ecdf)
    def weights_from_cdf(cdf): return np.hstack([cdf[0], np.diff(cdf)])

    def tw_crps0(y_i, p_i, w_i, x_i, t_i):
        x_i = np.maximum(x_i, t_i)
        y_i = np.maximum(y_i, t_i)
        return 2 * np.sum(w_i * ((y_i < x_i).astype(float) - p_i + 0.5 * w_i) * (x_i - y_i))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(weights_from_cdf, p))
    T = [t] * len(y)
    return list(map(tw_crps0, y, p, w, x, T))

def _tw_pc(prob_pred, y, t):
    type(prob_pred).tw_crps = _tw_crps
    return float(np.mean(prob_pred.tw_crps(y, t)))


def _tw_pc0(y: np.ndarray, t: float) -> float:
    """
    TW reference PC^(0): pairwise average absolute difference AFTER clipping obs at threshold t.
    This is the same quantity used in _tw_pcs as the denominator.
    """
    y_thresh = np.maximum(y, t)
    return float(np.mean(np.abs(np.subtract.outer(y_thresh, y_thresh))) / 2.0)

def _tw_pcs(tw_pc_val, y, t):
    y_thresh = np.maximum(y, t)
    pc_ref = np.mean(np.abs(np.subtract.outer(y_thresh, y_thresh))) / 2.0
    if not np.isfinite(pc_ref) or pc_ref == 0:
        return np.nan
    return float((pc_ref - tw_pc_val) / pc_ref)

# QW CRPS/PC/PCS

def qw_crps0(y, w, x, q):
    c_cum = np.cumsum(w)
    c_cum_prev = np.hstack(([0], c_cum[:-1]))
    c_cum_star = np.maximum(c_cum, q)
    c_cum_prev_star = np.maximum(c_cum_prev, q)
    indicator = (x >= y).astype(float)
    terms = indicator * (c_cum_star - c_cum_prev_star) - 0.5 * (c_cum_star**2 - c_cum_prev_star**2)
    return 2 * np.sum(terms * (x - y))

def _qw_crps(self, obs, q=0.9):
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1: raise ValueError("obs must be 1-D")
    if np.isnan(np.sum(y)): raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or same length as predictions")

    def get_points(pred): return np.array(pred.points)
    def get_cdf(pred):    return np.array(pred.ecdf)
    def get_weights(cdf): return np.hstack([cdf[0], np.diff(cdf)])

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(get_weights, p))
    Q = [q] * len(y)
    return list(map(qw_crps0, y, w, x, Q))

def _qw_pc(prob_pred, y, q=0.9):
    type(prob_pred).qw_crps = _qw_crps
    return float(np.mean(prob_pred.qw_crps(y, q=q)))

def qw_pc0(y, q):
    y = np.sort(np.asarray(y))
    n = len(y)
    w = np.full(n, 1.0 / n)  # uniform weights

    pc_ref = sum(qw_crps0(y_i, w, y, q) for y_i in y) / n
    return pc_ref


# Kept for reference; qw_pc0(y,q) is the direct empirical equivalent used in this file.
def _climatological_qw_pc(y, q):
    x_dummy = np.zeros_like(y)
    fitted_idr = idr(y, pd.DataFrame({'x': x_dummy}))
    prob_pred = fitted_idr.predict(pd.DataFrame({'x': x_dummy}))
    type(prob_pred).qw_crps = _qw_crps
    return float(np.mean(prob_pred.qw_crps(y, q)))

def _qw_pcs(qw_pc_val, y, q):
    pc_ref = qw_pc0(y, q)
    return float((pc_ref - qw_pc_val) / pc_ref)

# Data access

def _open_data():
    obs_key = OBS_NAME.lower()
    meta = OBS_SOURCES[obs_key]
    observations = xr.open_zarr(meta["path"], decode_timedelta=True)
    model_names = meta["models"]
    forecast_ds = {}
    for mdl in model_names:
        f_path = f"data/{mdl}_64x32.zarr"
        if os.path.exists(f_path):
            forecast_ds[mdl] = xr.open_zarr(f_path, decode_timedelta=True)
    return observations, forecast_ds, model_names

# --------------------------- CSV builders with latitude-weighted mean + sanity check ---------------------------

def _build_tw_csv(var: str, lead_days: int):
    observations, forecast_ds, model_names = _open_data()
    lats = observations.latitude.values
    lons = observations.longitude.values
    lead_td = np.timedelta64(lead_days, "D")

    def score_point(model, lat, lon):
        try:
            preds = forecast_ds[model].sel(
                prediction_timedelta=lead_td, latitude=lat, longitude=lon, method="nearest",
            ).sel(time=TIME_RANGE)[var].load()
            obs = observations.sel(latitude=lat, longitude=lon, time=preds.time + lead_td)[var].load()
            x, y = preds.values, obs.values
            if x.size == 0 or y.size == 0:
                raise ValueError("empty series")
            fit = idr(y, pd.DataFrame({"x": x}))
            pp  = fit.predict(pd.DataFrame({"x": x}))
            base_pc  = float(np.mean(pp.crps(y)))
            base_pcs = float((np.mean(np.abs(np.subtract.outer(y, y))) / 2.0 - base_pc) / (np.mean(np.abs(np.subtract.outer(y, y))) / 2.0))
            thr = np.nanquantile(y, Q_GRID)
            vals_pc, vals_pcs = [], []
            for t in thr:
                twpc  = _tw_pc(pp, y, float(t))     # raw PC
                twpcs = _tw_pcs(twpc, y, float(t))  # normalized PCS
                vals_pc.append(twpc)
                vals_pcs.append(twpcs)
            return {"ok": True, "lat": float(preds.latitude), "lon": float(preds.longitude),
                    "pcs": base_pcs, "vals_pc": vals_pc, "vals_pcs": vals_pcs}
        except Exception:
            return {"ok": False, "lat": float(lat), "lon": float(lon), "pcs": np.nan,
                    "vals_pc": [np.nan]*len(Q_GRID), "vals_pcs": [np.nan]*len(Q_GRID)}

    rows_pc, rows_pcs = [], []
    for mdl in model_names:
        if mdl not in forecast_ds:
            print(f"[build][TW] skip model (no zarr): {mdl}")
            continue

        jobs = [(mdl, float(la), float(lo)) for la in lats for lo in lons]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(delayed(score_point)(*a) for a in jobs)

        nl, nm = len(lats), len(lons)
        cube_pc  = np.full((nl, nm, len(Q_GRID)), np.nan)
        cube_pcs = np.full((nl, nm, len(Q_GRID)), np.nan)
        pcs_grid = np.full((nl, nm), np.nan)
        lat_to_i = {float(la): i for i, la in enumerate(lats)}
        lon_to_j = {float(lo): j for j, lo in enumerate(lons)}
        for r in results:
            i, j = lat_to_i.get(float(r["lat"])), lon_to_j.get(float(r["lon"]))
            if i is None or j is None: continue
            pcs_grid[i, j]   = r["pcs"]
            cube_pc[i, j, :]  = r["vals_pc"]
            cube_pcs[i, j, :] = r["vals_pcs"]

        pcs_mean = _spatial_weighted_mean(pcs_grid, lats, lons)
        means_pc, means_pcs = [], []
        for k in range(len(Q_GRID)):
            means_pc.append(_spatial_weighted_mean(cube_pc[:, :, k],  lats, lons))
            means_pcs.append(_spatial_weighted_mean(cube_pcs[:, :, k], lats, lons))
        means_pc  = np.array(means_pc, dtype=float)
        means_pcs = np.array(means_pcs, dtype=float)

        # sanity check: q==0 TW_PCS should match PCS within 1e-3
        if not np.isnan(means_pcs[0]) and not np.isnan(pcs_mean):
            diff = float(means_pcs[0] - pcs_mean)
            if abs(diff) > 1e-3:
                print(f"[WARN][TW] q=0 mismatch for {mdl} {var}: diff={diff:.6g} (>1e-3)")

        for q, pc_val, pcs_val in zip(Q_GRID_SMALL, means_pc, means_pcs):
            rows_pc.append({
                "obs_source": OBS_NAME,
                "variable": var,
                "model": mdl,
                "lead_time_days": lead_days,
                "quantile": float(q),
                "tw_pc": float(pc_val),
            })
            rows_pcs.append({
                "obs_source": OBS_NAME,
                "variable": var,
                "model": mdl,
                "lead_time_days": lead_days,
                "quantile": float(q),
                "tw_pcs": float(pcs_val),
            })

    # Write two separate CSVs (keep original naming scheme)
    df_pc  = pd.DataFrame(rows_pc).sort_values(["model", "quantile"]).reset_index(drop=True)
    df_pcs = pd.DataFrame(rows_pcs).sort_values(["model", "quantile"]).reset_index(drop=True)

    csv_name_pc  = csv_filename_for(var, LEAD_DAYS, OBS_NAME, "TW_PC")
    csv_name_pcs = csv_filename_for(var, LEAD_DAYS, OBS_NAME, "TW_PCS")
    out_path_pc  = os.path.join(SAVE_DIR, csv_name_pc)
    out_path_pcs = os.path.join(SAVE_DIR, csv_name_pcs)
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_pc.to_csv(out_path_pc, index=False)
    df_pcs.to_csv(out_path_pcs, index=False)
    print(f"[build][TW] wrote -> {out_path_pc} and {out_path_pcs}")


def _build_qw_csv(var: str, lead_days: int):
    observations, forecast_ds, model_names = _open_data()
    lats = observations.latitude.values
    lons = observations.longitude.values
    lead_td = np.timedelta64(lead_days, "D")

    def score_point(model, lat, lon):
        try:
            preds = forecast_ds[model].sel(
                prediction_timedelta=lead_td, latitude=lat, longitude=lon, method="nearest",
            ).sel(time=TIME_RANGE)[var].load()
            obs = observations.sel(latitude=lat, longitude=lon, time=preds.time + lead_td)[var].load()
            x, y = preds.values, obs.values
            if x.size == 0 or y.size == 0:
                raise ValueError("empty series")
            fit = idr(y, pd.DataFrame({"x": x}))
            pp  = fit.predict(pd.DataFrame({"x": x}))
            base_pc  = float(np.mean(pp.crps(y)))
            base_pcs = float((np.mean(np.abs(np.subtract.outer(y, y))) / 2.0 - base_pc) / (np.mean(np.abs(np.subtract.outer(y, y))) / 2.0))

            vals_pc, vals_pcs = [], []
            for q in Q_GRID:
                qpc   = _qw_pc(pp, y, q=float(q))    # raw PC
                qwpcs = _qw_pcs(qpc, y, float(q))    # normalized PCS
                vals_pc.append(qpc)
                vals_pcs.append(qwpcs)

            return {"ok": True, "lat": float(preds.latitude), "lon": float(preds.longitude),
                    "pcs": base_pcs, "vals_pc": vals_pc, "vals_pcs": vals_pcs}
        except Exception:
            return {"ok": False, "lat": float(lat), "lon": float(lon), "pcs": np.nan,
                    "vals_pc": [np.nan]*len(Q_GRID), "vals_pcs": [np.nan]*len(Q_GRID)}

    rows_pc, rows_pcs = [], []
    for mdl in model_names:
        if mdl not in forecast_ds:
            print(f"[build][QW] skip model (no zarr): {mdl}")
            continue

        jobs = [(mdl, float(la), float(lo)) for la in lats for lo in lons]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(delayed(score_point)(*a) for a in jobs)

        nl, nm = len(lats), len(lons)
        cube_pc  = np.full((nl, nm, len(Q_GRID)), np.nan)
        cube_pcs = np.full((nl, nm, len(Q_GRID)), np.nan)
        pcs_grid = np.full((nl, nm), np.nan)
        lat_to_i = {float(la): i for i, la in enumerate(lats)}
        lon_to_j = {float(lo): j for j, lo in enumerate(lons)}
        for r in results:
            i, j = lat_to_i.get(float(r["lat"])), lon_to_j.get(float(r["lon"]))
            if i is None or j is None: continue
            pcs_grid[i, j]    = r["pcs"]
            cube_pc[i, j, :]  = r["vals_pc"]
            cube_pcs[i, j, :] = r["vals_pcs"]

        pcs_mean = _spatial_weighted_mean(pcs_grid, lats, lons)
        means_pc, means_pcs = [], []
        for k in range(len(Q_GRID)):
            means_pc.append(_spatial_weighted_mean(cube_pc[:, :, k],  lats, lons))
            means_pcs.append(_spatial_weighted_mean(cube_pcs[:, :, k], lats, lons))
        means_pc  = np.array(means_pc, dtype=float)
        means_pcs = np.array(means_pcs, dtype=float)

        # sanity check: q==0 QW_PCS should match PCS within 1e-3
        if not np.isnan(means_pcs[0]) and not np.isnan(pcs_mean):
            diff = float(means_pcs[0] - pcs_mean)
            if abs(diff) > 1e-3:
                print(f"[WARN][QW] q=0 mismatch for {mdl} {var}: diff={diff:.6g} (>1e-3)")

        for q, pc_val, pcs_val in zip(Q_GRID, means_pc, means_pcs):
            rows_pc.append({
                "obs_source": OBS_NAME,
                "variable": var,
                "model": mdl,
                "lead_time_days": lead_days,
                "quantile": float(q),
                "qw_pc": float(pc_val),
            })
            rows_pcs.append({
                "obs_source": OBS_NAME,
                "variable": var,
                "model": mdl,
                "lead_time_days": lead_days,
                "quantile": float(q),
                "qw_pcs": float(pcs_val),
            })

    # Write two separate CSVs (keep original naming scheme)
    df_pc  = pd.DataFrame(rows_pc).sort_values(["model", "quantile"]).reset_index(drop=True)
    df_pcs = pd.DataFrame(rows_pcs).sort_values(["model", "quantile"]).reset_index(drop=True)

    csv_name_pc  = csv_filename_for(var, LEAD_DAYS, OBS_NAME, "QW_PC")
    csv_name_pcs = csv_filename_for(var, LEAD_DAYS, OBS_NAME, "QW_PCS")
    out_path_pc  = os.path.join(SAVE_DIR, csv_name_pc)
    out_path_pcs = os.path.join(SAVE_DIR, csv_name_pcs)
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_pc.to_csv(out_path_pc, index=False)
    df_pcs.to_csv(out_path_pcs, index=False)
    print(f"[build][QW] wrote -> {out_path_pc} and {out_path_pcs}")
    
def _build_tw_small_csv(var: str, lead_days: int):
    """
    Build lower-tail (small) TW PC/PCS across Q_GRID at local thresholds t=quantile(y,q).
    """
    observations, forecast_ds, model_names = _open_data()
    lats = observations.latitude.values
    lons = observations.longitude.values
    lead_td = np.timedelta64(lead_days, "D")

    def score_point(model, lat, lon):
        try:
            preds = forecast_ds[model].sel(
                prediction_timedelta=lead_td, latitude=lat, longitude=lon, method="nearest",
            ).sel(time=TIME_RANGE)[var].load()
            obs = observations.sel(latitude=lat, longitude=lon, time=preds.time + lead_td)[var].load()
            x, y = preds.values, obs.values
            if x.size == 0 or y.size == 0:
                raise ValueError("empty series")

            fit = idr(y, pd.DataFrame({"x": x}))
            pp  = fit.predict(pd.DataFrame({"x": x}))

            thr = np.nanquantile(y, Q_GRID_SMALL)
            vals_pc, vals_pcs = [], []
            for t in thr:
                v_pc  = float(tw_pc_small(pp, y, float(t)))
                v_pcs = float(tw_pcs_small(v_pc, y, float(t)))
                vals_pc.append(v_pc); vals_pcs.append(v_pcs)
            return {"ok": True, "lat": float(preds.latitude), "lon": float(preds.longitude),
                    "vals_pc": vals_pc, "vals_pcs": vals_pcs}
        except Exception:
            return {"ok": False, "lat": float(lat), "lon": float(lon),
                    "vals_pc": [np.nan]*len(Q_GRID), "vals_pcs": [np.nan]*len(Q_GRID)}

    rows_pc, rows_pcs = [], []
    for mdl in model_names:
        if mdl not in forecast_ds:
            print(f"[build][TW-small] skip model (no zarr): {mdl}")
            continue

        jobs = [(mdl, float(la), float(lo)) for la in lats for lo in lons]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(
            delayed(score_point)(*a) for a in jobs
        )

        nl, nm = len(lats), len(lons)
        cube_pc  = np.full((nl, nm, len(Q_GRID)), np.nan)
        cube_pcs = np.full((nl, nm, len(Q_GRID)), np.nan)
        lat_to_i = {float(la): i for i, la in enumerate(lats)}
        lon_to_j = {float(lo): j for j, lo in enumerate(lons)}
        for r in results:
            i, j = lat_to_i.get(float(r["lat"])), lon_to_j.get(float(r["lon"]))
            if i is None or j is None: continue
            cube_pc[i, j, :]  = r["vals_pc"]
            cube_pcs[i, j, :] = r["vals_pcs"]

        means_pc  = np.asarray([_spatial_weighted_mean(cube_pc[:, :, k],  lats, lons) for k in range(len(Q_GRID))], dtype=float)
        means_pcs = np.asarray([_spatial_weighted_mean(cube_pcs[:, :, k], lats, lons) for k in range(len(Q_GRID))], dtype=float)

        for q, pc_val, pcs_val in zip(Q_GRID, means_pc, means_pcs):
            rows_pc.append({
                "obs_source": OBS_NAME, "variable": var, "model": mdl,
                "lead_time_days": lead_days, "quantile": float(q), "tw_pc_small": float(pc_val)
            })
            rows_pcs.append({
                "obs_source": OBS_NAME, "variable": var, "model": mdl,
                "lead_time_days": lead_days, "quantile": float(q), "tw_pcs_small": float(pcs_val)
            })

    df_pc  = pd.DataFrame(rows_pc).sort_values(["model", "quantile"]).reset_index(drop=True)
    df_pcs = pd.DataFrame(rows_pcs).sort_values(["model", "quantile"]).reset_index(drop=True)

    out_pc  = os.path.join(SAVE_DIR, csv_filename_for_small(var, lead_days, OBS_NAME, "TW_PC"))
    out_pcs = os.path.join(SAVE_DIR, csv_filename_for_small(var, lead_days, OBS_NAME, "TW_PCS"))
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_pc.to_csv(out_pc, index=False); df_pcs.to_csv(out_pcs, index=False)
    print(f"[build][TW-small] wrote -> {out_pc} and {out_pcs}")


def _build_qw_small_csv(var: str, lead_days: int):
    """
    Build lower-tail (small) QW PC/PCS across Q_GRID.
    """
    observations, forecast_ds, model_names = _open_data()
    lats = observations.latitude.values
    lons = observations.longitude.values
    lead_td = np.timedelta64(lead_days, "D")

    def score_point(model, lat, lon):
        try:
            preds = forecast_ds[model].sel(
                prediction_timedelta=lead_td, latitude=lat, longitude=lon, method="nearest",
            ).sel(time=TIME_RANGE)[var].load()
            obs = observations.sel(latitude=lat, longitude=lon, time=preds.time + lead_td)[var].load()
            x, y = preds.values, obs.values
            if x.size == 0 or y.size == 0:
                raise ValueError("empty series")

            fit = idr(y, pd.DataFrame({"x": x}))
            pp  = fit.predict(pd.DataFrame({"x": x}))

            vals_pc, vals_pcs = [], []
            for q in Q_GRID_SMALL:
                v_pc  = float(qw_pc_small(pp, y, float(q)))
                v_pcs = float(qw_pcs_small(v_pc, y, float(q)))
                vals_pc.append(v_pc); vals_pcs.append(v_pcs)
            return {"ok": True, "lat": float(preds.latitude), "lon": float(preds.longitude),
                    "vals_pc": vals_pc, "vals_pcs": vals_pcs}
        except Exception:
            return {"ok": False, "lat": float(lat), "lon": float(lon),
                    "vals_pc": [np.nan]*len(Q_GRID), "vals_pcs": [np.nan]*len(Q_GRID)}

    rows_pc, rows_pcs = [], []
    for mdl in model_names:
        if mdl not in forecast_ds:
            print(f"[build][QW-small] skip model (no zarr): {mdl}")
            continue

        jobs = [(mdl, float(la), float(lo)) for la in lats for lo in lons]
        results = Parallel(n_jobs=N_JOBS, backend=BACKEND, verbose=10)(
            delayed(score_point)(*a) for a in jobs
        )

        nl, nm = len(lats), len(lons)
        cube_pc  = np.full((nl, nm, len(Q_GRID)), np.nan)
        cube_pcs = np.full((nl, nm, len(Q_GRID)), np.nan)
        lat_to_i = {float(la): i for i, la in enumerate(lats)}
        lon_to_j = {float(lo): j for j, lo in enumerate(lons)}
        for r in results:
            i, j = lat_to_i.get(float(r["lat"])), lon_to_j.get(float(r["lon"]))
            if i is None or j is None: continue
            cube_pc[i, j, :]  = r["vals_pc"]
            cube_pcs[i, j, :] = r["vals_pcs"]

        means_pc  = np.asarray([_spatial_weighted_mean(cube_pc[:, :, k],  lats, lons) for k in range(len(Q_GRID))], dtype=float)
        means_pcs = np.asarray([_spatial_weighted_mean(cube_pcs[:, :, k], lats, lons) for k in range(len(Q_GRID))], dtype=float)

        for q, pc_val, pcs_val in zip(Q_GRID_SMALL, means_pc, means_pcs):
            rows_pc.append({
                "obs_source": OBS_NAME, "variable": var, "model": mdl,
                "lead_time_days": lead_days, "quantile": float(q), "qw_pc_small": float(pc_val)
            })
            rows_pcs.append({
                "obs_source": OBS_NAME, "variable": var, "model": mdl,
                "lead_time_days": lead_days, "quantile": float(q), "qw_pcs_small": float(pcs_val)
            })

    df_pc  = pd.DataFrame(rows_pc).sort_values(["model", "quantile"]).reset_index(drop=True)
    df_pcs = pd.DataFrame(rows_pcs).sort_values(["model", "quantile"]).reset_index(drop=True)

    out_pc  = os.path.join(SAVE_DIR, csv_filename_for_small(var, lead_days, OBS_NAME, "QW_PC"))
    out_pcs = os.path.join(SAVE_DIR, csv_filename_for_small(var, lead_days, OBS_NAME, "QW_PCS"))
    os.makedirs(SAVE_DIR, exist_ok=True)
    df_pc.to_csv(out_pc, index=False); df_pcs.to_csv(out_pcs, index=False)
    print(f"[build][QW-small] wrote -> {out_pc} and {out_pcs}")


# Plotting

def ensure_csv_exists(score_upper: str, var: str, lead_days: int):
    # Build if the specific (PC or PCS) CSV is missing.
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_name = csv_filename_for(var, lead_days, OBS_NAME, score_upper)
    if not os.path.exists(os.path.join(SAVE_DIR, csv_name)):
        family, _ = score_family_and_metric(score_upper)
        if family == "TW":
            _build_tw_csv(var, lead_days)
        else:
            _build_qw_csv(var, lead_days)


def plot_quantile_grid(score: str):
    score_upper = score.upper()
    family, metric = score_family_and_metric(score_upper)
    # Select y column by score
    if score_upper == "TW_PCS":
        y_col = "tw_pcs"
    elif score_upper == "QW_PCS":
        y_col = "qw_pcs"
    elif score_upper == "TW_PC":
        y_col = "tw_pc"
    else:
        y_col = "qw_pc"

    var_order = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]

    # Share y only for PCS (normalized, comparable across variables)
    share_y = (metric == "PCS")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=140, sharey=share_y)
    handles_all, labels_all = [], []

    # Per-axis min/max (used for PC)
    per_axis_minmax = []
    # Global min/max (used for PCS)
    y_min_global, y_max_global = np.inf, -np.inf

    for ax, var in zip(axes, var_order):
        ensure_csv_exists(score_upper, var, LEAD_DAYS)
        csv_path = os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, score_upper))
        if not os.path.exists(csv_path):
            ax.set_title(f"missing {VAR_DISPLAY.get(var, var)}")
            per_axis_minmax.append((np.nan, np.nan))
            continue

        df = pd.read_csv(csv_path).sort_values(["model", "quantile"])

        y_min_local, y_max_local = np.inf, -np.inf
        for mdl_key in MODEL_ORDER:
            options = [mdl_key, f"{mdl_key}_operational"]
            m_sub = df[df["model"].isin(options)]
            if m_sub.empty:
                continue
            actual_key = options[-1] if (options[-1] in m_sub["model"].unique()) else options[0]
            lbl = display_model_name(actual_key, OBS_NAME)
            h = ax.plot(m_sub["quantile"], m_sub[y_col], marker="o", linestyle="-", label=lbl)[0]
            handles_all.append(h)
            labels_all.append(lbl)
            y_min_local = min(y_min_local, m_sub[y_col].min())
            y_max_local = max(y_max_local, m_sub[y_col].max())

        ax.set_title(VAR_DISPLAY.get(var, var))
        ax.set_xlabel(xaxis_label_for(score_upper))
        ax.grid(True, alpha=0.3)

        per_axis_minmax.append((y_min_local, y_max_local))
        if share_y:
            y_min_global = min(y_min_global, y_min_local)
            y_max_global = max(y_max_global, y_max_local)

    # Apply y-limits + padding
    if share_y:
        if np.isfinite(y_min_global) and np.isfinite(y_max_global):
            y_range = y_max_global - y_min_global
            if y_range <= 0 or not np.isfinite(y_range):
                scale = abs(y_max_global) if np.isfinite(y_max_global) and y_max_global != 0 else 1.0
                y_pad = Y_PAD_FRAC * scale
            else:
                y_pad = Y_PAD_FRAC * y_range
            lower, upper = y_min_global - y_pad, y_max_global + y_pad
            for ax in axes:
                ax.set_ylim(lower, upper)
        else:
            for ax in axes:
                ax.set_ylim(0, 1)
    else:
        for ax, (y_min, y_max) in zip(axes, per_axis_minmax):
            if np.isfinite(y_min) and np.isfinite(y_max):
                y_range = y_max - y_min
                if y_range <= 0 or not np.isfinite(y_range):
                    scale = abs(y_max) if np.isfinite(y_max) and y_max != 0 else 1.0
                    y_pad = Y_PAD_FRAC * scale
                else:
                    y_pad = Y_PAD_FRAC * y_range
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            else:
                ax.set_ylim(0, 1)

    # Y-axis label (use the score string directly)
    axes[0].set_ylabel(score_upper)

    want_order = ["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME.lower() == "era5" else ["GC-IFS", "PW-IFS", "HRES"]
    uniq, uniq_handles = [], []
    for want in want_order:
        for h, l in zip(handles_all, labels_all):
            if l == want and l not in uniq:
                uniq.append(l)
                uniq_handles.append(h)
                break

    fig.tight_layout()
    fig.legend(uniq_handles, uniq, ncol=min(len(uniq), 6), loc="lower center", bbox_to_anchor=(0.5, -0.06), frameon=False)
    out_png = f"sensitivity_{score_upper.lower()}_ALLVARS_lead{LEAD_DAYS}d_{OBS_NAME}_quantileX.png"
    fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {os.path.join(SAVE_DIR, out_png)}")
    

## Quantile/ Threshold Decomposition Helpers
# --------------------------- Region → representative grid point ---------------------------

# Region definitions from your table (all longitudes interpreted in degrees East in [-180, 180])
REGION_DEFS = {
    "nhextratropics":   {"lat_cond": ("ge",  20.0)},                          # lat >=  20
    "shextratropics":   {"lat_cond": ("le", -20.0)},                          # lat <= -20
    "tropics":          {"lat_range": (-20.0, 20.0)},                         # -20 <= lat <= 20
    "extratropics":     {"abs_lat_ge": 20.0},                                 # |lat| >= 20
    "arctic":           {"lat_cond": ("ge",  60.0)},                          # lat >=  60
    "antarctic":        {"lat_cond": ("le", -60.0)},                          # lat <= -60
    "europe":           {"lat_range": (35.0, 75.0),   "lon_range": (-12.5,  42.5)},
    "north_america":    {"lat_range": (25.0, 60.0),   "lon_range": (-120.0, -75.0)},
    "north_atlantic":   {"lat_range": (25.0, 60.0),   "lon_range": (-70.0,  -20.0)},
    # Note: North Pacific crosses the dateline → wrap handled in _mask_lon_range()
    "north_pacific":    {"lat_range": (25.0, 60.0),   "lon_range": (145.0, -130.0)},
    "east_asia":        {"lat_range": (25.0, 60.0),   "lon_range": (102.5, 150.0)},
    "ausnz":            {"lat_range": (-45.0, -12.5), "lon_range": (120.0, 175.0)},
}
REGION_ALIASES = {
    "nh": "nhextratropics", "sh": "shextratropics", "et": "extratropics",
    "arct": "arctic", "ant": "antarctic", "eu": "europe", "na": "north_america",
    "natl": "north_atlantic", "npac": "north_pacific", "easia": "east_asia",
}

def _to_pm180(lon_vals: np.ndarray) -> np.ndarray:
    """Convert longitudes to [-180, 180] degrees East for consistent masking."""
    return ((np.asarray(lon_vals) + 180.0) % 360.0) - 180.0

def _mask_lon_range(lon_pm180: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Boolean mask for lon in [lo, hi] on [-180,180] axis.
    Wrap-around is supported: if hi < lo, the interval crosses the dateline.
    """
    if hi >= lo:
        return (lon_pm180 >= lo) & (lon_pm180 <= hi)
    # wrap case, e.g., 145 .. -130  (i.e., 145E..230E)
    return (lon_pm180 >= lo) | (lon_pm180 <= hi)

def _pick_point_in_region(ds: xr.Dataset, region: str) -> tuple[float, float]:
    """
    Pick a representative grid point inside 'region':
      - Build a lat/lon mask from REGION_DEFS on the dataset grid
      - Take the geometric center (mean lat/lon of masked grid cells)
      - Snap to the nearest grid coordinate present in ds
    Returns (lat_sel, lon_sel) in dataset coordinates.
    """
    key = region.strip().lower().replace(" ", "_")
    key = REGION_ALIASES.get(key, key)
    if key not in REGION_DEFS:
        raise ValueError(f"Unknown region '{region}'. Allowed: {sorted(REGION_DEFS.keys())}")

    spec = REGION_DEFS[key]
    lats = np.asarray(ds.latitude)
    lons_native = np.asarray(ds.longitude)
    lons_pm180 = _to_pm180(lons_native)

    # Latitude mask
    lat_mask = np.ones_like(lats, dtype=bool)
    if "lat_range" in spec:
        lo, hi = spec["lat_range"]; lat_mask &= (lats >= lo) & (lats <= hi)
    if "lat_cond" in spec:
        op, v = spec["lat_cond"]
        lat_mask &= (lats >= v) if op == "ge" else (lats <= v)
    if "abs_lat_ge" in spec:
        lat_mask &= (np.abs(lats) >= float(spec["abs_lat_ge"]))

    # Longitude mask (allow full-planet if not specified)
    lon_mask = np.ones_like(lons_pm180, dtype=bool)
    if "lon_range" in spec:
        lo, hi = spec["lon_range"]
        lon_mask = _mask_lon_range(lons_pm180, float(lo), float(hi))

    # Build 2D mask on the grid
    mask2d = np.outer(lat_mask, lon_mask)
    if not np.any(mask2d):
        raise RuntimeError(f"No grid cells found inside region '{region}' on this dataset grid.")

    # Coordinates of all masked cells
    lat_grid, lon_grid = np.meshgrid(lats, lons_pm180, indexing="ij")
    lat_sel = float(np.mean(lat_grid[mask2d]))
    lon_sel_pm180 = float(np.mean(lon_grid[mask2d]))

    # Snap to nearest actual coordinates present in the dataset
    lat_snap = float(ds.latitude.sel(latitude=lat_sel, method="nearest"))
    # Convert the mean lon back to dataset convention before snapping
    if lons_native.min() >= 0.0 and lons_native.max() <= 360.0:
        lon_for_snap = lon_sel_pm180 % 360.0
    else:
        lon_for_snap = lon_sel_pm180
    lon_snap = float(ds.longitude.sel(longitude=lon_for_snap, method="nearest"))

    return lat_snap, lon_snap

# Optional small helpers for display/logging
def fmt_lat(lat: float) -> str: return f"{abs(lat):.2f}°{'N' if lat>=0 else 'S'}"
def fmt_lon(lon: float) -> str:
    ll = ((lon + 180) % 360) - 180
    return f"{abs(ll):.2f}°{'E' if ll>=0 else 'W'}"



def _cdf_right_at(x: np.ndarray, cdf: np.ndarray, z: float) -> float:
    """
    Return F(z) from a step ECDF defined by (x, cdf), using the RIGHT limit.
    Ties at z are INCLUDED (event {Y <= z}).
    Safe for all edges (z below min(x) or above max(x)).
    """
    x = np.asarray(x, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    n = x.size
    if n == 0:
        return np.nan
    # index of first point with x > z
    k = int(np.searchsorted(x, z, side="right"))
    Fz = 0.0 if k == 0 else float(cdf[min(k - 1, n - 1)])
    return float(min(1.0, max(0.0, Fz)))


def _brier(prob: np.ndarray, event: np.ndarray) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    prob = np.asarray(prob, dtype=float)
    event = np.asarray(event, dtype=float)
    m = min(prob.size, event.size)
    if m == 0:
        return np.nan
    return float(np.mean((prob[:m] - event[:m]) ** 2))


def _ecdf_quantile_left_inverse(x: np.ndarray, cdf: np.ndarray, alpha: float) -> float:
    """
    Return F^{-1}(alpha) := inf{z : F(z) >= alpha} from a step ECDF (x, cdf).
    Uses the LEFT inverse. Safe for edges (alpha in [0,1]).
    """
    x = np.asarray(x, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    n = x.size
    if n == 0:
        return np.nan
    # first index where CDF >= alpha
    k = int(np.searchsorted(cdf, float(alpha), side="left"))
    k = min(max(k, 0), n - 1)
    return float(x[k])

def _qs_curve_from_idr(pp, y: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
    """
    Quantile score curve on q_grid according to
        QS_alpha = 2 * ( I{y <= F^{-1}(alpha)} - alpha ) * ( F^{-1}(alpha) - y )
    where the quantile F^{-1} is the LEFT inverse of the predictive CDF and the
    event uses a RIGHT-closed inequality (<=), matching the screenshot definition.
    """
    y = np.asarray(y, dtype=float)
    # Extract ECDF supports once per time step
    X = [np.asarray(pred.points, dtype=float) for pred in pp.predictions]
    F = [np.asarray(pred.ecdf,   dtype=float) for pred in pp.predictions]

    out = np.empty(len(q_grid), dtype=float)
    for k, alpha in enumerate(q_grid):
        # Predict the alpha-quantile at each time via ECDF left-inverse
        qhat = np.array([_ecdf_quantile_left_inverse(X[i], F[i], float(alpha))
                         for i in range(y.size)], dtype=float)
        # Event is right-closed: I{y <= qhat}
        event = (y <= qhat).astype(float)
        # QS per definition (factor 2)
        out[k] = float(2.0 * np.mean((event - float(alpha)) * (qhat - y)))
    return out

    
# --------------------------- Graphical help to understand sensitivity analysis results (plots (among others) PC(0)) --------------------------- 
def plot_interpretation_grid():
    """
    4x3 interpretation grid (full quantile range):
      Columns: [MSLP, T2M, WS10]
      Rows: [0] TW_PC (+ ref PC^(0)), [1] TW_PCS,
            [2] QW_PC (+ ref PC^(0)), [3] QW_PCS
    PCS rows (TW_PCS & QW_PCS) share one global y-scale across all columns.
    PC rows keep per-column scales.
    """
    var_order = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]

    # Ensure CSVs + reference curves exist
    for var in var_order:
        ensure_csv_exists("TW_PC",  var, LEAD_DAYS)
        ensure_csv_exists("TW_PCS", var, LEAD_DAYS)
        ensure_csv_exists("QW_PC",  var, LEAD_DAYS)
        ensure_csv_exists("QW_PCS", var, LEAD_DAYS)
        _ensure_ref_pc0_csv(var, LEAD_DAYS, "TW")
        _ensure_ref_pc0_csv(var, LEAD_DAYS, "QW")

    fig, axes = plt.subplots(4, 3, figsize=(13, 12.5), dpi=140, sharex="col")

    # Column titles
    for j, var in enumerate(var_order):
        axes[0, j].set_title(VAR_DISPLAY.get(var, var), fontsize=14)

    # Row labels
    axes[0, 0].set_ylabel("TW_PC", fontsize=14)
    axes[1, 0].set_ylabel("TW_PCS", fontsize=14)
    axes[2, 0].set_ylabel("QW_PC", fontsize=14)
    axes[3, 0].set_ylabel("QW_PCS", fontsize=14)

    handles_all, labels_all = [], []

    # Per-column ranges for PC rows
    pc_min_col = np.full(3, np.inf)
    pc_max_col = np.full(3, -np.inf)

    # One global PCS range (shared by TW_PCS & QW_PCS rows)
    pcs_min_global, pcs_max_global = np.inf, -np.inf

    def _pad(lo: float, hi: float) -> tuple[float, float]:
        """Add symmetric padding based on global Y_PAD_FRAC."""
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0.0, 1.0
        rng = hi - lo
        pad = Y_PAD_FRAC * (rng if (rng > 0 and np.isfinite(rng)) else (abs(hi) if (np.isfinite(hi) and hi != 0) else 1.0))
        return lo - pad, hi + pad

    def _plot_family(ax_pc, ax_pcs, var: str, family: str, col_idx: int):
        nonlocal pcs_min_global, pcs_max_global, pc_min_col, pc_max_col

        # Select CSVs and column names
        if family == "TW":
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "TW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "TW_PCS")))
            pc_col, pcs_col = "tw_pc", "tw_pcs"
            ref_path = _ensure_ref_pc0_csv(var, LEAD_DAYS, "TW"); ref_col = "tw_pc0_ref"
        else:
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "QW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "QW_PCS")))
            pc_col, pcs_col = "qw_pc", "qw_pcs"
            ref_path = _ensure_ref_pc0_csv(var, LEAD_DAYS, "QW"); ref_col = "qw_pc0_ref"

        df_pc  = df_pc.sort_values(["model", "quantile"])
        df_pcs = df_pcs.sort_values(["model", "quantile"])

        # Track local ranges to update global/column ranges
        y_min_pc,  y_max_pc  = np.inf, -np.inf
        y_min_pcs, y_max_pcs = np.inf, -np.inf

        # Plot all models
        for base in MODEL_ORDER:
            options = [base, f"{base}_operational"]
            sub_pc = df_pc[df_pc["model"].isin(options)]
            if sub_pc.empty:
                continue
            actual_key = options[-1] if (options[-1] in sub_pc["model"].unique()) else options[0]
            label = display_model_name(actual_key, OBS_NAME)

            # PC curve
            seg_pc = sub_pc[sub_pc["model"] == actual_key]
            h = ax_pc.plot(seg_pc["quantile"], seg_pc[pc_col], marker="o", markersize=3.5, linestyle="-", label=label)[0]
            handles_all.append(h); labels_all.append(label)
            y_min_pc = min(y_min_pc, seg_pc[pc_col].min()); y_max_pc = max(y_max_pc, seg_pc[pc_col].max())

            # PCS curve
            seg_pcs = df_pcs[df_pcs["model"] == actual_key]
            if not seg_pcs.empty:
                ax_pcs.plot(seg_pcs["quantile"], seg_pcs[pcs_col], marker="o", markersize=3.5, linestyle="-")
                y_min_pcs = min(y_min_pcs, seg_pcs[pcs_col].min()); y_max_pcs = max(y_max_pcs, seg_pcs[pcs_col].max())

        # Reference PC^(0) on PC row
        dfr = pd.read_csv(ref_path).sort_values("quantile")
        h_ref = ax_pc.plot(dfr["quantile"], dfr[ref_col], linestyle="--", linewidth=2.2, color="black",
                           label="(tw / qw) PC⁽⁰⁾")[0]
        handles_all.append(h_ref); labels_all.append("(tw / qw) PC⁽⁰⁾")
        y_min_pc = min(y_min_pc, dfr[ref_col].min()); y_max_pc = max(y_max_pc, dfr[ref_col].max())

        # Cosmetics
        ax_pc.grid(True, alpha=0.3); ax_pcs.grid(True, alpha=0.3)
        ax_pc.set_xlabel(xaxis_label_for(f"{family}_PC"),  fontsize=14)
        ax_pcs.set_xlabel(xaxis_label_for(f"{family}_PCS"), fontsize=14)
        ax_pc.tick_params(axis="both", labelsize=14); ax_pcs.tick_params(axis="both", labelsize=14)

        # Update ranges
        if np.isfinite(y_min_pc) and np.isfinite(y_max_pc):
            pc_min_col[col_idx] = min(pc_min_col[col_idx], y_min_pc)
            pc_max_col[col_idx] = max(pc_max_col[col_idx], y_max_pc)
        if np.isfinite(y_min_pcs) and np.isfinite(y_max_pcs):
            pcs_min_global = min(pcs_min_global, y_min_pcs)
            pcs_max_global = max(pcs_max_global, y_max_pcs)

    # Fill the grid
    for j, var in enumerate(var_order):
        _plot_family(axes[0, j], axes[1, j], var, "TW", col_idx=j)
    for j, var in enumerate(var_order):
        _plot_family(axes[2, j], axes[3, j], var, "QW", col_idx=j)

    # Apply y-limits: PC rows per column
    for j in range(3):
        lo, hi = _pad(pc_min_col[j], pc_max_col[j])
        for row in (0, 2):  # TW_PC, QW_PC
            axes[row, j].set_ylim(lo, hi)

    # Apply y-limits: ONE global scale for both PCS rows
    lo_pcs, hi_pcs = _pad(pcs_min_global, pcs_max_global)
    for j in range(3):
        axes[1, j].set_ylim(lo_pcs, hi_pcs)  # TW_PCS
        axes[3, j].set_ylim(lo_pcs, hi_pcs)  # QW_PCS

    # Legend (keep reference curve in legend)
    want_order = (["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME.lower() == "era5"
                  else ["GC-IFS", "PW-IFS", "HRES"]) + ["(tw / qw) PC⁽⁰⁾"]
    uniq, uniq_handles = [], []
    for want in want_order:
        for h, l in zip(handles_all, labels_all):
            if l == want and l not in uniq:
                uniq.append(l); uniq_handles.append(h); break

    fig.suptitle(f"Sensitivity to Quantile Levels ({OBS_NAME.upper()}, Lead Time={LEAD_DAYS}d)",
                 fontsize=20, y=0.97)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.legend(uniq_handles, uniq, ncol=min(len(uniq), 6),
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               frameon=False, fontsize=14)

    out_png = f"sensitivity_interpretation_4x3_lead{LEAD_DAYS}d_{OBS_NAME}.png"
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {os.path.join(SAVE_DIR, out_png)}")
    
    
def plot_interpretation_grid_zoom_last10(qmin: float = 0.9, qmax: float = 1.0):
    """
    Tail-zoom interpretation grid with unified PCS scale:
      - Columns: [MSLP, T2M, WS10]
      - Rows: [0] TW_PC, [1] TW_PCS, [2] QW_PC, [3] QW_PCS
    A single global y-range is used for BOTH PCS rows (rows 1 and 3) across all columns.
    PC rows keep independent per-column scales. Reference PC^(0) is drawn on PC rows.
    """
    var_order = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]

    # Ensure inputs/CSV/refs exist
    for var in var_order:
        ensure_csv_exists("TW_PC",  var, LEAD_DAYS)
        ensure_csv_exists("TW_PCS", var, LEAD_DAYS)
        ensure_csv_exists("QW_PC",  var, LEAD_DAYS)
        ensure_csv_exists("QW_PCS", var, LEAD_DAYS)
        _ensure_ref_pc0_csv(var, LEAD_DAYS, "TW")
        _ensure_ref_pc0_csv(var, LEAD_DAYS, "QW")

    fig, axes = plt.subplots(4, 3, figsize=(13, 12.5), dpi=140, sharex="col")

    # Column titles
    for j, var in enumerate(var_order):
        axes[0, j].set_title(VAR_DISPLAY.get(var, var), fontsize=14)

    # Row labels
    axes[0, 0].set_ylabel("TW_PC", fontsize=14)
    axes[1, 0].set_ylabel("TW_PCS", fontsize=14)
    axes[2, 0].set_ylabel("QW_PC", fontsize=14)
    axes[3, 0].set_ylabel("QW_PCS", fontsize=14)

    handles_all, labels_all = [], []

    # Per-column ranges for PC rows
    pc_min_col = np.full(3, np.inf)
    pc_max_col = np.full(3, -np.inf)

    # One global PCS range shared by TW_PCS and QW_PCS rows
    pcs_min_global, pcs_max_global = np.inf, -np.inf

    def _plot_family(ax_pc, ax_pcs, var: str, family: str, col_idx: int):
        nonlocal pcs_min_global, pcs_max_global, pc_min_col, pc_max_col

        if family == "TW":
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "TW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "TW_PCS")))
            pc_col, pcs_col = "tw_pc", "tw_pcs"
        else:
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "QW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for(var, LEAD_DAYS, OBS_NAME, "QW_PCS")))
            pc_col, pcs_col = "qw_pc", "qw_pcs"

        # Zoom to [qmin, qmax]
        df_pc  = df_pc[(df_pc["quantile"] >= qmin) & (df_pc["quantile"] <= qmax)].sort_values(["model", "quantile"])
        df_pcs = df_pcs[(df_pcs["quantile"] >= qmin) & (df_pcs["quantile"] <= qmax)].sort_values(["model", "quantile"])

        y_min_pc, y_max_pc = np.inf, -np.inf
        y_min_pcs, y_max_pcs = np.inf, -np.inf

        for base in MODEL_ORDER:
            options = [base, f"{base}_operational"]
            m_sub_pc = df_pc[df_pc["model"].isin(options)]
            if m_sub_pc.empty:
                continue
            actual_key = options[-1] if (options[-1] in m_sub_pc["model"].unique()) else options[0]
            label = display_model_name(actual_key, OBS_NAME)

            # PC row
            h = ax_pc.plot(m_sub_pc["quantile"], m_sub_pc[pc_col],
                           marker="o", markersize=3.5, linestyle="-", label=label)[0]
            handles_all.append(h); labels_all.append(label)
            y_min_pc = min(y_min_pc, m_sub_pc[pc_col].min()); y_max_pc = max(y_max_pc, m_sub_pc[pc_col].max())

            # PCS row
            m_sub_pcs = df_pcs[df_pcs["model"] == actual_key]
            if not m_sub_pcs.empty:
                ax_pcs.plot(m_sub_pcs["quantile"], m_sub_pcs[pcs_col],
                            marker="o", markersize=3.5, linestyle="-")
                y_min_pcs = min(y_min_pcs, m_sub_pcs[pcs_col].min()); y_max_pcs = max(y_max_pcs, m_sub_pcs[pcs_col].max())

        # Reference curve on PC row
        ref_path = _ensure_ref_pc0_csv(var, LEAD_DAYS, family)
        dfr = pd.read_csv(ref_path).sort_values("quantile")
        dfr = dfr[(dfr["quantile"] >= qmin) & (dfr["quantile"] <= qmax)]
        ref_col = f"{family.lower()}_pc0_ref"
        h_ref = ax_pc.plot(dfr["quantile"], dfr[ref_col],
                           linestyle="--", linewidth=2.2, color="black",
                           label="(tw / qw) PC⁽⁰⁾")[0]
        handles_all.append(h_ref); labels_all.append("(tw / qw) PC⁽⁰⁾")
        if not dfr.empty:
            y_min_pc = min(y_min_pc, dfr[ref_col].min()); y_max_pc = max(y_max_pc, dfr[ref_col].max())

        # Cosmetics
        ax_pc.grid(True, alpha=0.3); ax_pcs.grid(True, alpha=0.3)
        ax_pc.set_xlabel(xaxis_label_for(f"{family}_PC"), fontsize=14)
        ax_pcs.set_xlabel(xaxis_label_for(f"{family}_PCS"), fontsize=14)
        ax_pc.tick_params(axis="both", labelsize=14); ax_pcs.tick_params(axis="both", labelsize=14)
        ax_pc.set_xlim(qmin, qmax); ax_pcs.set_xlim(qmin, qmax)

        # Update ranges
        if np.isfinite(y_min_pc) and np.isfinite(y_max_pc):
            pc_min_col[col_idx] = min(pc_min_col[col_idx], y_min_pc)
            pc_max_col[col_idx] = max(pc_max_col[col_idx], y_max_pc)
        if np.isfinite(y_min_pcs) and np.isfinite(y_max_pcs):
            pcs_min_global = min(pcs_min_global, y_min_pcs)
            pcs_max_global = max(pcs_max_global, y_max_pcs)

    # Fill panels
    for j, var in enumerate(var_order):
        _plot_family(axes[0, j], axes[1, j], var, "TW", col_idx=j)
    for j, var in enumerate(var_order):
        _plot_family(axes[2, j], axes[3, j], var, "QW", col_idx=j)

    # Apply y-limits:
    # 1) PC rows — per column
    for j in range(3):
        y_min = pc_min_col[j]; y_max = pc_max_col[j]
        if np.isfinite(y_min) and np.isfinite(y_max):
            rng = y_max - y_min
            pad = Y_PAD_FRAC * (rng if rng > 0 and np.isfinite(rng)
                                else (abs(y_max) if np.isfinite(y_max) and y_max != 0 else 1.0))
            for row in (0, 2):  # TW_PC, QW_PC
                axes[row, j].set_ylim(y_min - pad, y_max + pad)
        else:
            for row in (0, 2):
                axes[row, j].set_ylim(0, 1)

    # 2) PCS rows — single global scale shared by both rows
    if np.isfinite(pcs_min_global) and np.isfinite(pcs_max_global):
        rng = pcs_max_global - pcs_min_global
        pad = Y_PAD_FRAC * (rng if rng > 0 and np.isfinite(rng)
                            else (abs(pcs_max_global) if np.isfinite(pcs_max_global) and pcs_max_global != 0 else 1.0))
        lo, hi = pcs_min_global - pad, pcs_max_global + pad
        for j in range(3):
            axes[1, j].set_ylim(lo, hi)  # TW_PCS
            axes[3, j].set_ylim(lo, hi)  # QW_PCS
    else:
        for j in range(3):
            axes[1, j].set_ylim(0, 1)
            axes[3, j].set_ylim(0, 1)

    # Legend (ordered + reference)
    want_order = (["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME.lower() == "era5"
                  else ["GC-IFS", "PW-IFS", "HRES"]) + ["(tw / qw) PC⁽⁰⁾"]
    uniq, uniq_handles = [], []
    for want in want_order:
        for h, l in zip(handles_all, labels_all):
            if l == want and l not in uniq:
                uniq.append(l); uniq_handles.append(h); break

    fig.suptitle(
        f"Sensitivity to Quantile Levels — tail zoom q ∈ [{qmin:.2f}, {qmax:.2f}] "
        f"({OBS_NAME.upper()}, Lead Time={LEAD_DAYS} days)",
        fontsize=20, y=0.97
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.legend(uniq_handles, uniq, ncol=min(len(uniq), 6),
               loc="lower center", bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=14)

    out_png = f"sensitivity_interpretation_4x3_tail_q{int(qmin*100)}-{int(qmax*100)}_lead{LEAD_DAYS}d_{OBS_NAME}.png"
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {os.path.join(SAVE_DIR, out_png)}")


    
def plot_interpretation_grid_small():
    """
    4x2 interpretation grid (small extremes):
      Columns: [MSLP – low extremes, T2M – cold extremes]
      Rows: [0] TW_PC (+ small ref), [1] TW_PCS,
            [2] QW_PC (+ small ref), [3] QW_PCS
    PCS rows (TW_PCS & QW_PCS) share one global y-scale across both columns.
    PC rows keep per-column scales.
    """
    var_order  = ["mean_sea_level_pressure", "2m_temperature"]
    col_titles = ["MSLP - low extremes", "T2M - cold extremes"]

    # Ensure CSVs + small reference curves exist
    for var in var_order:
        ensure_csv_exists_small("TW_PC",  var, LEAD_DAYS)
        ensure_csv_exists_small("TW_PCS", var, LEAD_DAYS)
        ensure_csv_exists_small("QW_PC",  var, LEAD_DAYS)
        ensure_csv_exists_small("QW_PCS", var, LEAD_DAYS)
        _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "TW")
        _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "QW")

    fig, axes = plt.subplots(4, 2, figsize=(13, 12.5), dpi=140, sharex="col")

    # Column titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)

    # Row labels
    axes[0, 0].set_ylabel("TW_PC", fontsize=14)
    axes[1, 0].set_ylabel("TW_PCS", fontsize=14)
    axes[2, 0].set_ylabel("QW_PC", fontsize=14)
    axes[3, 0].set_ylabel("QW_PCS", fontsize=14)

    handles_all, labels_all = [], []

    # Per-column ranges for PC rows
    pc_min_col = np.full(2, np.inf)
    pc_max_col = np.full(2, -np.inf)

    # One global PCS range (shared by TW_PCS & QW_PCS)
    pcs_min_global, pcs_max_global = np.inf, -np.inf

    def _pad(lo: float, hi: float) -> tuple[float, float]:
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0.0, 1.0
        rng = hi - lo
        pad = Y_PAD_FRAC * (rng if (rng > 0 and np.isfinite(rng)) else (abs(hi) if (np.isfinite(hi) and hi != 0) else 1.0))
        return lo - pad, hi + pad

    def _plot_family_small(ax_pc, ax_pcs, var: str, family: str, col_idx: int):
        nonlocal pcs_min_global, pcs_max_global, pc_min_col, pc_max_col

        if family == "TW":
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "TW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "TW_PCS")))
            pc_col, pcs_col = "tw_pc_small", "tw_pcs_small"
            ref_path = _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "TW"); ref_col = "tw_pc0_small_ref"
        else:
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "QW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "QW_PCS")))
            pc_col, pcs_col = "qw_pc_small", "qw_pcs_small"
            ref_path = _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "QW"); ref_col = "qw_pc0_small_ref"

        df_pc  = df_pc.sort_values(["model", "quantile"])
        df_pcs = df_pcs.sort_values(["model", "quantile"])

        y_min_pc,  y_max_pc  = np.inf, -np.inf
        y_min_pcs, y_max_pcs = np.inf, -np.inf

        for base in MODEL_ORDER:
            options = [base, f"{base}_operational"]
            avail_pc = df_pc[df_pc["model"].isin(options)]
            if avail_pc.empty:
                continue
            actual_key = options[-1] if (options[-1] in avail_pc["model"].unique()) else options[0]
            label = display_model_name(actual_key, OBS_NAME)

            seg_pc = df_pc[df_pc["model"] == actual_key]
            h = ax_pc.plot(seg_pc["quantile"], seg_pc[pc_col], marker="o", markersize=3.5, linestyle="-", label=label)[0]
            handles_all.append(h); labels_all.append(label)
            y_min_pc = min(y_min_pc, seg_pc[pc_col].min()); y_max_pc = max(y_max_pc, seg_pc[pc_col].max())

            seg_pcs = df_pcs[df_pcs["model"] == actual_key]
            if not seg_pcs.empty:
                ax_pcs.plot(seg_pcs["quantile"], seg_pcs[pcs_col], marker="o", markersize=3.5, linestyle="-")
                y_min_pcs = min(y_min_pcs, seg_pcs[pcs_col].min()); y_max_pcs = max(y_max_pcs, seg_pcs[pcs_col].max())

        # Small reference PC^(0) on PC row
        dfr = pd.read_csv(ref_path).sort_values("quantile")
        h_ref = ax_pc.plot(dfr["quantile"], dfr[ref_col], linestyle="--", linewidth=2.2, color="black",
                           label="(tw / qw) PC⁽⁰⁾")[0]
        handles_all.append(h_ref); labels_all.append("(tw / qw) PC⁽⁰⁾")
        y_min_pc = min(y_min_pc, dfr[ref_col].min()); y_max_pc = max(y_max_pc, dfr[ref_col].max())

        # Cosmetics
        ax_pc.grid(True, alpha=0.3); ax_pcs.grid(True, alpha=0.3)
        ax_pc.set_xlabel(xaxis_label_for(f"{family}_PC"),  fontsize=14)
        ax_pcs.set_xlabel(xaxis_label_for(f"{family}_PCS"), fontsize=14)
        ax_pc.tick_params(axis="both", labelsize=14); ax_pcs.tick_params(axis="both", labelsize=14)

        # Update ranges
        if np.isfinite(y_min_pc) and np.isfinite(y_max_pc):
            pc_min_col[col_idx] = min(pc_min_col[col_idx], y_min_pc)
            pc_max_col[col_idx] = max(pc_max_col[col_idx], y_max_pc)
        if np.isfinite(y_min_pcs) and np.isfinite(y_max_pcs):
            pcs_min_global = min(pcs_min_global, y_min_pcs)
            pcs_max_global = max(pcs_max_global, y_max_pcs)

    # Fill panels
    for j, var in enumerate(var_order):
        _plot_family_small(axes[0, j], axes[1, j], var, "TW", col_idx=j)
    for j, var in enumerate(var_order):
        _plot_family_small(axes[2, j], axes[3, j], var, "QW", col_idx=j)

    # Apply y-limits: PC rows per column
    for j in range(2):
        lo, hi = _pad(pc_min_col[j], pc_max_col[j])
        for row in (0, 2):  # TW_PC, QW_PC
            axes[row, j].set_ylim(lo, hi)

    # Apply y-limits: ONE global scale for both PCS rows
    lo_pcs, hi_pcs = _pad(pcs_min_global, pcs_max_global)
    for j in range(2):
        axes[1, j].set_ylim(lo_pcs, hi_pcs)  # TW_PCS
        axes[3, j].set_ylim(lo_pcs, hi_pcs)  # QW_PCS

    # Legend (ordered + reference)
    want_order = (["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME.lower() == "era5"
                  else ["GC-IFS", "PW-IFS", "HRES"]) + ["(tw / qw) PC⁽⁰⁾"]
    uniq, uniq_handles = [], []
    for want in want_order:
        for h, l in zip(handles_all, labels_all):
            if l == want and l not in uniq:
                uniq.append(l); uniq_handles.append(h); break

    fig.suptitle(f"Sensitivity to Quantile Levels - small extremes \n ({OBS_NAME.upper()}, Lead Time={LEAD_DAYS} days)",
                 fontsize=20, y=0.96)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.legend(uniq_handles, uniq, ncol=min(len(uniq), 6),
               loc="lower center", bbox_to_anchor=(0.5, -0.005),
               frameon=False, fontsize=14)

    out_png = f"sensitivity_interpretation_small_4x2_lead{LEAD_DAYS}d_{OBS_NAME}.png"
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {os.path.join(SAVE_DIR, out_png)}")

    
def _bs_curve_from_idr_zgrid(pp, y: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
    """
    Brier-score curve on a direct threshold grid z_grid for the event {Y <= z}.
    Implements PS(F(z), 1{Y<=z}) with F taken as the RIGHT limit of the ECDF.
    """
    y = np.asarray(y, dtype=float)
    X = [np.asarray(pr.points, dtype=float) for pr in pp.predictions]
    F = [np.asarray(pr.ecdf,   dtype=float) for pr in pp.predictions]

    out = np.empty(len(z_grid), dtype=float)
    for k, zk in enumerate(np.asarray(z_grid, dtype=float)):
        event = (y <= zk).astype(float)
        prob  = np.array([_cdf_right_at(X[i], F[i], zk) for i in range(y.size)], dtype=float)
        out[k] = _brier(prob, event)
    return out


def plot_bs_qs_gridpoint(
    score_type: str = "BS",
    ground_truth: str = "ifs",
    pct_extremes: float = 5.0,
    lat: float | None = None,
    lon: float | None = None,
    region: str | None = None,
    lead_times: list[int] = (1, 3, 5, 7, 10),
    debug: bool = True,
):
    """
    Single-point BS/QS panel (5x3).
    - BS is plotted against physical thresholds z with numeric ticks and unit-aware labels.
    - QS is plotted against quantiles alpha in (0,1).
    - Extreme regions are shaded (two tails for MSLP/T2M, upper tail for WS10).

    Requirements available in module scope:
      OBS_SOURCES, MODEL_ORDER, VAR_DISPLAY, Q_GRID, TIME_RANGE,
      display_model_name(), _open_data(), idr, pandas as pd, xarray as xr, numpy as np,
      _cdf_right_at(), _brier().

    Returns
    -------
    (lat_used, lon_used) : tuple[float, float]
        The actual dataset coordinates used.
    """
    import traceback
    from matplotlib.patches import Patch

    # Variable → unit for axis labels (fallbacks if not defined in caller)
    default_var_units = {
        "mean_sea_level_pressure": "Pa",    # use "hPa" if you rescale values consistently
        "2m_temperature":          "K",
        "10m_wind_speed":          "m s⁻¹",
    }
    var_units = globals().get("VAR_UNITS", default_var_units)

    def dbg(*a, **k):
        if debug: print(*a, **k)

    # Helper for pretty lat/lon in legend tags
    def _fmt_lat(latv: float) -> str: return f"{abs(latv):.2f}°{'N' if latv >= 0 else 'S'}"
    def _fmt_lon(lonv: float) -> str:
        ll = ((lonv + 180) % 360) - 180
        return f"{abs(ll):.2f}°{'E' if ll >= 0 else 'W'}"

    assert score_type.upper() in {"BS", "QS"}
    global OBS_NAME
    prev_obs = OBS_NAME
    try:
        OBS_NAME = ground_truth.strip().lower()
        observations, forecast_ds, model_tokens = _open_data()
        dbg(f"[setup] ground_truth={OBS_NAME}, models_available={sorted(forecast_ds.keys())}")

        # Choose location: explicit lat/lon > region center > equator default
        if lat is not None and lon is not None:
            lat_used = float(observations.latitude.sel(latitude=float(lat), method="nearest"))
            lon_used = float(observations.longitude.sel(longitude=float(lon),  method="nearest"))
            loc_tag = f"({_fmt_lat(lat_used)}, {_fmt_lon(lon_used)})"
        elif region is not None:
            lat_used, lon_used = _pick_point_in_region(observations, region)
            loc_tag = f"{region} ({_fmt_lat(lat_used)}, {_fmt_lon(lon_used)})"
        else:
            lat_used = float(observations.latitude.sel(latitude=0.0, method="nearest"))
            lon_used = float(observations.longitude.sel(longitude=0.0, method="nearest"))
            loc_tag = f"({_fmt_lat(lat_used)}, {_fmt_lon(lon_used)})"
        dbg(f"[location] using {loc_tag}")

        # Choose best-available token per model base
        chosen_by_base = {}
        for base in MODEL_ORDER:
            opts = [base, f"{base}_operational"]
            chosen = opts[-1] if opts[-1] in model_tokens else opts[0]
            chosen_by_base[base] = chosen
        dbg(f"[models] chosen per base: {chosen_by_base}")

        # Figure scaffold
        vars_order = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]
        titles = [VAR_DISPLAY[v] for v in vars_order]
        nrows, ncols = len(lead_times), len(vars_order)
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), dpi=140, sharex=False)
        if nrows == 1: axes = np.expand_dims(axes, 0)
        if ncols == 1: axes = np.expand_dims(axes, 1)
        for j, ttl in enumerate(titles): axes[0, j].set_title(ttl, fontsize=14)

        handles_all, labels_all = [], []
        p_tail = max(0.0, min(1.0, pct_extremes / 100.0))

        for i, lead in enumerate(lead_times):
            lead = int(lead)
            lead_td = np.timedelta64(lead, "D")
            for j, var in enumerate(vars_order):
                ax = axes[i, j]
                ylab = "Brier score" if score_type.upper() == "BS" else "Quantile score"

                drawn = 0
                for base in MODEL_ORDER:
                    mdl = chosen_by_base[base]
                    if mdl not in forecast_ds:
                        dbg(f"[skip] lead={lead}d var={var} model={mdl}: zarr missing")
                        continue
                    try:
                        preds = forecast_ds[mdl].sel(
                            prediction_timedelta=lead_td,
                            latitude=lat_used, longitude=lon_used, method="nearest"
                        ).sel(time=TIME_RANGE)[var].load()
                        valid_time = preds.time + lead_td
                        obs = observations.sel(latitude=lat_used, longitude=lon_used, time=valid_time)[var].load()

                        n_pred = int(preds.sizes.get("time", preds.size))
                        n_obs  = int(obs.sizes.get("time", obs.size))
                        if n_pred == 0 or n_obs == 0:
                            dbg(f"[empty] lead={lead}d var={VAR_DISPLAY.get(var,var)}: n_pred={n_pred}, n_obs={n_obs}")
                            continue

                        x = preds.values; y = obs.values
                        fit = idr(y, pd.DataFrame({"x": x}))
                        pp  = fit.predict(pd.DataFrame({"x": x}))

                        if score_type.upper() == "BS":
                            # Build z-grid internally from central quantile range to avoid outliers.
                            z_lo, z_hi = np.nanquantile(y, [0.005, 0.995])
                            if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_lo == z_hi:
                                z_lo, z_hi = float(np.nanmin(y)), float(np.nanmax(y))
                            z_grid = np.linspace(z_lo, z_hi, num=len(Q_GRID))
                            curve  = _bs_curve_from_idr_zgrid(pp, y, z_grid)
                            xvals  = z_grid
                            unit   = var_units.get(var, "")
                            unit_txt = f" [{unit}]" if unit else ""
                            xlab   = f"z{unit_txt}"
                        else:
                            curve = _qs_curve_from_idr(pp, y, Q_GRID)
                            xvals = Q_GRID
                            xlab  = "Quantile"

                        if not np.any(np.isfinite(curve)):
                            dbg(f"[nan-curve] lead={lead}d var={var} model={mdl}")
                            continue

                        lbl = display_model_name(mdl, OBS_NAME)
                        h = ax.plot(xvals, curve, marker="o", markersize=3.0, linewidth=1.4, label=lbl)[0]
                        handles_all.append(h); labels_all.append(lbl)
                        drawn += 1

                    except Exception as e:
                        dbg(f"[error] lead={lead}d var={var} model={mdl}: {e}")
                        if debug: traceback.print_exc(limit=1)
                        continue

                # Axis labels, ticks, grid
                if j == 0:
                    ax.set_ylabel(f"{ylab}\nLead Time={lead} days", fontsize=12)
                ax.set_xlabel(xlab, fontsize=12)
                ax.grid(True, alpha=0.3)

                # Numeric x ticks for BS; standard for QS
                if score_type.upper() == "BS":
                    ax.locator_params(axis="x", nbins=6)
                    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

                # Shade extreme regions
                if score_type.upper() == "QS":
                    # QS: shade by quantiles directly
                    if var in ("mean_sea_level_pressure", "2m_temperature"):
                        if p_tail > 0: ax.axvspan(0.0, p_tail, alpha=0.10, color="gray")
                    if p_tail > 0: ax.axvspan(1.0 - p_tail, 1.0, alpha=0.10, color="gray")
                else:
                    # BS: map quantile bands to z via observed y at this point
                    if var in ("mean_sea_level_pressure", "2m_temperature"):
                        zl, zr = np.nanquantile(y, [p_tail, 1.0 - p_tail])
                        x0, x1 = ax.get_xlim()
                        if np.isfinite(zl): ax.axvspan(x0, zl, alpha=0.10, color="gray")
                        if np.isfinite(zr): ax.axvspan(zr, x1, alpha=0.10, color="gray")
                    else:
                        zr = float(np.nanquantile(y, 1.0 - p_tail))
                        x0, x1 = ax.get_xlim()
                        if np.isfinite(zr): ax.axvspan(zr, x1, alpha=0.10, color="gray")

        # Legend: enforce desired order and add shading explanation
        want_order = (["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME == "era5"
                      else ["GC-IFS", "PW-IFS", "HRES"])
        uniq, uniq_handles = [], []
        for want in want_order:
            for h, l in zip(handles_all, labels_all):
                if l == want and l not in uniq:
                    uniq.append(l); uniq_handles.append(h); break

        shade_label = (
            f"Grey bands: extreme regions (p={pct_extremes:.0f}%). "
            "MSLP/T2M: both tails; WS10: upper tail"
        )
        shade_patch = Patch(facecolor="gray", alpha=0.10, edgecolor="none", label=shade_label)

        fig.legend(
            uniq_handles + [shade_patch],
            uniq + [shade_label],
            ncol=min(len(uniq_handles) + 1, 6),
            loc="lower center", bbox_to_anchor=(0.5, 0.01),
            frameon=False, fontsize=12
        )

        # Title and layout
        head = "Brier score" if score_type.upper() == "BS" else "Quantile score"
        fig.suptitle(f"CRPS decomposition via {head} - {OBS_NAME.upper()} as Ground Truth\n{loc_tag}",
                     fontsize=19, y=0.985)
        fig.tight_layout(rect=[0, 0.05, 1, 0.97])

        # Save
        tag = "bs" if score_type.upper() == "BS" else "qs"
        out_png = (
            f"sensitivity_{tag}_5x3_singlepoint_{OBS_NAME}_"
            f"{'region_' + region.replace(' ', '_').lower() if region and (lat is None and lon is None) else 'latlon'}.png"
        )
        os.makedirs(SAVE_DIR, exist_ok=True)
        fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
        plt.close(fig)
        dbg(f"[saved] {os.path.join(SAVE_DIR, out_png)}")

        return float(lat_used), float(lon_used)

    finally:
        OBS_NAME = prev_obs

        

def plot_interpretation_grid_small_zoom(qmin: float = 0.0, qmax: float = 0.15):
    """
    Tail-zoom (lower quantiles) version of the small-extremes interpretation grid.
      - Layout: 4x2 (same as plot_interpretation_grid_small)
      - Columns: [MSLP – low extremes, T2M – cold extremes]
      - Rows:    [0] TW_PC (+ small ref), [1] TW_PCS,
                 [2] QW_PC (+ small ref), [3] QW_PCS
    PCS rows share a single global y-scale across both columns; PC rows have per-column scales.
    Only quantiles in [qmin, qmax] are displayed; defaults zoom to the first ~15%.
    """
    # Clamp and sanity-check zoom window
    qmin = float(np.clip(qmin, 0.0, 1.0))
    qmax = float(np.clip(qmax, 0.0, 1.0))
    if qmax <= qmin:
        qmin, qmax = 0.0, 0.15  # fallback to default window

    var_order  = ["mean_sea_level_pressure", "2m_temperature"]
    col_titles = ["MSLP - low extremes", "T2M - cold extremes"]

    # Ensure inputs and small-reference curves exist
    for var in var_order:
        ensure_csv_exists_small("TW_PC",  var, LEAD_DAYS)
        ensure_csv_exists_small("TW_PCS", var, LEAD_DAYS)
        ensure_csv_exists_small("QW_PC",  var, LEAD_DAYS)
        ensure_csv_exists_small("QW_PCS", var, LEAD_DAYS)
        _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "TW")
        _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "QW")

    fig, axes = plt.subplots(4, 2, figsize=(13, 12.5), dpi=140, sharex="col")

    # Column titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=14)

    # Row labels
    axes[0, 0].set_ylabel("TW_PC", fontsize=14)
    axes[1, 0].set_ylabel("TW_PCS", fontsize=14)
    axes[2, 0].set_ylabel("QW_PC", fontsize=14)
    axes[3, 0].set_ylabel("QW_PCS", fontsize=14)

    handles_all, labels_all = [], []

    # Per-column ranges for PC rows
    pc_min_col = np.full(2, np.inf)
    pc_max_col = np.full(2, -np.inf)

    # Single global PCS range shared by TW_PCS and QW_PCS
    pcs_min_global, pcs_max_global = np.inf, -np.inf

    def _pad(lo: float, hi: float) -> tuple[float, float]:
        """Symmetric y-padding using Y_PAD_FRAC; robust to degenerate ranges."""
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return 0.0, 1.0
        rng = hi - lo
        pad = Y_PAD_FRAC * (rng if (rng > 0 and np.isfinite(rng))
                            else (abs(hi) if (np.isfinite(hi) and hi != 0) else 1.0))
        return lo - pad, hi + pad

    def _plot_family_small_zoom(ax_pc, ax_pcs, var: str, family: str, col_idx: int):
        """Plot one family (TW/QW) for a given variable in the zoomed quantile window."""
        nonlocal pcs_min_global, pcs_max_global, pc_min_col, pc_max_col

        if family == "TW":
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "TW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "TW_PCS")))
            pc_col, pcs_col = "tw_pc_small", "tw_pcs_small"
            ref_path = _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "TW"); ref_col = "tw_pc0_small_ref"
        else:
            df_pc  = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "QW_PC")))
            df_pcs = pd.read_csv(os.path.join(SAVE_DIR, csv_filename_for_small(var, LEAD_DAYS, OBS_NAME, "QW_PCS")))
            pc_col, pcs_col = "qw_pc_small", "qw_pcs_small"
            ref_path = _ensure_ref_pc0_small_csv(var, LEAD_DAYS, "QW"); ref_col = "qw_pc0_small_ref"

        # Restrict to zoom window and sort consistently
        df_pc  = df_pc[(df_pc["quantile"] >= qmin) & (df_pc["quantile"] <= qmax)].sort_values(["model", "quantile"])
        df_pcs = df_pcs[(df_pcs["quantile"] >= qmin) & (df_pcs["quantile"] <= qmax)].sort_values(["model", "quantile"])

        y_min_pc, y_max_pc = np.inf, -np.inf
        y_min_pcs, y_max_pcs = np.inf, -np.inf

        # Plot models in canonical order, folding *_operational when present
        for base in MODEL_ORDER:
            options = [base, f"{base}_operational"]
            avail_pc = df_pc[df_pc["model"].isin(options)]
            if avail_pc.empty:
                continue
            actual_key = options[-1] if (options[-1] in avail_pc["model"].unique()) else options[0]
            label = display_model_name(actual_key, OBS_NAME)

            seg_pc = df_pc[df_pc["model"] == actual_key]
            h = ax_pc.plot(seg_pc["quantile"], seg_pc[pc_col],
                           marker="o", markersize=3.5, linestyle="-", label=label)[0]
            handles_all.append(h); labels_all.append(label)
            y_min_pc = min(y_min_pc, seg_pc[pc_col].min()); y_max_pc = max(y_max_pc, seg_pc[pc_col].max())

            seg_pcs = df_pcs[df_pcs["model"] == actual_key]
            if not seg_pcs.empty:
                ax_pcs.plot(seg_pcs["quantile"], seg_pcs[pcs_col],
                            marker="o", markersize=3.5, linestyle="-")
                y_min_pcs = min(y_min_pcs, seg_pcs[pcs_col].min()); y_max_pcs = max(y_max_pcs, seg_pcs[pcs_col].max())

        # Small-tail reference PC^(0) on PC row (also clipped to zoom window)
        dfr = pd.read_csv(ref_path).sort_values("quantile")
        dfr = dfr[(dfr["quantile"] >= qmin) & (dfr["quantile"] <= qmax)]
        if not dfr.empty:
            h_ref = ax_pc.plot(dfr["quantile"], dfr[ref_col],
                               linestyle="--", linewidth=2.2, color="black", label="(tw / qw) PC⁽⁰⁾")[0]
            handles_all.append(h_ref); labels_all.append("(tw / qw) PC⁽⁰⁾")
            y_min_pc = min(y_min_pc, dfr[ref_col].min()); y_max_pc = max(y_max_pc, dfr[ref_col].max())

        # Cosmetics and limits
        ax_pc.grid(True, alpha=0.3); ax_pcs.grid(True, alpha=0.3)
        ax_pc.set_xlabel(xaxis_label_for(f"{family}_PC"),  fontsize=14)
        ax_pcs.set_xlabel(xaxis_label_for(f"{family}_PCS"), fontsize=14)
        ax_pc.tick_params(axis="both", labelsize=14); ax_pcs.tick_params(axis="both", labelsize=14)
        ax_pc.set_xlim(qmin, qmax); ax_pcs.set_xlim(qmin, qmax)

        # Update tracked ranges
        if np.isfinite(y_min_pc) and np.isfinite(y_max_pc):
            pc_min_col[col_idx] = min(pc_min_col[col_idx], y_min_pc)
            pc_max_col[col_idx] = max(pc_max_col[col_idx], y_max_pc)
        if np.isfinite(y_min_pcs) and np.isfinite(y_max_pcs):
            pcs_min_global = min(pcs_min_global, y_min_pcs)
            pcs_max_global = max(pcs_max_global, y_max_pcs)

    # Fill all panels
    for j, var in enumerate(var_order):
        _plot_family_small_zoom(axes[0, j], axes[1, j], var, "TW", col_idx=j)
    for j, var in enumerate(var_order):
        _plot_family_small_zoom(axes[2, j], axes[3, j], var, "QW", col_idx=j)

    # Apply y-limits: PC rows per column
    for j in range(2):
        lo, hi = _pad(pc_min_col[j], pc_max_col[j])
        for row in (0, 2):
            axes[row, j].set_ylim(lo, hi)

    # Apply y-limits: single global scale for both PCS rows
    lo_pcs, hi_pcs = _pad(pcs_min_global, pcs_max_global)
    for j in range(2):
        axes[1, j].set_ylim(lo_pcs, hi_pcs)  # TW_PCS
        axes[3, j].set_ylim(lo_pcs, hi_pcs)  # QW_PCS

    # Legend ordering (models + reference)
    want_order = (["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME.lower() == "era5"
                  else ["GC-IFS", "PW-IFS", "HRES"]) + ["(tw / qw) PC⁽⁰⁾"]
    uniq, uniq_handles = [], []
    for want in want_order:
        for h, l in zip(handles_all, labels_all):
            if l == want and l not in uniq:
                uniq.append(l); uniq_handles.append(h); break

    fig.suptitle(
        f"Sensitivity to Quantile Levels – small extremes (zoom q ∈ [{qmin:.2f}, {qmax:.2f}])"
        f"\n({OBS_NAME.upper()}, Lead Time={LEAD_DAYS} days)",
        fontsize=20, y=0.96
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.legend(uniq_handles, uniq, ncol=min(len(uniq), 6),
               loc="lower center", bbox_to_anchor=(0.5, -0.005), frameon=False, fontsize=14)

    out_png = f"sensitivity_interpretation_small_tail_q{int(qmin*100)}-{int(qmax*100)}_4x2_lead{LEAD_DAYS}d_{OBS_NAME}.png"
    os.makedirs(SAVE_DIR, exist_ok=True)
    fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
    plt.close(fig)
    print(f"saved plot -> {os.path.join(SAVE_DIR, out_png)}")


        
def plot_scores_gridpoint_fixed_lead(
    lead_days: int = 5,
    ground_truth: str = "ifs",
    pct_extremes: float = 5.0,
    lat: float | None = None,
    lon: float | None = None,
    region: str | None = None,
    debug: bool = False,
):
    """
    2×3 panel at a single grid point and fixed lead time:
      - Cols: [MSLP, T2M, WS10]
      - Row 1: Brier-score curve PS(F(z), 1{Y ≤ z}) vs *physical threshold* z
      - Row 2: Quantile-score curve QS_α = 2(1{y ≤ F^{-1}(α)} − α)(F^{-1}(α) − y) vs α

    Notes
    -----
    Requires globals already present in the module:
      OBS_NAME, TIME_RANGE, MODEL_ORDER, VAR_DISPLAY, Q_GRID, SAVE_DIR,
      display_model_name(), _open_data(), and imports: numpy as np, pandas as pd,
      xarray as xr, matplotlib.pyplot as plt, idr from isodisreg.
    """
    from matplotlib.patches import Patch

    # Variable → unit for axis labels (override globally via VAR_UNITS if desired)
    default_var_units = {
        "mean_sea_level_pressure": "Pa",   # use "hPa" if values are converted before scoring
        "2m_temperature":          "K",
        "10m_wind_speed":          "m s⁻¹",
    }
    var_units = globals().get("VAR_UNITS", default_var_units)

    def dbg(*a, **k):
        if debug: print(*a, **k)

    # ---------------- Region helpers (self-contained) ----------------
    REGION_DEFS = {
        "nhextratropics":   {"lat_cond": ("ge",  20.0)},
        "shextratropics":   {"lat_cond": ("le", -20.0)},
        "tropics":          {"lat_range": (-20.0, 20.0)},
        "extratropics":     {"abs_lat_ge": 20.0},
        "arctic":           {"lat_cond": ("ge",  60.0)},
        "antarctic":        {"lat_cond": ("le", -60.0)},
        "europe":           {"lat_range": (35.0, 75.0),   "lon_range": (-12.5,  42.5)},
        "north_america":    {"lat_range": (25.0, 60.0),   "lon_range": (-120.0, -75.0)},
        "north_atlantic":   {"lat_range": (25.0, 60.0),   "lon_range": (-70.0,  -20.0)},
        "north_pacific":    {"lat_range": (25.0, 60.0),   "lon_range": (145.0, -130.0)},  # wraps dateline
        "east_asia":        {"lat_range": (25.0, 60.0),   "lon_range": (102.5, 150.0)},
        "ausnz":            {"lat_range": (-45.0, -12.5), "lon_range": (120.0, 175.0)},
    }
    REGION_ALIASES = {
        "nh": "nhextratropics", "sh": "shextratropics", "et": "extratropics",
        "arct": "arctic", "ant": "antarctic", "eu": "europe", "na": "north_america",
        "natl": "north_atlantic", "npac": "north_pacific", "easia": "east_asia",
    }
    def _to_pm180(lon_vals: np.ndarray) -> np.ndarray:
        return ((np.asarray(lon_vals) + 180.0) % 360.0) - 180.0
    def _mask_lon_range(lon_pm180: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if hi >= lo: return (lon_pm180 >= lo) & (lon_pm180 <= hi)
        return (lon_pm180 >= lo) | (lon_pm180 <= hi)
    def _pick_point_in_region(ds: xr.Dataset, name: str) -> tuple[float, float]:
        key = name.strip().lower().replace(" ", "_")
        key = REGION_ALIASES.get(key, key)
        spec = REGION_DEFS.get(key)
        if spec is None:
            raise ValueError(f"Unknown region '{name}'. Allowed: {sorted(REGION_DEFS)}")
        lats = np.asarray(ds.latitude); lons_native = np.asarray(ds.longitude); lons_pm180 = _to_pm180(lons_native)
        lat_mask = np.ones_like(lats, dtype=bool)
        if "lat_range" in spec:
            lo, hi = spec["lat_range"]; lat_mask &= (lats >= lo) & (lats <= hi)
        if "lat_cond" in spec:
            op, v = spec["lat_cond"]; lat_mask &= (lats >= v) if op == "ge" else (lats <= v)
        if "abs_lat_ge" in spec:
            lat_mask &= (np.abs(lats) >= float(spec["abs_lat_ge"]))
        lon_mask = np.ones_like(lons_pm180, dtype=bool)
        if "lon_range" in spec:
            lo, hi = spec["lon_range"]; lon_mask = _mask_lon_range(lons_pm180, float(lo), float(hi))
        mask2d = np.outer(lat_mask, lon_mask)
        if not np.any(mask2d): raise RuntimeError(f"No grid cells inside region '{name}'.")
        lat_grid, lon_grid = np.meshgrid(lats, lons_pm180, indexing="ij")
        lat_sel = float(np.mean(lat_grid[mask2d])); lon_sel_pm180 = float(np.mean(lon_grid[mask2d]))
        lat_snap = float(ds.latitude.sel(latitude=lat_sel, method="nearest"))
        lon_for_snap = lon_sel_pm180 % 360.0 if (lons_native.min() >= 0 and lons_native.max() <= 360) else lon_sel_pm180
        lon_snap = float(ds.longitude.sel(longitude=lon_for_snap, method="nearest"))
        return lat_snap, lon_snap
    def _fmt_lat(latv: float) -> str: return f"{abs(latv):.2f}°{'N' if latv >= 0 else 'S'}"
    def _fmt_lon(lonv: float) -> str:
        ll = ((lonv + 180) % 360) - 180
        return f"{abs(ll):.2f}°{'E' if ll >= 0 else 'W'}"

    # ---------------- Scoring helpers (exact definitions) ----------------
    def _cdf_right_at(x: np.ndarray, cdf: np.ndarray, z: float) -> float:
        """Right-limit ECDF value F(z) (ties included)."""
        x = np.asarray(x, dtype=float); cdf = np.asarray(cdf, dtype=float); n = x.size
        if n == 0: return np.nan
        k = int(np.searchsorted(x, z, side="right"))  # first index where x > z
        Fz = 0.0 if k == 0 else float(cdf[min(k - 1, n - 1)])
        return float(min(1.0, max(0.0, Fz)))

    def _ecdf_quantile_left_inverse(x: np.ndarray, cdf: np.ndarray, alpha: float) -> float:
        """Left inverse F^{-1}(alpha) = inf{z: F(z) >= alpha}."""
        x = np.asarray(x, dtype=float); cdf = np.asarray(cdf, dtype=float); n = x.size
        if n == 0: return np.nan
        k = int(np.searchsorted(cdf, float(alpha), side="left"))
        return float(x[min(max(k, 0), n - 1)])

    def _bs_curve_from_idr_zgrid(pp, y: np.ndarray, z_grid: np.ndarray) -> np.ndarray:
        """
        Brier-score curve on a direct threshold grid z_grid for the event {Y ≤ z}.
        Implements PS(F(z), 1{Y≤z}) with F taken as the RIGHT limit of the ECDF.
        """
        y = np.asarray(y, dtype=float)
        X = [np.asarray(pr.points, dtype=float) for pr in pp.predictions]
        F = [np.asarray(pr.ecdf,   dtype=float) for pr in pp.predictions]
        out = np.empty(len(z_grid), dtype=float)
        for k, zk in enumerate(np.asarray(z_grid, dtype=float)):
            event = (y <= zk).astype(float)
            prob  = np.array([_cdf_right_at(X[i], F[i], zk) for i in range(y.size)], dtype=float)
            out[k] = float(np.mean((prob - event) ** 2))
        return out

    def _qs_curve_from_idr(pp, y: np.ndarray, q_grid: np.ndarray) -> np.ndarray:
        """QS curve: 2*(1{y ≤ F^{-1}(α)} − α)*(F^{-1}(α) − y)."""
        y = np.asarray(y, dtype=float)
        X = [np.asarray(pred.points, dtype=float) for pred in pp.predictions]
        F = [np.asarray(pred.ecdf,   dtype=float) for pred in pp.predictions]
        out = np.empty(len(q_grid), dtype=float)
        for k, alpha in enumerate(q_grid):
            qhat = np.array([_ecdf_quantile_left_inverse(X[i], F[i], float(alpha))
                             for i in range(y.size)], dtype=float)
            event = (y <= qhat).astype(float)
            out[k] = float(2.0 * np.mean((event - float(alpha)) * (qhat - y)))
        return out

    # ---------------- Data selection and figure ----------------
    assert lead_days >= 0
    global OBS_NAME
    prev_obs = OBS_NAME
    try:
        OBS_NAME = ground_truth.strip().lower()
        observations, forecast_ds, model_tokens = _open_data()
        dbg(f"[setup] truth={OBS_NAME}, models={sorted(forecast_ds.keys())}")

        # Grid point: explicit lat/lon > region center > equator default
        if lat is not None and lon is not None:
            lat_used = float(observations.latitude.sel(latitude=float(lat), method="nearest"))
            lon_used = float(observations.longitude.sel(longitude=float(lon),  method="nearest"))
            loc_tag = f"({_fmt_lat(lat_used)}, {_fmt_lon(lon_used)})"
        elif region is not None:
            lat_used, lon_used = _pick_point_in_region(observations, region)
            loc_tag = f"{region} ({_fmt_lat(lat_used)}, {_fmt_lon(lon_used)})"
        else:
            lat_used = float(observations.latitude.sel(latitude=0.0, method="nearest"))
            lon_used = float(observations.longitude.sel(longitude=0.0, method="nearest"))
            loc_tag = f"({_fmt_lat(lat_used)}, {_fmt_lon(lon_used)})"
        dbg(f"[location] {loc_tag}")

        # Concrete model token per base
        chosen_by_base = {}
        for base in MODEL_ORDER:
            opts = [base, f"{base}_operational"]
            chosen_by_base[base] = (opts[-1] if opts[-1] in model_tokens else opts[0])
        dbg(f"[models] chosen={chosen_by_base}")

        # Figure scaffold
        var_order = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]
        titles = [VAR_DISPLAY[v] for v in var_order]
        fig, axes = plt.subplots(2, 3, figsize=(14, 9), dpi=140, sharex=False)
        for j, ttl in enumerate(titles): axes[0, j].set_title(ttl, fontsize=14)

        handles_all, labels_all = [], []
        p_tail = max(0.0, min(1.0, pct_extremes / 100.0))
        lead_td = np.timedelta64(int(lead_days), "D")

        for j, var in enumerate(var_order):
            ax_bs = axes[0, j]  # BS vs z
            ax_qs = axes[1, j]  # QS vs quantile

            for base in MODEL_ORDER:
                mdl = chosen_by_base[base]
                if mdl not in forecast_ds:
                    dbg(f"[skip] var={var} model={mdl} (no zarr)")
                    continue
                try:
                    preds = forecast_ds[mdl].sel(
                        prediction_timedelta=lead_td,
                        latitude=lat_used, longitude=lon_used, method="nearest"
                    ).sel(time=TIME_RANGE)[var].load()
                    valid_time = preds.time + lead_td
                    obs = observations.sel(latitude=lat_used, longitude=lon_used, time=valid_time)[var].load()

                    x, y = preds.values, obs.values
                    if x.size == 0 or y.size == 0:
                        dbg(f"[empty] var={var} model={mdl}")
                        continue

                    fit = idr(y, pd.DataFrame({"x": x}))
                    pp  = fit.predict(pd.DataFrame({"x": x}))

                    # --- BS: build a z-grid from central quantile range to avoid extreme outliers
                    z_lo, z_hi = np.nanquantile(y, [0.005, 0.995])
                    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_lo == z_hi:
                        z_lo, z_hi = float(np.nanmin(y)), float(np.nanmax(y))
                    z_grid = np.linspace(z_lo, z_hi, num=len(Q_GRID))
                    bs_curve = _bs_curve_from_idr_zgrid(pp, y, z_grid)

                    # --- QS on standard quantile grid
                    qs_curve = _qs_curve_from_idr(pp, y, Q_GRID)

                    # Plot curves
                    lbl = display_model_name(mdl, OBS_NAME)
                    h1 = ax_bs.plot(z_grid, bs_curve, marker="o", markersize=3.0, linewidth=1.4, label=lbl)[0]
                    ax_qs.plot(Q_GRID, qs_curve, marker="o", markersize=3.0, linewidth=1.4, label=lbl)
                    handles_all.append(h1); labels_all.append(lbl)

                    if debug:
                        dbg(f"[curve] var={VAR_DISPLAY.get(var,var)} model={mdl}: "
                            f"BS med={np.nanmedian(bs_curve):.4g}, QS med={np.nanmedian(qs_curve):.4g}")

                except Exception as e:
                    dbg(f"[error] var={var} model={mdl}: {e}")
                    continue

            # Axis labels, ticks, grid
            unit = var_units.get(var, "")
            ax_bs.set_ylabel("Brier score", fontsize=12)
            ax_qs.set_ylabel("Quantile score", fontsize=12)
            ax_bs.set_xlabel(f"z{' [' + unit + ']' if unit else ''}", fontsize=12)
            ax_qs.set_xlabel("Quantile", fontsize=12)
            ax_bs.grid(True, alpha=0.3); ax_qs.grid(True, alpha=0.3)

            # Numeric x ticks for BS
            ax_bs.locator_params(axis="x", nbins=6)
            ax_bs.ticklabel_format(axis="x", style="plain", useOffset=False)

            # Shading: map quantile tails → z for BS; direct quantiles for QS
            if var in ("mean_sea_level_pressure", "2m_temperature"):
                # Two-sided tails for BS
                zl, zr = np.nanquantile(y, [p_tail, 1.0 - p_tail])
                x0, x1 = ax_bs.get_xlim()
                if np.isfinite(zl): ax_bs.axvspan(x0, zl, alpha=0.10, color="gray")
                if np.isfinite(zr): ax_bs.axvspan(zr, x1, alpha=0.10, color="gray")
                # Two-sided quantile shading for QS
                if p_tail > 0: ax_qs.axvspan(0.0, p_tail, alpha=0.10, color="gray")
                if p_tail > 0: ax_qs.axvspan(1.0 - p_tail, 1.0, alpha=0.10, color="gray")
            else:
                # WS10: right tail only
                zr = float(np.nanquantile(y, 1.0 - p_tail))
                x0, x1 = ax_bs.get_xlim()
                if np.isfinite(zr): ax_bs.axvspan(zr, x1, alpha=0.10, color="gray")
                if p_tail > 0: ax_qs.axvspan(1.0 - p_tail, 1.0, alpha=0.10, color="gray")

        # Legend (models in desired order) + shading explanation
        want_order = (["GC-ERA5", "PW-ERA5", "HRES"] if OBS_NAME == "era5" else ["GC-IFS", "PW-IFS", "HRES"])
        uniq, uniq_handles = [], []
        for want in want_order:
            for h, l in zip(handles_all, labels_all):
                if l == want and l not in uniq:
                    uniq.append(l); uniq_handles.append(h); break
        shade_label = (
            f"Grey bands: extreme regions (p={pct_extremes:.0f}%). "
            "MSLP/T2M: both tails; WS10: upper tail"
        )
        shade_patch = Patch(facecolor="gray", alpha=0.10, edgecolor="none", label=shade_label)

        fig.legend(
            uniq_handles + [shade_patch],
            uniq + [shade_label],
            ncol=min(len(uniq_handles) + 1, 6),
            loc="lower center", bbox_to_anchor=(0.5, 0.01),
            frameon=False, fontsize=12
        )

        # Title, layout, save
        fig.suptitle(
            f"CRPS decomposition via Brier & Quantile scores\n"
            f"at Lead Time={lead_days} days in {loc_tag} - {OBS_NAME.upper()} as Ground Truth",
            fontsize=18, y=0.96
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])

        tag = (f"region_{region.replace(' ', '_').lower()}" if region and (lat is None and lon is None)
               else "latlon")
        out_png = f"sensitivity_bothscores_2x3_lead{lead_days}d_{OBS_NAME}_{tag}.png"
        os.makedirs(SAVE_DIR, exist_ok=True)
        fig.savefig(os.path.join(SAVE_DIR, out_png), bbox_inches="tight")
        plt.close(fig)
        dbg(f"[saved] {os.path.join(SAVE_DIR, out_png)}")

        return float(lat_used), float(lon_used)

    finally:
        OBS_NAME = prev_obs
        
        

# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------



if __name__ == "__main__":
    # =============================== SWITCHES ================================
    # Panels / figures
    RUN_QUANTILE_GRID                   = False   # plot_quantile_grid(SCORE)
    RUN_INTERPRETATION_GRID             = False   # Sensitivity to quantile levels (high extremes)
    RUN_INTERPRETATION_GRID_ZOOM_LAST10 = False   # Zoom in sensitivity to quantile levels (high extremes)
    RUN_INTERPRETATION_GRID_SMALL       = False   # Sensitivity to quantile levels (small extremes)
    RUN_INTERPRETATION_GRID_SMALL_ZOOM  = False   # Zoom in sensitivity to quantile levels (small extremes)

    RUN_BS_QS_GRIDPOINT                 = False   # Brier OR quantile scores at single point for all lead times
    RUN_BOTH_SCORES_GRIDPOINT_FIXED_LEAD     = False   # Brier & quantile scores at single point for fixed lead time

    # =============================== THESIS KNOBS ============================
    # Common across all plots (data, paths, models, grids, cosmetics)

    # Quantile grids used by builders/plots
    N_POINTS     = 50
    Q_GRID       = np.linspace(0.0, 0.99, N_POINTS).astype(float)
    Q_GRID_SMALL = np.linspace(0.01, 1.00, N_POINTS).astype(float)

    # Runtime knobs with safe defaults (overridden in __main__)
    OBS_NAME  = "ifs"
    LEAD_DAYS = 5

    SAVE_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    os.makedirs(SAVE_DIR, exist_ok=True)


    # ========================== PER-FIGURE PARAMETERS ========================
    # Quantile grid panel
    SCORE = "QW_PCS"  # one of {"TW_PCS","QW_PCS","TW_PC","QW_PC"}

    # Single-point BS/QS (5x3)
    SP_SCORE_TYPE   = "QS"        # "BS" or "QS"
    SP_GROUND_TRUTH = "era5"      # {"era5","ifs"} (overrides OBS_NAME for this plot)
    SP_LAT, SP_LON  = None, None  # explicit location OR region:
    SP_REGION       = "Europe"    # e.g. "Europe","tropics","nhextratropics",...
    SP_PCT_EXTREMES = 10.0        # shade ±p% tails (MSLP/T2M both tails; WS10 upper)
    SP_LEAD_TIMES   = (1, 3, 5, 7, 10)
    SP_DEBUG        = False

    # Single-point, fixed lead (2x3, BS & QS)
    FL_LEAD_DAYS    = 5
    FL_GROUND_TRUTH = "era5"
    FL_LAT, FL_LON  = None, None
    FL_REGION       = "Europe"
    FL_PCT_EXTREMES = 10.0
    FL_DEBUG        = False

    # ================================ RUN ====================================
    if RUN_QUANTILE_GRID:
        plot_quantile_grid(SCORE)

    if RUN_INTERPRETATION_GRID:
        plot_interpretation_grid()

    if RUN_INTERPRETATION_GRID_SMALL:
        plot_interpretation_grid_small()

    if RUN_INTERPRETATION_GRID_SMALL_ZOOM:
        # example zoom: first ~15%
        plot_interpretation_grid_small_zoom(qmin=0.00, qmax=0.15)

    if RUN_INTERPRETATION_GRID_ZOOM_LAST10:
        # example zoom: last 10% (upper tail)
        plot_interpretation_grid_zoom_last10(qmin=0.90, qmax=1.00)

    if RUN_BS_QS_GRIDPOINT:
        plot_bs_qs_gridpoint(
            score_type=SP_SCORE_TYPE,
            ground_truth=SP_GROUND_TRUTH,
            pct_extremes=SP_PCT_EXTREMES,
            lat=SP_LAT, lon=SP_LON,
            region=SP_REGION,
            lead_times=SP_LEAD_TIMES,
            debug=SP_DEBUG,
        )

    if RUN_BOTH_SCORES_GRIDPOINT_FIXED_LEAD:
        plot_scores_gridpoint_fixed_lead(
            lead_days=FL_LEAD_DAYS,
            ground_truth=FL_GROUND_TRUTH,
            lat=FL_LAT, lon=FL_LON,
            region=FL_REGION,
            pct_extremes=FL_PCT_EXTREMES,
            debug=FL_DEBUG,
        )
