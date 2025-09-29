import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from weatherbench2.metrics import _spatial_average
from isodisreg import idr
from joblib import Parallel, delayed
import sys, os
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(__file__))
from metric_functions import (
    pc, pcs,
    tw_crps, tw_crps_small,   
    tw_pc, tw_pc_small,
    qw_crps, qw_crps_small,   
    qw_pc, qw_pc_small,
    qw_pc0, qw_pc0_small,  
    qw_pcs, qw_pcs_small
)
# --------------------------- Variables/ Definitions --------------------------------

SCORE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'score_data')
     
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

OBS_SOURCES    = ['era5', 'ifs']
VARIABLES      = ['2m_temperature', 'mean_sea_level_pressure', '10m_wind_speed']
LEAD_TIMES     = [1, 3, 5, 7, 10]
SCORE_COLS     = ['pc', 'pcs', 'tw_pc', 'tw_pcs', 'qw_pc', 'qw_pcs']

NAME_MAP = {
    'mean_sea_level_pressure': 'MSLP',
    '2m_temperature':           'T2M',
    '10m_wind_speed':           'WS10',
}
BASE_MODEL_TITLES = {
    'hres':      'IFS‑HRES',
    'pangu':     'Pangu',
    'graphcast': 'GraphCast',
}

BASE_DIR  = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE_DIR, 'data')
CLIM_FORECAST_PATH = os.path.join(DATA_DIR, "era5_climatology_forecasts.zarr")
ERA5_OBS_PATH      = os.path.join(DATA_DIR, "era5_64x32.zarr")
EVAL_PERIOD        = slice("2020-01-01", "2020-12-31")


REF_PC_DIR = os.path.join(SCORE_DATA_DIR, "era5_ref_pc")
os.makedirs(REF_PC_DIR, exist_ok=True)

# --------------------------- computation functions -------------------------
def compute_lat_weighted_mean(df: pd.DataFrame, metric: str) -> float:
    """
    Given a DataFrame with columns ['lat','lon',metric],
    collapse any duplicate (lat,lon) by averaging, then
    return the cosine‑latitude weighted mean via WB2’s _spatial_average.
    """
    # aggregate duplicates (e.g. multiple time‐steps at same grid point, esp. relevant for PC/ PCS)
    grid = df.pivot_table(
        index="lat",
        columns="lon",
        values=metric,
        aggfunc="mean",
    )

    # turn into an xarray DataArray for spatial averaging
    da = xr.DataArray(
        grid.values,
       coords={
            "latitude":  grid.index.values,
            "longitude": grid.columns.values
        },
        dims=("latitude", "longitude"),
    )
    # compute and return the weighted mean
    weighted = _spatial_average(da, region=None, skipna=True)
    return float(weighted.item())


def load_score_summary() -> pd.DataFrame: 
    rows = []
    q_levels = set()
    t_levels = set()

    for obs in OBS_SOURCES:
        for var in VARIABLES:
            print(f"Processing {obs} / {var}")
            for lt in LEAD_TIMES:
                fn   = f"{var}_lead{lt}d_{obs}.csv"
                path = os.path.join(SCORE_DATA_DIR, fn)
                if not os.path.exists(path):
                    print(f"⚠  Missing file: {path}")
                    continue

                df = pd.read_csv(path)
                for model, sub in df.groupby("model"):
                    pc_mean  = round(compute_lat_weighted_mean(sub, "pc"),  5)
                    pcs_mean = round(compute_lat_weighted_mean(sub, "pcs"), 5)

                    row = {
                        "model":      model,
                        "obs_source": obs,
                        "variable":   var,
                        "lead_time":  lt,
                        "PC":         pc_mean,
                        "PCS":        pcs_mean,
                    }

                    # per‑quantile (q)
                    for q in sorted(sub["q_value"].dropna().unique()):
                        q_levels.add(q)
                        df_q = sub[sub["q_value"] == q]
                        row[f"qw_PC_{q}"]  = round(compute_lat_weighted_mean(df_q,  "qw_pc"),  5)
                        row[f"qw_PCS_{q}"] = round(compute_lat_weighted_mean(df_q,  "qw_pcs"), 5)

                    # per‑threshold (t)
                    for t in sorted(sub["t_quantile"].dropna().unique()):
                        t_levels.add(t)
                        df_t = sub[sub["t_quantile"] == t]
                        row[f"tw_PC_{t}"]  = round(compute_lat_weighted_mean(df_t, "tw_pc"),  5)
                        row[f"tw_PCS_{t}"] = round(compute_lat_weighted_mean(df_t, "tw_pcs"), 5)

                    rows.append(row)

    # rebuild the full column list dynamically
    base_cols   = ["model","obs_source","variable","lead_time"]
    metric_cols = ["PC","PCS"]
    qw_cols     = [f"qw_PC_{q}"  for q in sorted(q_levels)] \
                + [f"qw_PCS_{q}" for q in sorted(q_levels)]
    tw_cols     = [f"tw_PC_{t}"  for t in sorted(t_levels)] \
                + [f"tw_PCS_{t}" for t in sorted(t_levels)]
    final_columns = base_cols + metric_cols + qw_cols + tw_cols

    # build and reorder the DataFrame
    result = pd.DataFrame(rows, columns=final_columns)

    # save to CSV for later reuse
    out_csv = os.path.join(SCORE_DATA_DIR, "score_summary.csv")
    result.to_csv(out_csv, index=False)
    print(f"Saved summary to {out_csv!r}")

    return result

def _era5_ref_pc_at_point(
    obs_ds: xr.Dataset,
    clim_ds: xr.Dataset,
    var: str,
    lead_days: int,
    lat: float,
    lon: float,
    mode: str,            # 'pcs' | 'tw' | 'qw'
    level: float | None   # None for 'pcs'; TW uses quantile; QW uses q in (0,1)
) -> float:
    """
    Reference PC from ERA5 rolling climatology vs ERA5 obs at one grid point.
    IMPORTANT: We sample OBS exactly at the provided (lat,lon) grid point to avoid any
               regridding or “nearest” bias. Only the climatology predictor uses nearest.
    """
    lead_td = np.timedelta64(lead_days, "D")

    # 1) Predictor from climatology at the *nearest* to the obs gridpoint
    preds = (
        clim_ds[var]
        .sel(
            prediction_timedelta=lead_td,
            latitude=lat,
            longitude=lon,
            method="nearest",
        )
        .sel(time=EVAL_PERIOD)
        .load()
    )
    if preds.size == 0:
        return np.nan

    # 2) Observation series at the *exact* obs grid labels we’re looping over
    #    (normalize only the longitude convention, not the grid cell choice)
    lon_obs = float(lon)
    if float(obs_ds.longitude.min()) >= 0.0 and lon_obs < 0.0:
        lon_obs = lon_obs % 360.0
    elif float(obs_ds.longitude.min()) < 0.0 and lon_obs > 180.0:
        lon_obs = ((lon_obs + 180.0) % 360.0) - 180.0

    # Observation series at the grid point
    obs_point = obs_ds[var].sel(latitude=lat, longitude=lon_obs)

    # Candidate verifying times
    target_times = preds.time.values + lead_td

    # Restrict to those that actually exist in obs
    valid_times = np.intersect1d(target_times, obs_point.time.values)

    # Select obs only where valid
    obs_series = obs_point.sel(time=valid_times)

    # Align with preds: keep only matching times
    preds_aligned = preds.sel(time=valid_times)

    x = preds_aligned.values
    y = obs_series.values
    if y.size == 0 or x.size == 0:
        return np.nan


    try:
        fitted   = idr(y, pd.DataFrame({"x": x}))
        pred_obj = fitted.predict(pd.DataFrame({"x": x}), digits=12)

        # Use your metric functions exactly as named
        if mode == "pcs":
            # base PC (mean CRPS) that goes into PCS
            return np.mean(pred_obj.crps(y))

        elif mode == "tw":
            if level is None:
                return np.nan
            t = float(np.nanquantile(y, level))
            if level < 0.5:
                # lower tail
                type(pred_obj).tw_crps_small = tw_crps_small
                return float(np.mean(pred_obj.tw_crps_small(y, t)))
            else:
                type(pred_obj).tw_crps = tw_crps
                return float(np.mean(pred_obj.tw_crps(y, t)))

        elif mode == "qw":
            if level is None:
                return np.nan
            # import only if you use QW here
            from metric_functions import qw_crps, qw_crps_small
            if level < 0.5:
                type(pred_obj).qw_crps_small = qw_crps_small
                return float(np.mean(pred_obj.qw_crps_small(y, q=level)))
            else:
                type(pred_obj).qw_crps = qw_crps
                return float(np.mean(pred_obj.qw_crps(y, q=level)))

        return np.nan
    except Exception:
        return np.nan



# --------------------- caching: filenames encode tail & level ---------------------
def _ref_pc_csv_path(var: str, lead_days: int, mode: str, level: float | None) -> str:
    """
    'pcs' → era5_refpc_pcs_none_<var>_leadXd.csv
    'tw'  → level<0.5:  era5_refpc_tw_lower_<level>_<var>_leadXd.csv
             level>=0.5: era5_refpc_tw_upper_<level>_<var>_leadXd.csv
    'qw'  → same lower/upper split with q
    """
    if mode == "pcs":
        tag = "pcs_none"
    else:
        if level is None:
            raise ValueError("level is required for TW/QW ref PCs")
        tail = "lower" if level < 0.5 else "upper"
        tag  = f"{mode}_{tail}_{level:.4f}"
    return os.path.join(REF_PC_DIR, f"era5_refpc_{tag}_{var}_lead{lead_days}d.csv")


def _build_ref_pc_csv_for_tile(
    var: str,
    lead_days: int,
    lat: float,
    lon: float,
    mode: str,
    level: float | None,
    obs_ds: xr.Dataset,
    clim_ds: xr.Dataset
) -> dict:
    ref_pc = _era5_ref_pc_at_point(
        obs_ds=obs_ds, clim_ds=clim_ds,
        var=var, lead_days=lead_days, lat=float(lat), lon=float(lon),
        mode=mode, level=level
    )
    return {"lat": float(lat), "lon": float(lon), "ref_pc": ref_pc}


def get_or_build_ref_pc_csv(
    var: str,
    lead_days: int,
    mode: str,                  # 'pcs' | 'tw' | 'qw'
    level: float | None = None, # required for 'tw' and 'qw'; ignored for 'pcs'
    n_jobs: int = 20
) -> str:
    """
    Ensure the reference PC CSV exists. If absent, build it in parallel (n_jobs default = 20).
    Returns the CSV path.
    """
    if mode not in ("pcs", "tw", "qw"):
        raise ValueError("mode must be 'pcs', 'tw', or 'qw'")

    out_csv = _ref_pc_csv_path(var, lead_days, mode, level)
    if os.path.exists(out_csv):
        return out_csv

    # Open datasets once
    obs_ds  = xr.open_zarr(ERA5_OBS_PATH, decode_timedelta=True).sel(time=EVAL_PERIOD)
    clim_ds = xr.open_zarr(CLIM_FORECAST_PATH, decode_timedelta=True).sel(time=EVAL_PERIOD)

    lats = obs_ds.latitude.values
    lons = obs_ds.longitude.values

    # Parallel grid loop
    tasks = (
        delayed(_build_ref_pc_csv_for_tile)(
            var, lead_days, lat, lon, mode, level, obs_ds, clim_ds
        )
        for lat in lats for lon in lons
    )
    rows = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(tasks)

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[cache] wrote reference PC → {out_csv}")
    return out_csv

# ------------------ Maps-functions -----------------------

def _plot_metric_grid(df: pd.DataFrame, metric: str, var: str, lead_time: int, suffix: str = "") -> None:
    if df.empty or metric not in df.columns:
        return

    df_mean = df.groupby(['lat', 'lon'])[metric].mean().reset_index()
    if df_mean.empty:
        return

    lat_vals = sorted(df_mean['lat'].unique())
    lon_vals = sorted(df_mean['lon'].unique())
    data_grid = df_mean.pivot(index='lat', columns='lon', values=metric).values

    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    lon_g, lat_g = np.meshgrid(lon_vals, lat_vals)
    norm = mcolors.Normalize(vmin=np.nanmin(data_grid), vmax=np.nanmax(data_grid))
    mesh = ax.pcolormesh(lon_g, lat_g, data_grid, cmap='coolwarm', norm=norm,
                         shading='nearest', transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, label=metric)
    ax.set_title(f"{var} – {metric} – Lead {lead_time}d ({suffix})")

    fname = f"{var}_{metric}_lead{lead_time}d_{suffix}.png"
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()

def plot_pcs_twq_map(
    lead_time:      int,
    t_level:        float,
    q_level:        float,
    obs_source:     str       = 'ifs',
    forecast_model: str       = 'hres',
    cmap:           str       = 'viridis',
    projection:     ccrs.CRS  = ccrs.Robinson()
) -> None:
    """
    3×N panel:
      • row 1: PCS (no threshold/quantile)
      • row 2: TW_PCS at t_level
      • row 3: QW_PCS at q_level

    für jedes VARIABLE bei festem lead_time, nur für forecast_model gegen obs_source.
    Farbskala ist fest auf 0..1.
    """
    # map CSV‐model keys to display names (lokal, unabhängig von globalen)
    NAME_MAP = {
        'mean_sea_level_pressure': 'MSLP',
        '2m_temperature':           'T2M',
        '10m_wind_speed':           'WS10',
    }
    BASE_MODEL_TITLES = {
        'hres':      'IFS-HRES',
        'pangu':     'Pangu',
        'graphcast': 'GraphCast',
    }

    # fester PCS-Normbereich 0..1
    PCS_NORM = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # determine how the model is named in the CSV
    if obs_source == 'ifs' and forecast_model in ('pangu','graphcast'):
        model_key = f"{forecast_model}_operational"
    else:
        model_key = forecast_model

    # helper zum Laden & Filtern
    def _load(var: str, metric: str, level: float|None) -> pd.DataFrame:
        fp = os.path.join(SCORE_DATA_DIR, f"{var}_lead{lead_time}d_{obs_source}.csv")
        if not os.path.exists(fp):
            return pd.DataFrame()
        df = pd.read_csv(fp)
        df = df[df['model'] == model_key]
        if metric not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=[metric])
        if metric == 'tw_pcs':
            df = df[df['t_quantile'] == t_level]
        elif metric == 'qw_pcs':
            df = df[df['q_value']    == q_level]
        return df

    # Panel aufbauen
    n_rows, n_cols = 3, len(VARIABLES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        subplot_kw={'projection': projection},
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    row_info = [
        ('pcs',    'PCS',        None),
        ('tw_pcs', f'TW_PCS\n(t={t_level:.2f})', t_level),
        ('qw_pcs', f'QW_PCS\n(q={q_level:.2f})', q_level),
    ]

    first_mesh = None

    for i, (metric, label, lvl) in enumerate(row_info):
        for j, var in enumerate(VARIABLES):
            ax = axes[i, j]
            df_sub = _load(var, metric, lvl)
            if df_sub.empty:
                ax.set_facecolor('lightgray')
                ax.coastlines()
                ax.set_title('no data')
                continue

            dfm  = df_sub.groupby(['lat','lon'])[metric].mean().reset_index()
            lons = np.sort(dfm['lon'].unique())
            lats = np.sort(dfm['lat'].unique())
            grid = dfm.pivot(index='lat', columns='lon', values=metric).values
            # optional absichern:
            # grid = np.clip(grid, 0.0, 1.0)

            lon_g, lat_g = np.meshgrid(lons, lats, indexing='xy')
            mesh = ax.pcolormesh(
                lon_g, lat_g, grid,
                cmap=cmap,
                norm=PCS_NORM,
                shading='nearest',
                transform=ccrs.PlateCarree()
            )
            if first_mesh is None:
                first_mesh = mesh

            ax.coastlines(linewidth=0.5)
            if i == 0:
                ax.set_title(NAME_MAP.get(var, var), fontsize=12)
            if j == 0:
                ax.text(-0.1, 0.5, label,
                        transform=ax.transAxes,
                        va='center', ha='right',
                        rotation=90, fontsize=10)

    # wenn gar keine Daten da waren
    if first_mesh is None:
        raise RuntimeError("No data for that model/levels/lead_time.")

    # super-title
    model_title = BASE_MODEL_TITLES.get(forecast_model, forecast_model)
    if obs_source == 'ifs' and forecast_model in ('pangu','graphcast'):
        model_title += '-Operational'

    fig.suptitle(
        f"Lead {lead_time}d — {model_title} vs {obs_source.upper()}",
        y=0.98, fontsize=14
    )

    # gemeinsame Colorbar
    cbar = fig.colorbar(first_mesh, ax=axes.ravel().tolist(),
                        orientation='horizontal',
                        fraction=0.05, pad=0.02)
    cbar.set_label('PCS variants', fontsize=12)

    # speichern & zeigen (mit "map" im Namen)
    fname = (f"pcs_twq_map_lead{lead_time}d_"
             f"{forecast_model}_vs_{obs_source}_"
             f"t{t_level:.2f}_q{q_level:.2f}.png")
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_map(
    q_level: float = 0.05,
    obs_source: str = "ifs",                 # "era5" or "ifs"
    projection: ccrs.CRS = ccrs.Robinson(),
    cmap: str = "viridis"
) -> None:
    """
    Two-row layout of quantile-based thresholds from the chosen ground truth.
    Top row: high thresholds (1 - q) for MSLP, T2M, WS10 → uses `cmap`.
    Bottom row (centered): low thresholds (q) for MSLP, T2M → uses 'OrRd'.
    """

    from matplotlib.gridspec import GridSpec

    # base directory: <this_file_dir>/data
    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    # map choice → full path
    zarr_map = {
        "ifs":  os.path.join(DATA_DIR, "ifs_analysis_64x32.zarr"),
        "era5": os.path.join(DATA_DIR, "era5_64x32.zarr"),
    }

    obs_source = obs_source.lower()
    if obs_source not in zarr_map:
        raise ValueError("obs_source must be 'era5' or 'ifs'.")

    obs_path = zarr_map[obs_source]
    gt_name  = obs_source.upper()

    UNITS_MAP = {
        "mean_sea_level_pressure": "Pa",
        "2m_temperature": "K",
        "10m_wind_speed": "m/s",
    }

    # --- helpers ---
    def _as_latlon(da: xr.DataArray) -> xr.DataArray:
        """Ensure last two dims are (latitude, longitude)."""
        dims = list(da.dims)
        lat_name = "latitude" if "latitude" in dims else ("lat" if "lat" in dims else None)
        lon_name = "longitude" if "longitude" in dims else ("lon" if "lon" in dims else None)
        if lat_name is None or lon_name is None:
            return da
        da2 = da.transpose(..., lat_name, lon_name, missing_dims="ignore")
        if da2.dims[-2] not in ("latitude", "lat"):
            da2 = da2.transpose(..., da2.dims[-1], da2.dims[-2])
        return da2

    def _panel_norm(C: np.ndarray) -> mcolors.Normalize:
        """Per-panel autoscale with fallback to [0, 1]."""
        vmin = float(np.nanmin(C)) if np.isfinite(np.nanmin(C)) else 0.0
        vmax = float(np.nanmax(C)) if np.isfinite(np.nanmax(C)) else 1.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin, vmax = 0.0, 1.0
        return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # --- load + quantiles (fixed evaluation window to match other plots) ---
    obs = xr.open_zarr(obs_path, decode_timedelta=True).sel(time=slice("2020-01-01", "2020-12-31"))

    var_mslp = "mean_sea_level_pressure"
    var_t2m  = "2m_temperature"
    var_ws10 = "10m_wind_speed"

    q_small = float(q_level)
    q_big   = float(1.0 - q_small)

    mslp_low  = _as_latlon(obs[var_mslp].chunk({"time": -1}).quantile(q_small, dim="time"))
    mslp_high = _as_latlon(obs[var_mslp].chunk({"time": -1}).quantile(q_big,   dim="time"))
    t2m_low   = _as_latlon(obs[var_t2m ].chunk({"time": -1}).quantile(q_small, dim="time"))
    t2m_high  = _as_latlon(obs[var_t2m ].chunk({"time": -1}).quantile(q_big,   dim="time"))
    ws10_high = _as_latlon(obs[var_ws10].chunk({"time": -1}).quantile(q_big,   dim="time"))

    lat = mslp_low["latitude"].values if "latitude" in mslp_low.dims else mslp_low["lat"].values
    lon = mslp_low["longitude"].values if "longitude" in mslp_low.dims else mslp_low["lon"].values
    lon_g, lat_g = np.meshgrid(lon, lat, indexing="xy")

    # --- colormaps ---
    cmap_high = plt.get_cmap(cmap)   # for high thresholds (top row)
    cmap_low  = plt.get_cmap("OrRd") # for low thresholds (bottom row)

    # --- figure with centered bottom row using GridSpec(2,6) ---
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 6, figure=fig, wspace=0.02, hspace=0.04)

    # Top row: three highs, each spanning 2 columns
    ax_mslp_hi = fig.add_subplot(gs[0, 0:2], projection=projection)
    ax_t2m_hi  = fig.add_subplot(gs[0, 2:4], projection=projection)
    ax_ws10_hi = fig.add_subplot(gs[0, 4:6], projection=projection)

    # Bottom row (centered): two lows occupying middle spans
    ax_mslp_lo = fig.add_subplot(gs[1, 1:3], projection=projection)
    ax_t2m_lo  = fig.add_subplot(gs[1, 3:5], projection=projection)

    pct_small = int(round(q_small * 100))
    pct_big   = int(round(q_big * 100))

    panels = [
        # top (highs) → cmap_high
        (ax_mslp_hi, var_mslp, f"{NAME_MAP['mean_sea_level_pressure']} - thresholds from {pct_big}% quantile",  mslp_high, cmap_high),
        (ax_t2m_hi,  var_t2m,  f"{NAME_MAP['2m_temperature']} - thresholds from {pct_big}% quantile",            t2m_high,  cmap_high),
        (ax_ws10_hi, var_ws10, f"{NAME_MAP['10m_wind_speed']} - thresholds from {pct_big}% quantile",           ws10_high, cmap_high),
        # bottom (lows, centered) → cmap_low
        (ax_mslp_lo, var_mslp, f"{NAME_MAP['mean_sea_level_pressure']} - thresholds from {pct_small}% quantile", mslp_low,  cmap_low),
        (ax_t2m_lo,  var_t2m,  f"{NAME_MAP['2m_temperature']} - thresholds from {pct_small}% quantile",          t2m_low,   cmap_low),
    ]

    for ax, var_key, title, da, panel_cmap in panels:
        C = da.values
        norm = _panel_norm(C)
        mesh = ax.pcolormesh(
            lon_g, lat_g, C,
            transform=ccrs.PlateCarree(),
            cmap=panel_cmap,
            norm=norm,
            shading="nearest"
        )
        ax.coastlines()
        ax.set_title(title, fontsize=14)
        cbar = plt.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)
        units = UNITS_MAP.get(var_key, "")
        if units:
            cbar.ax.set_xlabel(units)

    fig.suptitle(f"Quantile-based thresholds (Ground Truth: {gt_name})", y=0.975, fontsize=20)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.96)

    out_fname = f"threshold_map_{gt_name}_quantiles{pct_small}_{pct_big}_centered.png"
    fig.savefig(os.path.join(PLOTS_DIR, out_fname), dpi=150, bbox_inches="tight")
    plt.show()


def plot_spatial_map(
    metric_base: str,
    level: float | None,
    obs_source: str,
    forecast_model: str,
    projection: ccrs.CRS = ccrs.Robinson(),
    cmap: str = "viridis"
) -> None:
    """
    5×3 Panel globaler Maps für `metric_base` (optional `level`)
    von `forecast_model` gegen `obs_source`.
    Farbskala ist fest auf 0..1.

    metric_base:
      - 'pc' / 'pcs'                → pure score (ignoriert level)
      - 'tw_pc'/'tw_pcs'            → threshold‐weighted (braucht level)
      - 'qw_pc'/'qw_pcs'            → quantile‐weighted  (braucht level)
    """
    # fester PCS-Normbereich 0..1
    PCS_NORM = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Only append “_operational” when comparing to IFS
    if obs_source == "ifs" and forecast_model in ("pangu", "graphcast"):
        model_key = f"{forecast_model}_operational"
    else:
        model_key = forecast_model

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        dfm = df[df["model"] == model_key]
        if metric_base in ("pc", "pcs"):
            return dfm.dropna(subset=[metric_base])
        if level is None:
            raise ValueError(f"Must supply `level` for metric '{metric_base}'")
        if metric_base.startswith("tw_"):
            return dfm[(~dfm[metric_base].isna()) & (dfm["t_quantile"] == level)]
        if metric_base.startswith("qw_"):
            return dfm[(~dfm[metric_base].isna()) & (dfm["q_value"]    == level)]
        raise ValueError(f"Unknown metric_base '{metric_base}'")

    # Panel aufbauen
    fig, axes = plt.subplots(
        len(LEAD_TIMES), len(VARIABLES),
        figsize=(4*len(VARIABLES), 3*len(LEAD_TIMES)),
        subplot_kw={"projection": projection},
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    first_mesh = None

    for i, lt in enumerate(LEAD_TIMES):
        for j, var in enumerate(VARIABLES):
            ax   = axes[i, j]
            fn   = f"{var}_lead{lt}d_{obs_source}.csv"
            path = os.path.join(SCORE_DATA_DIR, fn)
            if not os.path.exists(path):
                ax.set_facecolor("lightgray")
                ax.coastlines()
                ax.set_title("missing")
                continue

            sub = _filter(pd.read_csv(path))
            if sub.empty:
                ax.set_facecolor("lightgray")
                ax.coastlines()
                ax.set_title("no data")
                continue

            grid = (
                sub.groupby(["lat","lon"])[metric_base]
                   .mean().reset_index()
                   .pivot(index="lat", columns="lon", values=metric_base)
                   .values
            )
            # optional absichern:
            # grid = np.clip(grid, 0.0, 1.0)

            lon_g, lat_g = np.meshgrid(
                np.sort(sub["lon"].unique()),
                np.sort(sub["lat"].unique())
            )
            mesh = ax.pcolormesh(
                lon_g, lat_g, grid,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=PCS_NORM,
                shading="nearest"
            )
            if first_mesh is None:
                first_mesh = mesh

            ax.coastlines(linewidth=0.5)
            if i == 0:
                ax.set_title(NAME_MAP.get(var, var), fontsize=12)
            if j == 0:
                ax.text(-0.1, 0.5, f"Lead {lt}d",
                        transform=ax.transAxes, va="center",
                        ha="right", rotation=90, fontsize=10)

    if first_mesh is None:
        suffix = "" if metric_base in ("pc","pcs") else f" (level={level})"
        raise RuntimeError(f"No data for {metric_base.upper()}{suffix} with model_key='{model_key}'")

    # super‐title & colorbar
    lvl_str    = "" if metric_base in ("pc","pcs") else f"={level}"
    mt         = BASE_MODEL_TITLES[forecast_model]
    if obs_source == "ifs" and forecast_model in ("pangu","graphcast"):
        mt += "-Operational"
    fig.suptitle(
        f"{metric_base.upper()}{lvl_str} for {mt} vs {obs_source.upper()}",
        y=0.98, fontsize=14
    )
    cbar = fig.colorbar(first_mesh, ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.05, pad=0.02)
    cbar.set_label(metric_base.upper(), fontsize=12)

    # speichern & zeigen (mit "map" im Namen)
    suffix = "" if metric_base in ("pc","pcs") else f"_{level}"
    out    = f"map_{metric_base}{suffix}_{forecast_model}_vs_{obs_source}.png"
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    plt.show()



    
def plot_climatology_ref_pcs_3x5(
    *,
    metric_base: str,                       # 'tw_pcs' | 'qw_pcs'
    level_low: float = 0.05,                # lower-tail level (e.g., 0.05)
    level_high: float = 0.95,               # upper-tail level (e.g., 0.95)
    lead_days: int = 3,                     # single lead time to plot (days)
    obs_source: str = "era5",
    projection: ccrs.CRS = ccrs.Robinson(),
    cmap_pos: str = "viridis",              # positive ramp (good skill)
    cmap_neg: str = "OrRd"                  # negative ramp (bad skill)
) -> None:
    """
    3×5 panel of TW/QW-PCS maps vs ERA5 rolling climatology as PC_ref (PC(0)) for a single lead time.
    Rows: GraphCast, Pangu-Weather, IFS-HRES.
    Columns: [MSLP↓, MSLP↑, T2M↓, T2M↑, WS10↑].
    Color scale is asymmetric with vmin = global min across panel and vmax = 1.

    Notes
    -----
    • Uses a single lead time specified by 'lead_days' (default 3).
    • Negative values use 'cmap_neg', positive values use 'cmap_pos'.
    """

    metric_base = metric_base.lower()
    if metric_base not in {"tw_pcs", "qw_pcs"}:
        raise ValueError("metric_base must be 'tw_pcs' or 'qw_pcs'.")

    # Map base to column names and labels
    if metric_base == "tw_pcs":
        pc_col, level_col, ref_mode, metric_lbl = "tw_pc", "t_quantile", "tw", "TW_PCS"
    else:
        pc_col, level_col, ref_mode, metric_lbl = "qw_pc", "q_value", "qw", "QW_PCS"

    # Columns identical to the 5×5 panel
    COL_SPECS = [
        ("mean_sea_level_pressure", "lower", "MSLP (low)"),
        ("mean_sea_level_pressure", "upper", "MSLP (high)"),
        ("2m_temperature",          "lower", "T2M (cold)"),
        ("2m_temperature",          "upper", "T2M (hot)"),
        ("10m_wind_speed",          "upper", "WS10 (high)"),
    ]

    # Row ordering: GraphCast, Pangu-Weather, IFS-HRES
    ROW_MODELS = ["graphcast", "pangu", "hres"]
    MODEL_TITLES = {"graphcast": "GC-ERA5", "pangu": "PW-ERA5", "hres": "HRES"}

    # Resolve model key for CSVs (operational suffix for IFS obs)
    def _model_key_for_csv(model: str) -> str:
        if obs_source == "ifs" and model in ("graphcast", "pangu"):
            return f"{model}_operational"
        return model

    # Figure and bookkeeping
    n_rows, n_cols = len(ROW_MODELS), len(COL_SPECS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        subplot_kw={"projection": projection},
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    first_mesh, mesh_handles = None, []
    global_vmin = 0.0

    for i, model in enumerate(ROW_MODELS):
        model_key = _model_key_for_csv(model)

        for j, (var, tail, col_title) in enumerate(COL_SPECS):
            ax = axes[i, j]
            level = level_low if tail == "lower" else level_high

            # Read the single-lead CSV for this variable/source
            csv_path = os.path.join(SCORE_DATA_DIR, f"{var}_lead{lead_days}d_{obs_source}.csv")
            if not os.path.exists(csv_path):
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("missing"); continue

            df = pd.read_csv(csv_path)
            if df.empty or pc_col not in df.columns or level_col not in df.columns:
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            sub = df[(df["model"] == model_key) & (~df[pc_col].isna())]
            sub = sub[sub[level_col] == level]
            if sub.empty:
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            # Build/load ERA5 climatology PC(0) for this var/lead/tail level
            ref_csv = get_or_build_ref_pc_csv(
                var=var, lead_days=int(lead_days), mode=ref_mode, level=level, n_jobs=5
            )
            ref_df = pd.read_csv(ref_csv)

            merged = (
                sub.groupby(["lat", "lon"])[pc_col].mean().reset_index()
                .merge(ref_df, on=["lat", "lon"], how="left")
            )
            if merged.empty:
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            # PCS at grid point relative to climatology PC(0)
            merged["val"] = np.where(
                np.isfinite(merged["ref_pc"]) & (merged["ref_pc"] > 0.0),
                (merged["ref_pc"] - merged[pc_col]) / merged["ref_pc"],
                np.nan
            )

            grid = (
                merged[["lat", "lon", "val"]]
                .pivot(index="lat", columns="lon", values="val")
            )
            if grid.isna().all().all():
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            # Track global minimum for asymmetric scaling
            local_min = np.nanmin(grid.values)
            if np.isfinite(local_min):
                global_vmin = min(global_vmin, float(local_min))

            # Plot with temporary cmap; unified cmap/norm applied later
            lon_g, lat_g = np.meshgrid(grid.columns.values, grid.index.values, indexing="xy")
            mesh = ax.pcolormesh(
                lon_g, lat_g, grid.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap_pos,
                shading="nearest"
            )
            if first_mesh is None:
                first_mesh = mesh
            mesh_handles.append(mesh)

            ax.coastlines(linewidth=0.5)
            if i == 0:
                ax.set_title(col_title, fontsize=12)
            if j == 0:
                ax.text(
                    -0.1, 0.5, MODEL_TITLES.get(model, model),
                    transform=ax.transAxes, va="center", ha="right",
                    rotation=90, fontsize=12
                )

    if first_mesh is None:
        raise RuntimeError("No data available to render the 3×5 climatology-reference PCS panel.")

    # Asymmetric colormap (negatives = cmap_neg, positives = cmap_pos)
    n_colors = 256
    vmin = float(global_vmin) if np.isfinite(global_vmin) else -1.0
    frac_neg = -vmin / (1.0 - vmin) if vmin < 0 else 0.0
    n_neg = int(np.clip(int(np.round(frac_neg * n_colors)), 1, n_colors - 1))

    neg_part = plt.get_cmap(cmap_neg)(np.linspace(0, 1, n_neg))
    pos_part = plt.get_cmap(cmap_pos)(np.linspace(0, 1, n_colors - n_neg))
    custom_cmap = mcolors.ListedColormap(np.vstack([neg_part, pos_part]))
    custom_norm = mcolors.Normalize(vmin=vmin, vmax=1.0)

    for m in mesh_handles:
        m.set_cmap(custom_cmap)
        m.set_norm(custom_norm)

    # Title and colorbar (explicitly state the extremes and lead time)
    lo_pct = int(round(level_low * 100))
    hi_pct = int(round(level_high * 100))
    fig.suptitle(
        f"{metric_lbl} with ERA5 climatology as twPC(0) - {lo_pct}% lower & {hi_pct}% upper extremes - Lead Time = {lead_days}d",
        y=0.99, fontsize=16
    )
    cbar = fig.colorbar(
        first_mesh, ax=axes.ravel().tolist(),
        orientation="horizontal", fraction=0.05, pad=0.02
    )
    cbar.set_label(metric_lbl, fontsize=12)

    out = (
        f"map_3x5_{metric_base}_era5clim_low{level_low:.2f}_high{level_high:.2f}_lead{lead_days}d.png"
    )
    _ensure_dir(os.path.join(PLOTS_DIR, out))
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    plt.show()


def climatological_pcs_heatmap(
    metric_base: str,                    # 'tw_pcs' | 'qw_pcs'
    level_low: float = 0.05,             # lower-tail level (e.g., 0.05 → 5%)
    level_high: float = 0.95,            # upper-tail level (e.g., 0.95 → 95%)
    obs_source: str      = "era5",
    forecast_model: str  = "graphcast",
    projection: ccrs.CRS  = ccrs.Robinson(),
    cmap: str             = "viridis"    # positive ramp; negatives use OrRd in the combined cmap
) -> None:
    """
    Heatmap with rows = lead times and 5 columns for extremes:
    [MSLP (low), MSLP (high), T2M (cold), T2M (hot), WS10 (high)].

    Each subplot shows TW/QW-PCS relative to ERA5 rolling climatology PC(0)
    at the requested tail level (lower uses 'level_low', upper uses 'level_high').

    Color scaling is asymmetric panel-wise: vmin = global min, vmax = 1.0.
    Negative values (worse than climatology) are mapped with OrRd; positives with 'cmap'.
    """

    # ------------------------ validation and config ------------------------
    metric_base = metric_base.lower()
    if metric_base not in {"tw_pcs", "qw_pcs"}:
        raise ValueError("This heatmap uses extreme columns; use 'tw_pcs' or 'qw_pcs'.")

    if not (0.0 < level_low < 0.5 and 0.5 < level_high < 1.0):
        raise ValueError("level_low must be in (0, 0.5) and level_high in (0.5, 1).")

    # Map metric selection → column names in CSVs and ERA5 ref mode
    if metric_base == "tw_pcs":
        pc_col, level_col, ref_mode, metric_lbl = "tw_pc", "t_quantile", "tw", "TW_PCS"
    else:
        pc_col, level_col, ref_mode, metric_lbl = "qw_pc", "q_value", "qw", "QW_PCS"

    # Fixed 5-column spec identical to your 5×5 panel
    COL_SPECS = [
        ("mean_sea_level_pressure", "lower", "MSLP (low)"),
        ("mean_sea_level_pressure", "upper", "MSLP (high)"),
        ("2m_temperature",          "lower", "T2M (cold)"),
        ("2m_temperature",          "upper", "T2M (hot)"),
        ("10m_wind_speed",          "upper", "WS10 (high)"),
    ]

    # Figure
    n_rows, n_cols = len(LEAD_TIMES), len(COL_SPECS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        subplot_kw={"projection": projection},
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    first_mesh = None
    mesh_handles = []
    global_vmin = 0.0

    # Keep non-operational keys for ERA5 ref (same convention as your earlier function)
    model_key = forecast_model

    # ------------------------ main plotting loops ------------------------
    for i, lt in enumerate(LEAD_TIMES):
        for j, (var, tail, col_title) in enumerate(COL_SPECS):
            ax = axes[i, j]

            # Select level per tail
            level = level_low if tail == "lower" else level_high

            # Ensure/load ERA5 climatology PC(0) at this tail/level
            ref_csv = get_or_build_ref_pc_csv(
                var=var, lead_days=int(lt), mode=ref_mode, level=level, n_jobs=5
            )
            ref_df = pd.read_csv(ref_csv)

            # Load model PCs for this variable/lead/source
            csv_path = os.path.join(SCORE_DATA_DIR, f"{var}_lead{lt}d_{obs_source}.csv")
            if not os.path.exists(csv_path):
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("missing"); continue

            df = pd.read_csv(csv_path)
            if df.empty or pc_col not in df.columns or level_col not in df.columns:
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            sub = df[(df["model"] == model_key) & (~df[pc_col].isna())]
            sub = sub[sub[level_col] == level]
            if sub.empty:
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            # Average duplicates to (lat, lon) then merge with reference
            dfm = (
                sub.groupby(["lat", "lon"])[pc_col].mean().reset_index()
                .merge(ref_df, on=["lat", "lon"], how="left")
            )
            if dfm.empty:
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            # Compute PCS relative to climatology PC(0)
            dfm["val"] = np.where(
                np.isfinite(dfm["ref_pc"]) & (dfm["ref_pc"] > 0.0),
                (dfm["ref_pc"] - dfm[pc_col]) / dfm["ref_pc"],
                np.nan
            )
            grid = dfm.pivot(index="lat", columns="lon", values="val")
            if grid.isna().all().all():
                ax.set_facecolor("lightgray"); ax.coastlines(); ax.set_title("no data"); continue

            # Track global minimum for panel-asymmetric scaling
            local_min = np.nanmin(grid.values)
            if np.isfinite(local_min):
                global_vmin = min(global_vmin, float(local_min))

            # Temporary plot (uniform cmap/norm applied later)
            lon_g, lat_g = np.meshgrid(grid.columns.values, grid.index.values, indexing="xy")
            mesh = ax.pcolormesh(
                lon_g, lat_g, grid.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap,                    # placeholder; will be overridden
                shading="nearest"
            )
            if first_mesh is None:
                first_mesh = mesh
            mesh_handles.append(mesh)

            ax.coastlines(linewidth=0.5)

            # Column headers on top row
            if i == 0:
                ax.set_title(col_title, fontsize=12)
            # Row labels on left column
            if j == 0:
                label = f"Lead Time = {lt} day" if int(lt) == 1 else f"Lead Time = {lt} days"
                ax.text(-0.1, 0.5, label, transform=ax.transAxes,
                        va="center", ha="right", rotation=90, fontsize=12)

    if first_mesh is None:
        raise RuntimeError("No data to plot for the requested settings.")

    # ------------------------ asymmetric colormap (OrRd for negatives) ------------------------
    n_colors = 256
    vmin = float(global_vmin) if np.isfinite(global_vmin) else -1.0
    frac_neg = -vmin / (1.0 - vmin) if vmin < 0 else 0.0
    n_neg = int(np.clip(int(np.round(frac_neg * n_colors)), 1, n_colors - 1))

    neg_part = plt.cm.OrRd(np.linspace(0, 1, n_neg))                  # negatives
    pos_part = plt.get_cmap(cmap)(np.linspace(0, 1, n_colors - n_neg))# positives
    custom_cmap  = mcolors.ListedColormap(np.vstack([neg_part, pos_part]))
    custom_norm  = mcolors.Normalize(vmin=vmin, vmax=1.0)

    for m in mesh_handles:
        m.set_cmap(custom_cmap)
        m.set_norm(custom_norm)

    # ------------------------ title, colorbar, save ------------------------
    lo_pct = int(round(level_low * 100))
    hi_pct = int(round(level_high * 100))
    mt = BASE_MODEL_TITLES.get(forecast_model, forecast_model)

    fig.suptitle(
        f"{metric_lbl} with ERA5 climatology as twPC(0) - {lo_pct}% lower & {hi_pct}% upper extremes - {mt}",
        y=0.99, fontsize=16
    )

    cbar = fig.colorbar(
        first_mesh, ax=axes.ravel().tolist(),
        orientation="horizontal", fraction=0.05, pad=0.02
    )
    cbar.set_label(metric_lbl, fontsize=12)

    out = (
        f"heatmap_leadtimes_5cols_{metric_base}_"
        f"low{level_low:.2f}_high{level_high:.2f}_{forecast_model}_vs_{obs_source}.png"
    )
    _ensure_dir(os.path.join(PLOTS_DIR, out))
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    plt.show()

# --------------------------- Panel functions -------------------------


def plot_model_panel(
    summary_df: pd.DataFrame,
    obs_source: str,
    t_level: float,
    q_level: float,
    base_metric: str = "pcs"
) -> None:
    """
    3 rows (BASE, TW, QW) × 5 columns (MSLP low/high, T2M cold/hot, WS10 high).
    Each subplot: three lines (GC, PW, HRES) vs lead time.
    Row labels include the actually used level (column/tail dependent).
    """

    BASE = base_metric.upper()  # "PC" or "PCS"

    COL_SPECS = [
        ("mean_sea_level_pressure", "lower", "MSLP (low extremes)"),
        ("mean_sea_level_pressure", "upper", "MSLP (high extremes)"),
        ("2m_temperature",          "lower", "T2M (cold extremes)"),
        ("2m_temperature",          "upper", "T2M (hot extremes)"),
        ("10m_wind_speed",          "upper", "WS10 (high extremes)"),
    ]

    # Filter by Ground Truth
    df = summary_df[summary_df["obs_source"] == obs_source].copy()
    if df.empty:
        print(f"⚠️ No rows for obs_source={obs_source!r} in summary_df")
        return

    # --- map model names to canonical keys ---
    if obs_source == "ifs":
        model_map = {
            "graphcast_operational": "graphcast",
            "pangu_operational": "pangu",
            "hres": "hres",
        }
    else:  # era5 or anything else already uses canonical keys
        model_map = {"graphcast": "graphcast", "pangu": "pangu", "hres": "hres"}

    df["model_canon"] = df["model"].map(model_map).fillna(df["model"])

    # Helper to detect available TW/QW levels
    def _levels(prefix: str) -> list[float]:
        vals = []
        for col in df.columns:
            if col.startswith(prefix):
                try:
                    vals.append(float(col.split("_")[-1]))
                except Exception:
                    pass
        return sorted(set(vals))

    tw_levels = _levels(f"tw_{BASE}_")
    qw_levels = _levels(f"qw_{BASE}_")

    def _pick_level(levels: list[float], target: float, tail: str) -> float | None:
        if not levels:
            return None
        if tail == "lower":
            cand = [x for x in levels if x < 0.5] or levels
            return min(cand, key=lambda x: abs(x - min(target, 1 - target)))
        else:
            cand = [x for x in levels if x > 0.5] or levels
            return min(cand, key=lambda x: abs(x - max(target, 1 - target)))

    def _metric_and_label(row_kind: str, tail: str) -> tuple[str | None, str]:
        if row_kind == "BASE":
            return BASE, BASE
        if row_kind == "TW":
            lvl = _pick_level(tw_levels, t_level, tail)
            if lvl is None:
                return None, f"TW_{BASE}"
            return f"tw_{BASE}_{lvl}", f"TW_{BASE} (quantile for local t = {lvl:.2f})"
        if row_kind == "QW":
            lvl = _pick_level(qw_levels, q_level, tail)
            if lvl is None:
                return None, f"QW_{BASE}"
            return f"qw_{BASE}_{lvl}", f"QW_{BASE} (q = {lvl:.2f})"
        raise ValueError(row_kind)

    ROWS = [("BASE",), ("TW",), ("QW",)]

    # Desired plotting/legend order using canonical keys
    wanted_order = ["graphcast", "pangu", "hres"]
    models = [m for m in wanted_order if m in df["model_canon"].unique()]
    if not models:
        print("⚠️ No models found in summary_df for given obs_source.")
        return

    def _label(m: str) -> str:
        if m == "graphcast": return f"GC-{obs_source.upper()}"
        if m == "pangu":     return f"PW-{obs_source.upper()}"
        if m == "hres":      return "HRES"
        return m

    cmap   = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(wanted_order)}  # stable colors
    colors = {m: colors[m] for m in models}  # keep only present models

    n_rows, n_cols = len(ROWS), len(COL_SPECS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.6 * n_cols, 3.2 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for i, (row_kind,) in enumerate(ROWS):
        for j, (var, tail, col_title) in enumerate(COL_SPECS):
            ax = axes[i, j]
            metric, ylab = _metric_and_label(row_kind, tail)

            if i == 0:
                ax.set_title(col_title)

            if metric is None or metric not in df.columns:
                ax.set_facecolor("lightgray")
                ax.grid(alpha=0.2)
                ax.set_ylabel(ylab)
                if i == n_rows - 1:
                    ax.set_xlabel("Lead Time [d]")
                continue

            sub = df[df["variable"] == var]
            for m in models:
                seg = sub[sub["model_canon"] == m][["lead_time", metric]].dropna()
                if seg.empty:
                    continue
                ax.plot(
                    seg["lead_time"], seg[metric],
                    marker="o", ms=5, color=colors[m], label=_label(m)
                )

            ax.set_ylabel(ylab)
            if i == n_rows - 1:
                ax.set_xlabel("Lead Time [d]")
            ax.grid(alpha=0.3)

    # Legend
    handles = [Line2D([], [], marker="o", ls="-", ms=6, color=colors[m]) for m in models]
    labels  = [_label(m) for m in models]
    fig.legend(
        handles, labels,
        ncol=len(labels),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        frameon=False
    )

    out_fname = f"model_panel5cols_{BASE}_t{t_level}_q{q_level}_{obs_source}.png"
    fig.savefig(os.path.join(PLOTS_DIR, out_fname), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(PLOTS_DIR, out_fname))
    plt.show()


def plot_sensitivity_panel(
    summary_df: pd.DataFrame,
    obs_source:  str,
    base_metric: str = "pcs",      # "pc" or "pcs"
    weight_type: str = "qw"        # "qw" or "tw"
) -> None:
    """
    Sensitivity panel (line plots, not maps):
      rows = [ BASE,
               <weight_type>_BASE@lvl1,
               <weight_type>_BASE@lvl2,
               ... ]
      cols = VARIABLES,
      one line per model, filtered to obs_source.
    """

    # Build metric names
    BASE   = base_metric.upper()                # e.g. "PC" or "PCS"
    prefix = f"{weight_type}_{BASE}_"           # e.g. "qw_PC_"

    # Discover all levels by splitting on '_'
    levels = sorted(
        float(col.split("_")[-1])
        for col in summary_df.columns
        if col.startswith(prefix)
    )

    # The full list of columns to plot, and labels for each row
    metrics    = [BASE] + [f"{prefix}{lvl}" for lvl in levels]
    row_labels = [BASE] + [
        f"{weight_type.upper()}_{BASE} ({weight_type}={lvl})"
        for lvl in levels
    ]

    # Filter to the chosen ground truth
    df = summary_df[summary_df["obs_source"] == obs_source]

    # Sanity‐check
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print(f"⚠️ Missing these metrics in summary_df: {missing}")
        return

    # Setup colors for each model
    models = sorted(df["model"].unique())
    cmap   = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models)}

    # Panel shape: one row per metric, one column per variable
    n_rows, n_cols = len(metrics), len(VARIABLES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True, sharey=False,
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    # Plot
    for i, metric in enumerate(metrics):
        for j, var in enumerate(VARIABLES):
            ax = axes[i, j]
            sub = df[df["variable"] == var]
            for model in models:
                seg = sub[sub["model"] == model]
                if seg.empty:
                    continue
                ax.plot(
                    seg["lead_time"],
                    seg[metric],
                    marker="o", ms=5,
                    color=colors[model],
                    label=model
                )
            # Labels
            if i == n_rows - 1:
                ax.set_xlabel("Lead Time [d]")
            if j == 0:
                ax.set_ylabel(row_labels[i])
            if i == 0:
                ax.set_title(NAME_MAP.get(var, var))
            ax.grid(alpha=0.3)

    # Single legend at bottom
    handles = [
        Line2D([], [], marker="o", ls="-", ms=6, color=colors[m])
        for m in models
    ]
    labels = [
        m.replace("_operational","")
         .replace("graphcast","GraphCast")
         .replace("pangu","Pangu‑Weather")
         .replace("hres","IFS‑HRES")
        for m in models
    ]
    fig.legend(
        handles, labels,
        ncol=min(len(labels), 6),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        frameon=False
    )
    """
    # Title and save
    fig.suptitle(
        f"Sensitivity of {BASE} to {weight_type.upper()} levels {levels}\n"
        f"(obs: {obs_source.upper()})",
        y=1.02
    )
    """
    out = (
        f"sensitivity_{BASE}_{weight_type}_"
        f"{'_'.join(f'{lvl:.2f}' for lvl in levels)}"
        f"_vs_{obs_source}.png"
    )
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(PLOTS_DIR, out))
    plt.show()


def plot_performance_diff_panel(
    summary_df: pd.DataFrame,
    obs_source: str,
    ai_model: str,                    # "graphcast" or "pangu"
    base_metric: str = "pc",          # "pc" or "pcs"
    metric_type: str = "tw",          # "tw" or "qw"
    diff_type: str = "absolute"       # "absolute" or "percent"
) -> None:
    """
    Two-row, five-column panel.
      Cols:  [MSLP↓, MSLP↑, T2M↓, T2M↑, WS10↑]
      Row 1: Base metric curves (HRES vs AI), SAME y-scale across the row.
      Row 2: Differences: base metric diff (grey) + per-level diffs of metric_type
             (only the appropriate tail per column), SAME y-scale across the row.
    """

    # --------------------------- validate & normalize ---------------------------
    BASE = base_metric.lower()
    if BASE not in ("pc", "pcs"):
        raise ValueError("base_metric must be 'pc' or 'pcs'")
    BASE_UP = BASE.upper()

    if metric_type not in ("tw", "qw"):
        raise ValueError("metric_type must be 'tw' or 'qw'")

    if diff_type not in ("absolute", "percent"):
        raise ValueError("diff_type must be 'absolute' or 'percent'")

    # CSV keys for models
    hres_key = "hres"
    if obs_source == "ifs" and ai_model in ("pangu", "graphcast"):
        ai_key = f"{ai_model}_operational"
    else:
        ai_key = ai_model

    # Display names consistent with other plots
    def model_label(key: str) -> str:
        if "graphcast" in key:
            return f"GC-{'IFS' if obs_source=='ifs' else 'ERA5'}"
        if "pangu" in key:
            return f"PW-{'IFS' if obs_source=='ifs' else 'ERA5'}"
        return "HRES"

    # Variable labels
    var_label = {
        "mean_sea_level_pressure": "MSLP",
        "2m_temperature":          "T2M",
        "10m_wind_speed":          "WS10",
    }

    # Five columns (variable, tail)
    col_specs = [
        ("mean_sea_level_pressure", "lower"),
        ("mean_sea_level_pressure", "upper"),
        ("2m_temperature",          "lower"),
        ("2m_temperature",          "upper"),
        ("10m_wind_speed",          "upper"),
    ]
    first_lower_idx = next(i for i, (_, t) in enumerate(col_specs) if t == "lower")

    # Filter to chosen ground truth
    df = summary_df[summary_df["obs_source"] == obs_source].copy()
    if df.empty:
        raise RuntimeError(f"No rows in summary_df for obs_source={obs_source!r}")

    # Discover available levels for the chosen weight type (split into tails)
    prefix = f"{metric_type}_{BASE_UP}_"
    levels_all = sorted({
        float(col.split("_")[-1])
        for col in df.columns if col.startswith(prefix)
    })
    levels_low  = [x for x in levels_all if x < 0.5]
    levels_high = [x for x in levels_all if x > 0.5]

    # ---------------------- symmetric color mapping -----------------------
    # Build a single palette indexed by distance from 0.5 so 0.01 and 0.99 share color.
    model_colors = {hres_key: (0.70, 0.70, 0.70), ai_key: (0.25, 0.25, 0.25)}
    fam = plt.get_cmap("Oranges")

    def level_palette(n: int) -> list:
        n = max(1, n)
        return [fam(x) for x in np.linspace(0.55, 0.9, n)]

    # Unique symmetric distances from 0.5 (exclude exactly 0.5)
    sym_dists = sorted({round(min(q, 1.0 - q), 10) for q in levels_all if not np.isclose(q, 0.5)})
    colors_symmetric = level_palette(len(sym_dists))
    dist2color = {d: c for d, c in zip(sym_dists, colors_symmetric)}

    def level_color(q: float):
        d = round(min(q, 1.0 - q), 10)
        return dist2color.get(d, (0.5, 0.5, 0.5))

    # Panel setup
    n_rows, n_cols = 2, len(col_specs)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.1 * n_cols, 3.2 * n_rows),
        sharex=True,
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    # ------------------- helpers for differences (row 2) -----------------------
    def _diff_base(mrg: pd.DataFrame) -> np.ndarray:
        if diff_type == "absolute":
            if BASE == "pc":
                return mrg[f"{BASE_UP}_hres"] - mrg[f"{BASE_UP}_ai"]
            return mrg[f"{BASE_UP}_ai"] - mrg[f"{BASE_UP}_hres"]
        denom = mrg[f"{BASE_UP}_hres"].replace(0.0, np.nan)
        if BASE == "pc":
            val = (mrg[f"{BASE_UP}_hres"] - mrg[f"{BASE_UP}_ai"]) / denom * 100.0
        else:
            val = (mrg[f"{BASE_UP}_ai"] - mrg[f"{BASE_UP}_hres"]) / denom * 100.0
        return val.to_numpy()

    def _diff_level(mrg: pd.DataFrame, col: str) -> np.ndarray:
        if diff_type == "absolute":
            if BASE == "pc":
                return mrg[f"{col}_hres"] - mrg[f"{col}_ai"]
            return mrg[f"{col}_ai"] - mrg[f"{col}_hres"]
        denom = mrg[f"{col}_hres"].replace(0.0, np.nan)
        if BASE == "pc":
            val = (mrg[f"{col}_hres"] - mrg[f"{col}_ai"]) / denom * 100.0
        else:
            val = (mrg[f"{col}_ai"] - mrg[f"{col}_hres"]) / denom * 100.0
        return val.to_numpy()

    # Collect values for unified y-lims
    base_values: list[float] = []
    diff_values: list[float] = []

    # ------------------------------- plotting ----------------------------------
    for j, (var, tail) in enumerate(col_specs):
        tail_txt = "(low extremes)" if tail == "lower" else "(high extremes)"
        axes[0, j].set_title(f"{var_label.get(var, var)} {tail_txt}", fontsize=12)

        sub = df[df["variable"] == var]

        # ---------- Row 1: base curves ----------
        for key in (hres_key, ai_key):
            lab = model_label(key)
            seg = sub[sub["model"] == key][["lead_time", BASE_UP]].dropna()
            if not seg.empty:
                axes[0, j].plot(
                    seg["lead_time"], seg[BASE_UP],
                    marker="o", ms=5, color=model_colors[key], label=lab
                )
                base_values.extend(seg[BASE_UP].to_numpy(dtype=float))
        if j == 0:
            axes[0, j].set_ylabel(BASE_UP)
        axes[0, j].grid(alpha=0.3)
        if j == n_cols - 1:
            handles = [
                Line2D([], [], marker="o", ls="-", ms=6, color=model_colors[hres_key], label=model_label(hres_key)),
                Line2D([], [], marker="o", ls="-", ms=6, color=model_colors[ai_key],   label=model_label(ai_key)),
            ]
            axes[0, j].legend(handles=handles, loc="best", frameon=False)

        # ---------- Row 2: differences ----------
        h_base = sub[sub["model"] == hres_key][["lead_time", BASE_UP]].rename(columns={BASE_UP: f"{BASE_UP}_hres"})
        a_base = sub[sub["model"] == ai_key  ][["lead_time", BASE_UP]].rename(columns={BASE_UP: f"{BASE_UP}_ai"})
        m_base = h_base.merge(a_base, on="lead_time").dropna()
        if not m_base.empty:
            d = _diff_base(m_base)
            axes[1, j].plot(
                m_base["lead_time"], d,
                color=model_colors[ai_key], linewidth=2.0, marker="o", ms=5,
                label=f"{BASE_UP} diff"
            )
            diff_values.extend([x for x in d if np.isfinite(x)])

        # Per-level differences (tail-appropriate)
        lvl_list = levels_low if tail == "lower" else levels_high

        # CHANGED: sort by distance from 0.5 so “more extreme” quantiles map to darker colors consistently
        lvl_list_sorted = sorted(lvl_list, key=lambda q: min(q, 1.0 - q))

        for lvl in lvl_list_sorted:
            colname = f"{metric_type}_{BASE_UP}_{lvl}"
            if colname not in df.columns:
                continue
            h_l = sub[sub["model"] == hres_key][["lead_time", colname]].rename(columns={colname: f"{colname}_hres"})
            a_l = sub[sub["model"] == ai_key  ][["lead_time", colname]].rename(columns={colname: f"{colname}_ai"})
            m_lvl = h_l.merge(a_l, on="lead_time").dropna()
            if m_lvl.empty:
                continue
            d = _diff_level(m_lvl, colname)

            # CHANGED: get symmetric color (e.g., 0.01 and 0.99 share the same color)
            colr = level_color(lvl)

            axes[1, j].plot(
                m_lvl["lead_time"], d,
                marker="o", ms=5, color=colr,
                label=f"{metric_type.upper()}={lvl:.2f} {'(low)' if tail=='lower' else '(high)'}"
            )
            diff_values.extend([x for x in d if np.isfinite(x)])

        if j == 0:
            axes[1, j].set_ylabel(f"{BASE_UP} difference" + (" [%]" if diff_type == "percent" else ""))
        axes[1, j].set_xlabel("Lead Time [d]")
        axes[1, j].grid(alpha=0.3)

        # Legends for row 2
        extra_low_idx = 2
        if j == n_cols - 1 or j == extra_low_idx:
            axes[1, j].legend(loc="best", frameon=False)

    # ---------- unify y-limits across each row ----------
    if BASE == "pcs":
        for j in range(n_cols):
            axes[0, j].set_ylim(0.0, 1.0)
    else:
        if base_values:
            bmin = float(np.nanmin(base_values))
            bmax = float(np.nanmax(base_values))
            pad = 0.05 * (bmax - bmin if bmax > bmin else 1.0)
            for j in range(n_cols):
                axes[0, j].set_ylim(bmin - pad, bmax + pad)

    if diff_values:
        dmin = float(np.nanmin(diff_values))
        dmax = float(np.nanmax(diff_values))
        pad = 0.05 * (dmax - dmin if dmax > dmin else 1.0)
        for j in range(n_cols):
            axes[1, j].set_ylim(dmin - pad, dmax + pad)

    ai_disp = "GC" if ai_model == "graphcast" else "PW"
    fig.suptitle(
        f"Performance differences: {BASE_UP} & {metric_type.upper()}_{BASE_UP} - "
        f"{ai_disp} vs HRES (Ground Truth: {obs_source.upper()})",
        fontsize=18
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_filename = (
        f"perf_diff_panel5_{BASE}_{metric_type}_{diff_type}_"
        f"{ai_model}_vs_hres_{obs_source}.png"
    )
    fig.savefig(os.path.join(PLOTS_DIR, out_filename), dpi=150, bbox_inches="tight")
    plt.show()

def plot_performance_diff_summary_panel(
    summary_df: pd.DataFrame,
    ai_model: str,                    # "graphcast" or "pangu"
    diff_type: str = "absolute"       # "absolute" or "percent"
) -> None:
    """
    4×5 summary panel of tail-weighted PCS differences across lead times with two spacer rows
    for centered section headers (IFS at top, ERA5 in the middle).
    """
    import matplotlib.gridspec as gridspec

    # --------------------------- layout knobs ----------------------------------
    FIG_WIDTH_PER_COL = 4.1
    HEIGHT_UNITS      = [0.09, 1.0, 1.0, 0.09, 1.0, 1.0]
    WSPACE            = 0.06
    HSPACE            = 0.24
    SUPTITLE_Y        = 0.985

    # ------------------------------ constants ----------------------------------
    VAR_LABEL = {
        "mean_sea_level_pressure": "MSLP",
        "2m_temperature":          "T2M",
        "10m_wind_speed":          "WS10",
    }
    COL_SPECS = [
        ("mean_sea_level_pressure", "lower"),
        ("mean_sea_level_pressure", "upper"),
        ("2m_temperature",          "lower"),
        ("2m_temperature",          "upper"),
        ("10m_wind_speed",          "upper"),
    ]
    BLOCK_IFS  = [("ifs",  "tw", "(tw)PCS diff"), ("ifs",  "qw", "(qw)PCS diff")]
    BLOCK_ERA5 = [("era5", "tw", "(tw)PCS diff"), ("era5", "qw", "(qw)PCS diff")]
    LEGEND_COLS = {2, 4}

    # --------------------------- helpers & styling ------------------------------
    def csv_model_key(obs: str, m: str) -> str:
        if obs == "ifs" and m in ("graphcast", "pangu"):
            return f"{m}_operational"
        return m

    fam = plt.get_cmap("Oranges")
    def level_palette(n: int) -> list:
        n = max(1, n)
        return [fam(x) for x in np.linspace(0.55, 0.9, n)]

    def _pcs_diff_abs(mrg, col_ai, col_hres):
        return (mrg[col_ai] - mrg[col_hres]).to_numpy()

    def _pcs_diff_pct(mrg, col_ai, col_hres):
        denom = mrg[col_hres].replace(0.0, np.nan)
        return ((mrg[col_ai] - mrg[col_hres]) / denom * 100.0).to_numpy()

    diff_fn = _pcs_diff_abs if diff_type == "absolute" else _pcs_diff_pct
    y_unit  = " [%]" if diff_type == "percent" else ""

    def _levels_for(obs: str, metric_type: str):
        """Discover (low, high) levels from columns for a given obs+metric_type."""
        prefix = f"{metric_type}_PCS_"
        df_obs = summary_df[summary_df["obs_source"] == obs]
        levels = sorted({
            float(col.split("_")[-1])
            for col in df_obs.columns if col.startswith(prefix)
        })
        return [x for x in levels if x < 0.5], [x for x in levels if x > 0.5]

    # ------------------------------ figure -------------------------------------
    n_cols = len(COL_SPECS)
    fig_width  = FIG_WIDTH_PER_COL * n_cols
    fig_height = 2.9 * sum(HEIGHT_UNITS)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs  = gridspec.GridSpec(
        nrows=6, ncols=n_cols, figure=fig,
        height_ratios=HEIGHT_UNITS, hspace=HSPACE, wspace=WSPACE
    )

    axes = [[None]*n_cols for _ in range(6)]
    for i in range(6):
        if i in (0, 3):  # spacer rows
            continue
        for j in range(n_cols):
            axes[i][j] = fig.add_subplot(gs[i, j])

    per_row_values = [[] for _ in range(6)]

    # ------------------------------ plotting -----------------------------------
    def _plot_row(ax_row_index: int, obs_source: str, metric_type: str, row_ylabel: str, put_col_headers: bool):
        ai_key, hres_key = csv_model_key(obs_source, ai_model), "hres"
        df_obs = summary_df[summary_df["obs_source"] == obs_source].copy()
        levels_low, levels_high = _levels_for(obs_source, metric_type)

        # ===== build a single symmetric color map per (obs_source, metric_type) =====
        # This ensures 0.01 and 0.99 share identical colors and ordering is consistent.
        all_levels = levels_low + levels_high
        sym_dists = sorted({round(min(q, 1.0 - q), 12) for q in all_levels if not np.isclose(q, 0.5)})
        colors_symmetric = level_palette(len(sym_dists))
        dist2color = dict(zip(sym_dists, colors_symmetric))

        def level_color(q: float):
            d = round(min(q, 1.0 - q), 12)
            return dist2color.get(d, (0.5, 0.5, 0.5))
        # ================================================================================

        for j, (var, tail) in enumerate(COL_SPECS):
            ax = axes[ax_row_index][j]

            if put_col_headers:
                tail_txt = " (low extremes)" if tail == "lower" else " (high extremes)"
                ax.set_title(f"{VAR_LABEL[var]}{tail_txt}", fontsize=12)

            sub = df_obs[df_obs["variable"] == var].copy()
            if sub.empty:
                ax.set_facecolor("lightgray")
            else:
                # Raw PCS diff (black line)
                if "PCS" in sub.columns:
                    a_raw = sub[sub["model"] == ai_key  ][["lead_time", "PCS"]].rename(columns={"PCS": "ai"})
                    h_raw = sub[sub["model"] == hres_key][["lead_time", "PCS"]].rename(columns={"PCS": "hres"})
                    m_raw = a_raw.merge(h_raw, on="lead_time").dropna()
                    if not m_raw.empty:
                        d_raw = diff_fn(m_raw, "ai", "hres")
                        ax.plot(m_raw["lead_time"], d_raw, color="black", linewidth=2, marker="o", ms=4, label="PCS diff")
                        per_row_values[ax_row_index].extend([x for x in d_raw if np.isfinite(x)])

                # Tail-appropriate TW/QW curves — use symmetric colors & consistent order
                lvl_list = levels_low if tail == "lower" else levels_high
                lvl_list_sorted = sorted(lvl_list, key=lambda q: min(q, 1.0 - q))  # order by extremeness

                for lvl in lvl_list_sorted:
                    col = f"{metric_type}_PCS_{lvl}"
                    if col not in sub.columns:
                        continue
                    a = sub[sub["model"] == ai_key  ][["lead_time", col]].rename(columns={col: "ai"})
                    h = sub[sub["model"] == hres_key][["lead_time", col]].rename(columns={col: "hres"})
                    m = a.merge(h, on="lead_time").dropna()
                    if m.empty:
                        continue
                    d = diff_fn(m, "ai", "hres")
                    ax.plot(
                        m["lead_time"], d,
                        marker="o", ms=5, color=level_color(lvl),  # symmetric color
                        label=f"{metric_type.upper()}={lvl:.2f} {'(low)' if tail=='lower' else '(high)'}"
                    )
                    per_row_values[ax_row_index].extend([x for x in d if np.isfinite(x)])

            # Axis labels/ticks
            if j == 0:
                ax.set_ylabel(f"{row_ylabel}{y_unit}")
            else:
                ax.set_ylabel("")
                ax.yaxis.set_tick_params(labelleft=False)

            if ax_row_index == 5:
                ax.set_xlabel("Lead Time [d]")
                ax.tick_params(axis="x", labelbottom=True)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            ax.grid(alpha=0.3)

            # Legend only on the first data row (cols 3 & 5)
            if put_col_headers and j in LEGEND_COLS:
                ax.legend(loc="best", frameon=False)

    # Top spacer with centered IFS header
    spacer_ifs = fig.add_subplot(gs[0, :])
    spacer_ifs.axis("off")
    spacer_ifs.text(0.5, 0.5, "Ground Truth = IFS", ha="center", va="center", fontsize=13, weight="bold")

    # IFS data rows (1,2)
    _plot_row(1, *BLOCK_IFS[0], put_col_headers=True)
    _plot_row(2, *BLOCK_IFS[1], put_col_headers=False)

    # Middle spacer with centered ERA5 header
    spacer_era5 = fig.add_subplot(gs[3, :])
    spacer_era5.axis("off")
    spacer_era5.text(0.5, 0.5, "Ground Truth = ERA5", ha="center", va="center", fontsize=13, weight="bold")

    # ERA5 data rows (4,5)
    _plot_row(4, *BLOCK_ERA5[0], put_col_headers=False)
    _plot_row(5, *BLOCK_ERA5[1], put_col_headers=False)

    # Unified y-lims per data row
    for i in (1, 2, 4, 5):
        vals = per_row_values[i]
        if not vals:
            continue
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
        for j in range(n_cols):
            axes[i][j].set_ylim(vmin - pad, vmax + pad)

    # Suptitle and margins
    ai_disp = "GC" if ai_model == "graphcast" else "PW"
    fig.suptitle(
        f"Summary of Performance Differences - {ai_disp} vs IFS-HRES",
        fontsize=16, y=SUPTITLE_Y
    )
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.07, top=0.955)

    out = f"perf_diff_summary_panel4x5_pcs_{ai_model}_{diff_type}.png"
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    plt.show()


def plot_gaussian_vs_tw_lines(
    *,
    t_level: float,
    obs_source: str = "era5",                # "era5" or "ifs"
    score: str = "pcs",                      # "pcs" or "pc"
    gaussian_dir: str = "score_data",
    baseline_dir: str = "score_data",
    lead_times: list[int] | None = None,
    # For labeling only (how σ was chosen when you computed gauss files):
    sigma_is_fraction: bool = True,
    gauss_sigma: float = 0.1
) -> None:
    """
    Line panel (rows = Baseline TW, Gaussian-CDF TW, Difference; cols = VARIABLES),
    showing curves across lead times for models in order: GC, PW, HRES.

    Assumes Gaussian CSVs contain:
        - "gauss_tw_pcs" or "gauss_tw_pc"
        - "indicator_tw_pcs" or "indicator_tw_pc" (baseline column name)
    """
    score = score.lower()
    if score not in ("pc","pcs"):
        raise ValueError("score must be 'pc' or 'pcs'.")

    # Fixed variable order (labels only; underlying var names unchanged)
    VARS_ORDERED = [
        "mean_sea_level_pressure",  # MSLP
        "2m_temperature",           # T2M
        "10m_wind_speed",           # WS20 (label only)
    ]

    # Resolve lead times
    LTS = lead_times if lead_times is not None else list(LEAD_TIMES)

    # Display names
    name_map = {
        "mean_sea_level_pressure": "MSLP",
        "2m_temperature":          "T2M",
        "10m_wind_speed":          "WS20",
    }

    # CSV model keys (IFS → operational suffix for AI)
    def csv_model_key(m: str) -> str:
        if obs_source == "ifs" and m in ("graphcast","pangu"):
            return f"{m}_operational"
        return m

    # Model order + colors
    model_order = ["graphcast", "pangu", "hres"]  # GC, PW, HRES
    color_map = plt.get_cmap("tab10")
    model_colors = {m: color_map(i % 10) for i, m in enumerate(model_order)}

    # Legend labels incl. −ERA5/−IFS suffix for GC/PW
    def model_label(m: str) -> str:
        if m == "graphcast":
            return f"GC-{'IFS' if obs_source=='ifs' else 'ERA5'}"
        if m == "pangu":
            return f"PW-{'IFS' if obs_source=='ifs' else 'ERA5'}"
        return "HRES"

    # Column names in Gaussian CSVs
    gauss_col = f"gauss_tw_{score}"
    base_col  = f"indicator_tw_{score}"

    # Load one var+lead Gaussian CSV and ensure baseline present
    def _load_gaussian(var: str, lt: int) -> pd.DataFrame:
        gpath = os.path.join(
            os.path.dirname(__file__),
            gaussian_dir,
            f"{var}_lead{lt}d_{obs_source}_gaussianTW.csv"
        )
        if not os.path.exists(gpath):
            return pd.DataFrame()
        df = pd.read_csv(gpath)
        df = df[(df["t_quantile"] == t_level)].copy()

        if base_col not in df.columns:
            bpath = os.path.join(
                os.path.dirname(__file__),
                baseline_dir,
                f"{var}_lead{lt}d_{obs_source}.csv"
            )
            if os.path.exists(bpath):
                bdf = pd.read_csv(bpath)
                bdf = bdf[(bdf["t_quantile"] == t_level)][["model","lat","lon", f"tw_{score}"]]
                bdf = bdf.rename(columns={f"tw_{score}": base_col})
                df = df.merge(bdf, on=["model","lat","lon"], how="left")
        return df

    # Cosine-latitude weighted mean via your helper
    def _lat_weighted(df: pd.DataFrame, col: str) -> float:
        if df.empty or col not in df.columns:
            return np.nan
        tmp = df[["lat","lon",col]].rename(columns={col: "val"}).dropna(subset=["val"])
        if tmp.empty:
            return np.nan
        tmp2 = tmp.rename(columns={"val": col})
        return compute_lat_weighted_mean(tmp2[["lat","lon",col]], col)

    # Collect panel data
    data = {var: {m: {"base": [], "gauss": [], "diff": []} for m in model_order} for var in VARS_ORDERED}
    all_score_vals = []  # for PC y-lims

    for var in VARS_ORDERED:
        for lt in LTS:
            df_lt = _load_gaussian(var, lt)
            if df_lt.empty:
                for m in model_order:
                    data[var][m]["base"].append(np.nan)
                    data[var][m]["gauss"].append(np.nan)
                    data[var][m]["diff"].append(np.nan)
                continue

            for m in model_order:
                mk = csv_model_key(m)
                sub = df_lt[df_lt["model"] == mk]
                base_mean  = _lat_weighted(sub, base_col)
                gauss_mean = _lat_weighted(sub, gauss_col)
                diff_mean  = (gauss_mean - base_mean) if (np.isfinite(gauss_mean) and np.isfinite(base_mean)) else np.nan

                data[var][m]["base"].append(base_mean)
                data[var][m]["gauss"].append(gauss_mean)
                data[var][m]["diff"].append(diff_mean)

                if score == "pc":
                    if np.isfinite(base_mean):  all_score_vals.append(base_mean)
                    if np.isfinite(gauss_mean): all_score_vals.append(gauss_mean)

    # ---- Figure: use tight_layout(rect=...) to reserve space for title/legend
    n_rows, n_cols = 3, len(VARS_ORDERED)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        constrained_layout=False  # important: avoid clashes with tight_layout
    )
    axes = np.atleast_2d(axes)

    # Row 1: Baseline TW (solid)
    for j, var in enumerate(VARS_ORDERED):
        ax = axes[0, j]
        for m in model_order:
            ax.plot(LTS, data[var][m]["base"], linestyle="-", marker="o", ms=5,
                    color=model_colors[m], label=model_label(m))
        ax.set_title(name_map.get(var, var))
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(f"TW_{score.upper()} (Indicator)")
        if score == "pcs":
            ax.set_ylim(0.0, 1.0)

    # Row 2: Gaussian-CDF TW (solid) with sigma info in ylabel
    sigma_tag = (f"σ={gauss_sigma}·std" if sigma_is_fraction else f"σ={gauss_sigma}")
    for j, var in enumerate(VARS_ORDERED):
        ax = axes[1, j]
        for m in model_order:
            ax.plot(LTS, data[var][m]["gauss"], linestyle="-", marker="o", ms=5,
                    color=model_colors[m], label=model_label(m))
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(f"TW_{score.upper()} (Gaussian, μ=t, {sigma_tag})")
        if score == "pcs":
            ax.set_ylim(0.0, 1.0)

    # Row 3: Difference (Gaussian − Indicator), dashed
    for j, var in enumerate(VARS_ORDERED):
        ax = axes[2, j]
        for m in model_order:
            ax.plot(LTS, data[var][m]["diff"], linestyle="--", marker="o", ms=5,
                    color=model_colors[m], label=model_label(m))
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(r'$\Delta$ score (Gaussian − Indicator)')
        ax.set_xlabel("Lead Time [d]")

    # Consistent y-lims for PC rows
    if score == "pc" and all_score_vals:
        ymin = float(np.nanmin(all_score_vals))
        ymax = float(np.nanmax(all_score_vals))
        pad  = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        for row in (0, 1):
            for j in range(n_cols):
                axes[row, j].set_ylim(ymin - pad, ymax + pad)

    # Legend at the bottom; build handles once
    handles, labels = [], []
    for m in model_order:
        h, = axes[1, -1].plot([], [], color=model_colors[m], marker="o", ls="-", ms=5)
        handles.append(h)
        labels.append(model_label(m))
    # Place legend *below* the axes region; rect below reserves space
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), frameon=False)

    # Title at the top
    fig.suptitle(
        f"TW_{score.upper()} with different weight functions with q={t_level:.2f} used for t - Ground Truth: {obs_source.upper()}",
        y=0.985, fontsize=16
    )

    # Reserve space for legend (bottom) and title (top)
    fig.tight_layout(rect=[0, 0.04, 1, 0.985])

    # Save and show
    out = f"lines_gaussTW_vs_TW_{score}_t{t_level:.2f}_{obs_source}_GC_PW_HRES.png"
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(PLOTS_DIR, out))
    plt.show()
    
    
def _sigma_tag(x: float) -> str:
    """Compact, filename-safe tag: 0.1 -> '0p1', 2.22e-16 -> '2p22e-16'."""
    s = f"{x:.6g}"
    return s.replace(".", "p")

def plot_gaussian_vs_tw_lines_eps(
    *,
    t_level: float,
    obs_source: str = "ifs",               # "era5" or "ifs"
    score: str = "pcs",                    # "pcs" or "pc"
    gaussian_dir: str = "score_data",
    baseline_dir: str = "score_data",
    lead_times: list[int] | None = None,
    # Select the exact Gaussian file variant to read:
    sigma_is_fraction: bool = True,        # corresponds to writer's 'mode' tag: 'frac' or 'abs'
    gauss_sigma: float = np.finfo(float).eps,
    fallback_sigma: float = 1.0
) -> None:
    """
    Line panel (rows = Indicator TW, Gaussian-CDF TW, Difference; cols = VARIABLES),
    across lead times, for models [GraphCast, Pangu, HRES].
    """
    score = score.lower()
    if score not in ("pc", "pcs"):
        raise ValueError("score must be 'pc' or 'pcs'.")

    VARS_ORDERED = [
        "mean_sea_level_pressure",
        "2m_temperature",
        "10m_wind_speed",
    ]
    LTS = lead_times if lead_times is not None else list(LEAD_TIMES)

    name_map = {
        "mean_sea_level_pressure": "MSLP",
        "2m_temperature": "T2M",
        "10m_wind_speed": "WS10",
    }

    def csv_model_key(m: str) -> str:
        # Append "_operational" for AI models vs IFS, consistent with your CSVs
        if obs_source == "ifs" and m in ("graphcast", "pangu"):
            return f"{m}_operational"
        return m

    # Model order + aesthetics
    model_order = ["graphcast", "pangu", "hres"]
    color_map   = plt.get_cmap("tab10")
    model_colors = {m: color_map(i % 10) for i, m in enumerate(model_order)}

    def model_label(m: str) -> str:
        if m == "graphcast":
            return f"GC-{'IFS' if obs_source=='ifs' else 'ERA5'}"
        if m == "pangu":
            return f"PW-{'IFS' if obs_source=='ifs' else 'ERA5'}"
        return "HRES"


    mode_tag = "frac" if sigma_is_fraction else "abs"
    sig_tag  = _sigma_tag(gauss_sigma)
    fb_tag   = _sigma_tag(fallback_sigma)

    gauss_col = f"gauss_tw_{score}"
    base_col  = f"baseline_tw_{score}"  # matches writer's column names

    def _gauss_path(var: str, lt: int) -> str:
  
        modern = os.path.join(
            os.path.dirname(__file__),
            gaussian_dir,
            f"{var}_lead{lt}d_{obs_source}_gaussianTW_{mode_tag}_sig{sig_tag}_fb{fb_tag}.csv"
        )
        if os.path.exists(modern):
            return modern
        legacy = os.path.join(
            os.path.dirname(__file__),
            gaussian_dir,
            f"{var}_lead{lt}d_{obs_source}_gaussianTW.csv"
        )
        return legacy

    def _load_gaussian(var: str, lt: int) -> pd.DataFrame:
        """Load Gaussian CSV for a given var/lead and attach indicator baseline if needed."""
        gpath = _gauss_path(var, lt)
        if not os.path.exists(gpath):
            return pd.DataFrame()
        df = pd.read_csv(gpath)
        df = df[df["t_quantile"] == t_level].copy()

        # If file doesn't carry the baseline column (old runs), merge it from baseline CSV.
        if base_col not in df.columns:
            bpath = os.path.join(
                os.path.dirname(__file__),
                baseline_dir,
                f"{var}_lead{lt}d_{obs_source}.csv"
            )
            if os.path.exists(bpath):
                bdf = pd.read_csv(bpath)
                bdf = bdf[bdf["t_quantile"] == t_level][["model", "lat", "lon", f"tw_{score}"]]
                bdf = bdf.rename(columns={f"tw_{score}": base_col})
                df = df.merge(bdf, on=["model", "lat", "lon"], how="left")
        return df

    def _lat_weighted(df: pd.DataFrame, col: str) -> float:
        """Cosine-latitude weighted mean of 'col' using WB2’s _spatial_average."""
        if df.empty or col not in df.columns:
            return np.nan
        tmp = df[["lat", "lon", col]].rename(columns={col: "val"}).dropna(subset=["val"])
        if tmp.empty:
            return np.nan
        tmp2 = tmp.rename(columns={"val": col})
        return compute_lat_weighted_mean(tmp2[["lat", "lon", col]], col)

    # Collect panel values
    data = {v: {m: {"base": [], "gauss": [], "diff": []} for m in model_order} for v in VARS_ORDERED}
    all_score_vals = []

    for var in VARS_ORDERED:
        for lt in LTS:
            df_lt = _load_gaussian(var, lt)
            if df_lt.empty:
                for m in model_order:
                    data[var][m]["base"].append(np.nan)
                    data[var][m]["gauss"].append(np.nan)
                    data[var][m]["diff"].append(np.nan)
                continue

            for m in model_order:
                mk = csv_model_key(m)
                sub = df_lt[df_lt["model"] == mk]
                base_mean  = _lat_weighted(sub, base_col)
                gauss_mean = _lat_weighted(sub, gauss_col)
                diff_mean  = (gauss_mean - base_mean) if (np.isfinite(gauss_mean) and np.isfinite(base_mean)) else np.nan

                data[var][m]["base"].append(base_mean)
                data[var][m]["gauss"].append(gauss_mean)
                data[var][m]["diff"].append(diff_mean)

                if score == "pc":
                    if np.isfinite(base_mean):  all_score_vals.append(base_mean)
                    if np.isfinite(gauss_mean): all_score_vals.append(gauss_mean)

    # Figure
    n_rows, n_cols = 3, len(VARS_ORDERED)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.0 * n_rows),
        constrained_layout=False
    )
    axes = np.atleast_2d(axes)

    # Row 1: Indicator TW
    for j, var in enumerate(VARS_ORDERED):
        ax = axes[0, j]
        for m in model_order:
            ax.plot(LTS, data[var][m]["base"], linestyle="-", marker="o", ms=5,
                    color=model_colors[m], label=model_label(m))
        ax.set_title(name_map.get(var, var))
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(f"TW_{score.upper()} (Indicator)")
        if score == "pcs":
            ax.set_ylim(0.0, 1.0)

    # Row 2: Gaussian TW
    sig_lbl = f"σ={gauss_sigma}·std" if sigma_is_fraction else f"σ={gauss_sigma}"
    fb_lbl  = f", fb={fallback_sigma}"
    for j, var in enumerate(VARS_ORDERED):
        ax = axes[1, j]
        for m in model_order:
            ax.plot(LTS, data[var][m]["gauss"], linestyle="-", marker="o", ms=5,
                    color=model_colors[m], label=model_label(m))
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(f"TW_{score.upper()} \n (Gaussian, μ=t, σ = ~2.22e-16 * std)")
        if score == "pcs":
            ax.set_ylim(0.0, 1.0)

    # Row 3: Difference (Gaussian − Indicator)
    for j, var in enumerate(VARS_ORDERED):
        ax = axes[2, j]
        for m in model_order:
            ax.plot(LTS, data[var][m]["diff"], linestyle="--", marker="o", ms=5,
                    color=model_colors[m], label=model_label(m))
        ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
        ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel(r'$\Delta$ score (Gaussian − Indicator)')
        ax.set_xlabel("Lead Time [d]")

    # Harmonize y-lims for PC rows if needed
    if score == "pc" and all_score_vals:
        ymin = float(np.nanmin(all_score_vals))
        ymax = float(np.nanmax(all_score_vals))
        pad  = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        for row in (0, 1):
            for j in range(n_cols):
                axes[row, j].set_ylim(ymin - pad, ymax + pad)

    # Legend (bottom)
    handles, labels = [], []
    for m in model_order:
        h, = axes[1, -1].plot([], [], color=model_colors[m], marker="o", ls="-", ms=5)
        handles.append(h); labels.append(model_label(m))
    fig.legend(handles, labels, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.01), frameon=False)

    # Title and layout
    fig.suptitle(
        f"TW_{score.upper()} with q={t_level:.2f} used for t - {obs_source.upper()} - "
        f"weight: {'Gaussian (μ=t)' if True else ''} vs Indicator",
        y=0.985, fontsize=16
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.985])

    out = (
        f"lines_gaussTW_vs_TW_{score}_t{t_level:.2f}_{obs_source}_"
        f"{mode_tag}_sig{sig_tag}_fb{fb_tag}_GC_PW_HRES.png"
    )
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(PLOTS_DIR, out))
    plt.show()
    
def plot_extreme_pcs_5x5_map(
    *,
    obs_source: str = "era5",           # "era5" or "ifs"
    forecast_model: str = "graphcast",  # "graphcast", "pangu", "hres"
    t_low: float = 0.05,                # quantile for "low extremes"
    t_high: float = 0.95,               # quantile for "high extremes"
    projection: ccrs.CRS = ccrs.Robinson(),
    cmap: str = "viridis"
) -> None:
    """
    5 (lead times) × 5 (extreme columns) panel of TW_PCS maps for a chosen model.
    Columns: [MSLP↓, MSLP↑, T2M↓, T2M↑, WS10↑]
    Rows:    Lead times in LEAD_TIMES

    Notes
    -----
    • Color scale is fixed to [0, 1] (PCS convention used elsewhere).
    • For IFS ground truth, GraphCast/Pangu use the '_operational' suffix (kept consistent).
    • If a CSV is missing or empty for a cell, that subplot is greyed with 'missing'/'no data'.
    """
    # Fixed column spec: (variable, tail, display title)
    COL_SPECS = [
        ("mean_sea_level_pressure", "lower", "MSLP (low)"),
        ("mean_sea_level_pressure", "upper", "MSLP (high)"),
        ("2m_temperature",         "lower", "T2M (cold)"),
        ("2m_temperature",         "upper", "T2M (hot)"),
        ("10m_wind_speed",         "upper", "WS10 (high)"),
    ]

    # Display names
    var_label = {
        "mean_sea_level_pressure": "MSLP",
        "2m_temperature": "T2M",
        "10m_wind_speed": "WS10",
    }
    model_titles = {
        "hres": "IFS-HRES",
        "pangu": "Pangu",
        "graphcast": "GraphCast",
    }

    # CSV model key (operational suffix when comparing to IFS)
    if obs_source == "ifs" and forecast_model in ("graphcast", "pangu"):
        model_key = f"{forecast_model}_operational"
    else:
        model_key = forecast_model

    # Fixed normalization 0..1 for PCS-like maps
    PCS_NORM = mcolors.Normalize(vmin=0.0, vmax=1.0)

    # Figure
    n_rows, n_cols = len(LEAD_TIMES), len(COL_SPECS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        subplot_kw={"projection": projection},
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    first_mesh = None

    # Loop rows (lead times) × columns (extreme spec)
    for i, lt in enumerate(LEAD_TIMES):
        for j, (var, tail, col_title) in enumerate(COL_SPECS):
            ax = axes[i, j]
            csv_path = os.path.join(SCORE_DATA_DIR, f"{var}_lead{lt}d_{obs_source}.csv")
            if not os.path.exists(csv_path):
                ax.set_facecolor("lightgray")
                ax.coastlines()
                ax.set_title("missing")
                continue

            df = pd.read_csv(csv_path)
            if df.empty or "tw_pcs" not in df.columns:
                ax.set_facecolor("lightgray")
                ax.coastlines()
                ax.set_title("no data")
                continue

            # Select model and tail-appropriate quantile
            level = t_low if tail == "lower" else t_high
            sub = df[(df["model"] == model_key) & (~df["tw_pcs"].isna())]
            sub = sub[sub["t_quantile"] == level]

            if sub.empty:
                ax.set_facecolor("lightgray")
                ax.coastlines()
                ax.set_title("no data")
                continue

            # Average duplicates and pivot to 2D grid
            grid = (
                sub.groupby(["lat", "lon"])["tw_pcs"]
                .mean()
                .reset_index()
                .pivot(index="lat", columns="lon", values="tw_pcs")
            )

            if grid.isna().all().all():
                ax.set_facecolor("lightgray")
                ax.coastlines()
                ax.set_title("no data")
                continue

            lats = grid.index.values
            lons = grid.columns.values
            lon_g, lat_g = np.meshgrid(lons, lats, indexing="xy")

            mesh = ax.pcolormesh(
                lon_g, lat_g, grid.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=PCS_NORM,
                shading="nearest",
            )
            if first_mesh is None:
                first_mesh = mesh

            ax.coastlines(linewidth=0.5)

            # Column headers on top row
            if i == 0:
                ax.set_title(col_title, fontsize=12)
            # Row labels on left column
            if j == 0:
                ax.text(
                    -0.1, 0.5,
                    f"Lead {lt}d",
                    transform=ax.transAxes,
                    va="center", ha="right",
                    rotation=90, fontsize=10
                )

    if first_mesh is None:
        raise RuntimeError("No data available to render the 5×5 map panel.")

    # Title and colorbar
    mt = model_titles.get(forecast_model, forecast_model)
    if obs_source == "ifs" and forecast_model in ("pangu", "graphcast"):
        mt += "-Operational"

    fig.suptitle(
        f"TW_PCS at low/high extremes — {mt} vs {obs_source.upper()}",
        y=0.99, fontsize=16
    )
    cbar = fig.colorbar(
        first_mesh, ax=axes.ravel().tolist(),
        orientation="horizontal", fraction=0.05, pad=0.02
    )
    cbar.set_label("TW_PCS", fontsize=12)

    out = (
        f"map_5x5_tw_pcs_{forecast_model}_vs_{obs_source}"
        f"_tlow{t_low:.2f}_thigh{t_high:.2f}.png"
    )
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    plt.show()

def _ensure_dir(p: str) -> None:
    """Create directory if it does not exist."""
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------


if __name__ == "__main__":

    # Panels
    RUN_PANEL_AVERAGED_GRIDPOINTS = False
    RUN_PANEL_SENSITIVITY = False
    RUN_PERF_DIFF_PANEL = False
    RUN_PERF_DIFF_SUMMARY_PANEL = False

    RUN_GAUSSIAN_VS_TW_PANEL = False
    RUN_GAUSSIAN_EPS_VS_TW_PANEL = False

    # Maps
    RUN_THRESHOLD_MAP = False
    THRESHOLD_LEVEL = 0.05
    RUN_ERA5_CLIM_PCS_HEATMAP = True
    RUN_3_SCORES_HEATMAP = False
    RUN_FIXED_SCORE_HEATMAP = False

    # Data
    LOAD_SUMMARY = False

    if LOAD_SUMMARY:
        summary_df = load_score_summary()
    else:
        csv_path = os.path.join(SCORE_DATA_DIR, "score_summary.csv")
        summary_df = pd.read_csv(csv_path)

    # Map plotting
    if RUN_THRESHOLD_MAP:
        plot_threshold_map(
            THRESHOLD_LEVEL,
            obs_source= "ifs", #or era5
        )

    if RUN_FIXED_SCORE_HEATMAP:
        plot_spatial_map(
            metric_base="qw_pcs",
            level=0.95,
            obs_source="era5",
            forecast_model="graphcast",
        )
        plot_extreme_pcs_5x5_map(
            obs_source="ifs",
            forecast_model="graphcast",
            t_low=0.05,
            t_high=0.95,
        )

    if RUN_3_SCORES_HEATMAP:
        plot_pcs_twq_map(
            lead_time=5,
            t_level=0.95,
            q_level=0.95,
            obs_source="era5",
            forecast_model="graphcast",
        )

    if RUN_ERA5_CLIM_PCS_HEATMAP:
        climatological_pcs_heatmap(
            metric_base="tw_pcs",
            obs_source="era5",
            forecast_model="graphcast",
        )

    # Panels plotting
    if RUN_PANEL_AVERAGED_GRIDPOINTS:
        plot_model_panel(
            summary_df=summary_df,
            obs_source="ifs",
            t_level=0.95,
            q_level=0.95,
            base_metric="pcs",
        )

    if RUN_PANEL_SENSITIVITY:
        plot_sensitivity_panel(
            summary_df=summary_df,
            obs_source="era5",
            base_metric="pcs",
            weight_type="tw",
        )

    if RUN_PERF_DIFF_PANEL:
        plot_performance_diff_panel(
            summary_df=summary_df,
            obs_source="ifs",
            ai_model="graphcast",
            base_metric="pcs",
            metric_type="tw",
            diff_type="absolute",
        )

    if RUN_PERF_DIFF_SUMMARY_PANEL:
        plot_performance_diff_summary_panel(
            summary_df=summary_df,
            ai_model="graphcast",
            diff_type="absolute",
        )

    if RUN_GAUSSIAN_VS_TW_PANEL:
        plot_gaussian_vs_tw_lines(
            t_level=0.95,
            obs_source="ifs",
            score="pcs",
        )

    if RUN_GAUSSIAN_EPS_VS_TW_PANEL:
        plot_gaussian_vs_tw_lines_eps(
            t_level=0.95,
            obs_source="ifs",
            score="pcs",
            gaussian_dir="score_data",
            baseline_dir="score_data",
            lead_times=[1, 3, 5, 7, 10],
            sigma_is_fraction=True,
            gauss_sigma=np.finfo(float).eps,
            fallback_sigma=1.0,
        )

    # Example: single call
    plot_climatology_ref_pcs_3x5(
        metric_base="tw_pcs",
        level_low=0.05,
        level_high=0.95,
        obs_source="era5",
    )