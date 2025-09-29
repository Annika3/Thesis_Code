# block_permutation_test.py

import os
import re
import zlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import dask
from dask.diagnostics import ProgressBar

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE_DIR = os.path.join(HERE, "score_data")
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

RAW_ERA5 = os.path.join(SCORE_DIR, "raw_scores_for_permutation_era5.zarr")
RAW_IFS  = os.path.join(SCORE_DIR, "raw_scores_for_permutation_ifs.zarr")

LEAD_TIMES = [np.timedelta64(d, "D") for d in (1, 3, 5, 7, 10)]
VARIABLES_ALL = ["mean_sea_level_pressure", "2m_temperature", "10m_wind_speed"]
TITLES_ALL    = ["MSLP", "T2M", "WS10"]

# which tails to evaluate per variable when using tail metrics (TW/QW)
VAR_TO_TAILS = {
    "mean_sea_level_pressure": ("low", "high"),
    "2m_temperature":          ("low", "high"),
    "10m_wind_speed":          ("high",),
}

# ---------------------------------------------------------------------
# Core stats
# ---------------------------------------------------------------------

def block_permutation_test(
    score_a: np.ndarray,
    score_b: np.ndarray,
    seed_offset: int,
    block_length: int,
    *,
    b: int = 1000,
    root_seed: int = 42,
) -> np.float32:
    """
    One-sided block permutation on 1D arrays.
    H0: mean(sign * (a-b)) <= observed mean(a-b)
    Smaller p ⇒ 'a' significantly better than 'b' for "smaller is better" metrics.
    """
    if score_a.shape != score_b.shape or score_a.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays of identical length")

    d = score_a - score_b
    n = d.size
    d_mean = d.mean()

    n_blocks = int(np.ceil((n + block_length - 1) / block_length))
    rng = np.random.default_rng(root_seed + int(seed_offset))
    block_signs = rng.choice((-1, 1), size=(b, n_blocks), replace=True)

    offsets = np.arange(b, dtype=int) % block_length
    block_indices = (np.arange(n)[None, :] + offsets[:, None]) // block_length  # (b, n)

    signs = block_signs[np.arange(b)[:, None], block_indices]  # (b, n)
    m = (signs * d[None, :]).mean(axis=1)
    p = np.mean(m <= d_mean)  # one-sided
    return p.astype(np.float32)

def _lead_days_int(lead_coord: xr.DataArray) -> xr.DataArray:
    """Return integer lead days regardless of encoding."""
    if np.issubdtype(lead_coord.dtype, np.timedelta64):
        return (lead_coord / np.timedelta64(1, "D")).astype("int64")
    units = str(lead_coord.attrs.get("units", "")).lower()
    if "hour" in units:
        return (lead_coord.astype("int64") // 24).astype("int64")
    return lead_coord.astype("int64")

def run_block_permutation(
    raw_zarr_path: str,
    model_a: str,
    model_b: str,
    metric: str,
    variable: str,
    *,
    b: int = 1000,
    root_seed: int = 42,
    compute: bool = True,
    show_progress: bool = True,
    scheduler: str = "threads",
    num_workers: int | None = None,
) -> xr.DataArray:
    """
    Grid-wise p-values for two models and a metric for every (lon,lat,lead).
    If compute=False, returns dask-lazy DataArray.
    """
    ds = xr.open_zarr(raw_zarr_path, decode_timedelta=True)

    if metric not in ds:
        raise KeyError(
            f"metric '{metric}' not found in {raw_zarr_path}. "
            f"Available metrics: {list(ds.data_vars)}"
        )

    model_coord = ds[metric].coords.get("model", None)
    if model_coord is not None and np.issubdtype(model_coord.dtype, np.bytes_):
        ds = ds.assign_coords(model=model_coord.astype(str))

    da_a = ds[metric].sel(model=model_a, variable=variable).chunk(dict(time=-1))
    da_b = ds[metric].sel(model=model_b, variable=variable).chunk(dict(time=-1))

    lon = da_a["longitude"]
    lat = da_a["latitude"]
    leads = da_a["prediction_timedelta"]

    seed_vals = np.arange(lon.size * lat.size * leads.size, dtype="uint32")
    seed_da = xr.DataArray(
        seed_vals.reshape(lon.size, lat.size, leads.size),
        dims=("longitude", "latitude", "prediction_timedelta"),
        coords={"longitude": lon, "latitude": lat, "prediction_timedelta": leads},
        name="seed_offset",
    )

    lead_days = _lead_days_int(leads)
    block_length_da = (lead_days * 2).rename("block_length").broadcast_like(seed_da)

    p_da_lazy = xr.apply_ufunc(
        block_permutation_test,
        da_a, da_b, seed_da, block_length_da,
        input_core_dims=[["time"], ["time"], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.float32],
        kwargs={"b": b, "root_seed": root_seed},
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).assign_coords(
        longitude=lon, latitude=lat, prediction_timedelta=leads
    ).rename("p_value")

    if not compute:
        return p_da_lazy

    with dask.config.set(scheduler=scheduler, num_workers=num_workers):
        if show_progress:
            with ProgressBar():
                return p_da_lazy.compute()
        return p_da_lazy.compute()

# ---------------------------------------------------------------------
# Metric parsing helpers
# ---------------------------------------------------------------------

def _parse_metric_base(metric_or_base: str) -> tuple[str, float | None]:
    """
    Parse metric string. Returns (base_without_q, q_value_if_present).
    'qw_pc_0.9' -> ('qw_pc', 0.9),  'tw_pc' -> ('tw_pc', None), 'pc' -> ('pc', None)
    """
    m = re.match(r"^(.*?)(?:_([01](?:\.\d+)?))?$", metric_or_base)
    if not m:
        return metric_or_base, None
    base, q = m.group(1), m.group(2)
    return base, (float(q) if q is not None else None)

def _metric_name_for_tail(base: str, q: float | None) -> str:
    """Build metric var name from base and quantile q."""
    if q is None:
        return base
    q_str = f"{q:.2f}".rstrip("0").rstrip(".")
    return f"{base}_{q_str}"

# ---------------------------------------------------------------------
# Plots: styled p-value boxplots (ERA5 top, IFS bottom)
# ---------------------------------------------------------------------

def plot_p_value_boxplots_styled(
    p_vs_era5_by_key: dict[str, dict[str, xr.DataArray]],
    p_vs_ifs_by_key: dict[str, dict[str, xr.DataArray]],
    keys: list[str],
    titles: list[str],
    lead_times: list[np.timedelta64],
    output_file: str,
    metric_label: str,
) -> None:
    """
    Styled boxplots of p-values with ERA5 (top) and IFS Analysis (bottom).
    'keys' and 'titles' are parallel lists; each key is 'var:tail' (or ':all').
    """
    def _vals_for_lead(p_da: xr.DataArray, lt: np.timedelta64) -> np.ndarray:
        coord = p_da["prediction_timedelta"]
        if np.issubdtype(coord.dtype, np.timedelta64):
            key = np.timedelta64(int(lt / np.timedelta64(1, "D")), "D")
        else:
            units = str(coord.attrs.get("units", "")).lower()
            if "hour" in units:
                key = int(lt / np.timedelta64(1, "D")) * 24
            else:
                key = int(lt / np.timedelta64(1, "D"))
        try:
            sel = p_da.sel(prediction_timedelta=key)
        except Exception:
            sel = p_da.sel(prediction_timedelta=key, method="nearest")
        return sel.values.flatten()

    fig, axes = plt.subplots(2, len(keys), figsize=(4.6 * len(keys), 10))

    # ERA5 (top row)
    for ax, k, title in zip(axes[0, :], keys, titles):
        positions_graphcast_pangu, positions_graphcast_hres, positions_pangu_hres = [], [], []
        data_graphcast_pangu, data_graphcast_hres, data_pangu_hres = [], [], []
        x_ticks = []
        offset = 0.3

        for lt in lead_times:
            lt_days = int(lt / np.timedelta64(1, 'D'))
            x_ticks.append(lt_days)
            d1 = _vals_for_lead(p_vs_era5_by_key[k]["GC_vs_PW"], lt)
            d2 = _vals_for_lead(p_vs_era5_by_key[k]["GC_vs_HRES"], lt)
            d3 = _vals_for_lead(p_vs_era5_by_key[k]["PW_vs_HRES"], lt)

            data_graphcast_pangu.append(d1)
            data_graphcast_hres.append(d2)
            data_pangu_hres.append(d3)

            positions_graphcast_pangu.append(lt_days - offset)
            positions_graphcast_hres.append(lt_days)
            positions_pangu_hres.append(lt_days + offset)

        boxplot_kwargs = dict(
            widths=0.2,
            patch_artist=True,
            medianprops=dict(color='black'),
            showfliers=False
        )
        ax.boxplot(data_graphcast_pangu, positions=positions_graphcast_pangu, boxprops=dict(facecolor='tab:green'), **boxplot_kwargs)
        ax.boxplot(data_graphcast_hres,  positions=positions_graphcast_hres,  boxprops=dict(facecolor='tab:blue'),  **boxplot_kwargs)
        ax.boxplot(data_pangu_hres,      positions=positions_pangu_hres,      boxprops=dict(facecolor='tab:orange'), **boxplot_kwargs)

        ax.set_xlabel('Lead Time [d]', fontsize=18)
        ax.set_title(title, fontsize=22)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis='both', which='major', labelsize=17)
        ax.set_xticklabels([str(x) for x in x_ticks])
        if "MSLP" in title:
            ax.set_ylabel('p-Value', fontsize=18)

    # IFS (bottom row)
    for ax, k, title in zip(axes[1, :], keys, titles):
        positions_graphcast_pangu, positions_graphcast_hres, positions_pangu_hres = [], [], []
        data_graphcast_pangu, data_graphcast_hres, data_pangu_hres = [], [], []
        x_ticks = []
        offset = 0.3

        for lt in lead_times:
            lt_days = int(lt / np.timedelta64(1, 'D'))
            x_ticks.append(lt_days)
            d1 = _vals_for_lead(p_vs_ifs_by_key[k]["GC_vs_PW"], lt)
            d2 = _vals_for_lead(p_vs_ifs_by_key[k]["GC_vs_HRES"], lt)
            d3 = _vals_for_lead(p_vs_ifs_by_key[k]["PW_vs_HRES"], lt)

            data_graphcast_pangu.append(d1)
            data_graphcast_hres.append(d2)
            data_pangu_hres.append(d3)

            positions_graphcast_pangu.append(lt_days - offset)
            positions_graphcast_hres.append(lt_days)
            positions_pangu_hres.append(lt_days + offset)

        boxplot_kwargs = dict(
            widths=0.2,
            patch_artist=True,
            medianprops=dict(color='black'),
            showfliers=False
        )
        ax.boxplot(data_graphcast_pangu, positions=positions_graphcast_pangu, boxprops=dict(facecolor='tab:green'), **boxplot_kwargs)
        ax.boxplot(data_graphcast_hres,  positions=positions_graphcast_hres,  boxprops=dict(facecolor='tab:blue'),  **boxplot_kwargs)
        ax.boxplot(data_pangu_hres,      positions=positions_pangu_hres,      boxprops=dict(facecolor='tab:orange'), **boxplot_kwargs)

        ax.set_xlabel('Lead Time [d]', fontsize=18)
        ax.set_title(title, fontsize=22)
        ax.set_xticks(x_ticks)
        ax.tick_params(axis="both", which="major", labelsize=17)
        ax.set_xticklabels([str(x) for x in x_ticks])
        if "MSLP" in title:
            ax.set_ylabel('p-Value', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    # Row titles with metric (+ extremes label)
    row1_left = axes[0, 0].get_position().x0
    row1_right = axes[0, -1].get_position().x1
    row1_top = axes[0, 0].get_position().y1
    row1_center_x = (row1_left + row1_right) / 2

    row2_left = axes[1, 0].get_position().x0
    row2_right = axes[1, -1].get_position().x1
    row2_top = axes[1, 0].get_position().y1
    row2_center_x = (row2_left + row2_right) / 2

    fig = plt.gcf()
    fig.text(row1_center_x, row1_top + 0.04, f"Ground Truth: ERA5 - Metric: {metric_label}",
             ha='center', va='bottom', fontsize=22)
    fig.text(row2_center_x, row2_top + 0.04, f"Ground Truth: IFS Analysis - Metric: {metric_label}",
             ha='center', va='bottom', fontsize=22)

    legend_handles = [
        plt.Rectangle((0,0),1,1, facecolor='tab:green', label='GC vs PW'),
        plt.Rectangle((0,0),1,1, facecolor='tab:blue',  label='GC vs HRES'),
        plt.Rectangle((0,0),1,1, facecolor='tab:orange', label='PW vs HRES'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.075), fontsize=22)

    metric_safe = metric_label.replace(" ", "_").replace("(", "").replace(")", "")
    base, ext = os.path.splitext(output_file)
    output_file = f"{base}_{metric_safe}{ext}"
    plt.savefig(output_file, facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------------
# Regional time-series permutation + scorecards (significance-aware)
# ---------------------------------------------------------------------

def run_block_permutation_regional(
    *,
    timeseries_csv: str,          # e.g. f"{SCORE_DIR}/regional_timeseries_{obs_source}.csv"
    obs_source: str,              # {"era5","ifs"}
    region: str,
    variable: str,
    lead_time_d: int,
    model_a: str,
    model_b: str,
    metric_col: str = "pc",
    b: int = 1000,
    root_seed: int = 42,
) -> float:
    """
    Single regional p-value comparing model_a vs model_b on the time-resolved regional series.
    """
    df = pd.read_csv(timeseries_csv, parse_dates=["time"])
    need = {"obs_source","region","variable","lead_time_d","model","time", metric_col}
    missing = need.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {timeseries_csv}: {sorted(missing)}")

    base = df[
        (df["obs_source"] == obs_source) &
        (df["region"] == region) &
        (df["variable"] == variable) &
        (df["lead_time_d"] == int(lead_time_d)) &
        (df["model"].isin([model_a, model_b]))
    ][["time","model",metric_col]].dropna()

    if base.empty or base["model"].nunique() < 2:
        return float("nan")

    wide = base.pivot_table(index="time", columns="model", values=metric_col, aggfunc="mean")
    if model_a not in wide or model_b not in wide:
        return float("nan")

    ab = wide[[model_a, model_b]].dropna()
    if ab.shape[0] < 10:
        return float("nan")

    key = f"{obs_source}|{region}|{variable}|{lead_time_d}|{model_a}|{model_b}|{metric_col}"
    seed_offset = int(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)

    block_length = max(1, 2 * int(lead_time_d))

    p = float(block_permutation_test(
        ab[model_a].to_numpy(np.float64),
        ab[model_b].to_numpy(np.float64),
        seed_offset=seed_offset,
        block_length=block_length,
        b=b,
        root_seed=root_seed,
    ))
    return p

def plot_regional_scorecard_4x5(
    *,
    score_dir: str = SCORE_DIR,
    obs_source: str = "era5",          # {"era5","ifs"}
    metric: str = "pcs",               # {"pc","pcs"} – PC smaller-better, PCS larger-better
    figsize: tuple = (14, 16),
    savepath: str | None = None,
    # --- significance knobs ---
    timeseries_csv: str | None = None, # e.g. f"{score_dir}/regional_timeseries_{obs_source}.csv"
    alpha: float = 0.05,
    b_perm: int = 1000,
    root_seed: int = 42,
    fair_mix: float = 0.45,            # blend with white for non-significant tiles (0..1)
    compute_significance: bool = True,
):
    """
    4×5 regional scorecard of best model per (region × lead), with optional significance
    shading using block permutation tests on regional time series.

    Columns: [MSLP low, MSLP high, T2M low, T2M high, WS10 high]
    Rows:    [PC (unweighted), 10% extremes, 5% extremes, 1% extremes]
    """
    from matplotlib.colors import ListedColormap

    assert obs_source in {"era5", "ifs"}
    assert metric in {"pc", "pcs"}

    # Column/row specs
    col_specs = [
        ("mean_sea_level_pressure", "MSLP low",  "lower"),
        ("mean_sea_level_pressure", "MSLP high", "upper"),
        ("2m_temperature",          "T2M low",   "lower"),
        ("2m_temperature",          "T2M high",  "upper"),
        ("10m_wind_speed",          "WS10 high", "upper"),
    ]
    lead_days = [1, 3, 5, 7, 10]

    # Model keys per obs source
    if obs_source == "era5":
        models_csv = ["graphcast", "pangu", "hres"]
        group_label = {"graphcast": "GC", "pangu": "PW", "hres": "HRES"}
        leg_gc = "GC-ERA5"; leg_pw = "PW-ERA5"
        label_to_models = {"GC": ["graphcast"], "PW": ["pangu"], "HRES": ["hres"]}
    else:
        models_csv = ["graphcast_operational", "pangu_operational", "hres"]
        group_label = {"graphcast_operational": "GC", "pangu_operational": "PW", "hres": "HRES"}
        leg_gc = "GC-IFS";  leg_pw = "PW-IFS"
        label_to_models = {"GC": ["graphcast_operational"], "PW": ["pangu_operational"], "HRES": ["hres"]}

    # Colors
    color_for = {"GC": "#1f77b4", "PW": "#ff7f0e", "HRES": "#2ca02c"}
    cmap = ListedColormap([color_for["GC"], color_for["PW"], color_for["HRES"], "#e5e7eb"])

    def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
        s = hex_color.lstrip("#")
        return (int(s[0:2], 16)/255.0, int(s[2:4], 16)/255.0, int(s[4:6], 16)/255.0)

    def _blend_with_white(hex_color: str, t: float) -> tuple[float, float, float]:
        r, g, b = _hex_to_rgb01(hex_color)
        return (1.0 - (1.0 - r) * t, 1.0 - (1.0 - g) * t, 1.0 - (1.0 - b) * t)

    # Load TW aggregates
    agg_path = os.path.join(score_dir, "regional_t_aggregates.csv")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(f"Missing aggregated TW file: {agg_path}")
    agg = pd.read_csv(agg_path)
    agg = agg[agg["obs_source"] == obs_source].copy()

    regions_available = sorted(agg["region"].dropna().unique().tolist())
    if not regions_available:
        raise RuntimeError("No regions found in regional_t_aggregates.csv for chosen obs_source.")
    pretty_name = {
        "Northern_Hemisphere": "Northern Hemisphere",
        "Southern_Hemisphere": "Southern Hemisphere",
        "NH_extratropics": "NH Extra-Tropics",
        "SH_extratropics": "SH Extra-Tropics",
        "Tropics": "Tropics",
        "Extratropics": "Extra-Tropics",
        "Arctic": "Arctic",
        "Antarctic": "Antarctic",
        "Europe": "Europe",
        "North_America": "North America",
        "North_Atlantic": "North Atlantic",
        "North_Pacific": "North Pacific",
        "East_Asia": "East Asia",
        "AusNZ": "AusNZ",
        "Global": "Global",
    }
    preferred = [
        "Northern_Hemisphere","Southern_Hemisphere",
        "Tropics","Extratropics","Arctic","Antarctic",
        "Europe","North_America","North_Atlantic","North_Pacific","East_Asia","AusNZ","Global",
        "NH_extratropics","SH_extratropics",
    ]
    regions = [r for r in preferred if r in regions_available] or regions_available

    tw_col = "tw_pc_mean_wb2" if metric == "pc" else "tw_pcs_mean_wb2"

    # region masks + classic aggregation for PC/PCS per region from per-lead CSVs
    def _region_masks_for_grid(lat_vals: np.ndarray, lon_vals: np.ndarray) -> dict[str, np.ndarray]:
        lat_vals = np.asarray(lat_vals, dtype=float)
        lon_vals = ((np.asarray(lon_vals, dtype=float) + 180.0) % 360.0) - 180.0
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

        def _box(latmin, latmax, lonmin, lonmax):
            if lonmin <= lonmax:
                cond_lon = (lon2d >= lonmin) & (lon2d <= lonmax)
            else:
                cond_lon = (lon2d >= lonmin) | (lon2d <= lonmax)
            return (lat2d >= latmin) & (lat2d <= latmax) & cond_lon

        return {
            "Northern_Hemisphere": (lat2d >= 0),
            "Southern_Hemisphere": (lat2d < 0),
            "Tropics":             (lat2d >= -20.0) & (lat2d <= 20.0),
            "Extratropics":        (np.abs(lat2d) >= 20.0),
            "Arctic":              (lat2d >= 60.0),
            "Antarctic":           (lat2d <= -60.0),
            "Europe":        _box(35.0, 75.0,  -12.5,  42.5),
            "North_America": _box(25.0, 60.0, -120.0, -75.0),
            "North_Atlantic":_box(25.0, 60.0,  -70.0, -20.0),
            "North_Pacific": _box(25.0, 60.0,  145.0, -130.0),
            "East_Asia":     _box(25.0, 60.0,  102.5, 150.0),
            "AusNZ":         _box(-45.0,-12.5, 120.0, 175.0),
            "Global":        np.ones_like(lat2d, dtype=bool),
            "NH_extratropics": (lat2d >= 20.0),
            "SH_extratropics": (lat2d <= -20.0),
        }

    def _wb2_weighted_mean_grid(grid_2d: np.ndarray, lat_vals: np.ndarray) -> float:
        A = np.asarray(grid_2d, dtype=float)
        if A.size == 0:
            return np.nan
        mask = np.isfinite(A)
        if not mask.any():
            return np.nan
        w_lat = np.cos(np.deg2rad(np.asarray(lat_vals, dtype=float)))[:, None]
        W = np.where(mask, w_lat, 0.0)
        s = np.nansum(A * W)
        t = np.sum(W)
        return float(s / t) if t > 0 else np.nan

    def _classic_normal_by_region(var: str) -> pd.DataFrame:
        out_rows = []
        value_col = metric  # "pc" or "pcs"
        for D in lead_days:
            fpath = os.path.join(score_dir, f"{var}_lead{D}d_{obs_source}.csv")
            if not os.path.exists(fpath):
                for mdl in models_csv:
                    for reg in regions:
                        out_rows.append({"variable": var, "lead_time": D, "model": mdl,
                                         "region": reg, "value": np.nan})
                continue

            df = pd.read_csv(fpath)
            df = df[(df["variable"] == var) & (df["lead_time"] == D)][["model","lat","lon",value_col]].copy()

            for mdl in models_csv:
                sub = df[df["model"] == mdl][["lat","lon",value_col]].dropna()
                if sub.empty:
                    for reg in regions:
                        out_rows.append({"variable": var, "lead_time": D, "model": mdl,
                                         "region": reg, "value": np.nan})
                    continue

                grid = sub.pivot_table(index="lat", columns="lon", values=value_col, aggfunc="mean")
                lat_vals = grid.index.to_numpy(float)
                masks = _region_masks_for_grid(lat_vals, grid.columns.to_numpy(float))

                for reg in regions:
                    if reg not in masks:
                        out_rows.append({"variable": var, "lead_time": D, "model": mdl,
                                         "region": reg, "value": np.nan})
                        continue
                    arr_masked = np.where(masks[reg], grid.to_numpy(float), np.nan)
                    val = _wb2_weighted_mean_grid(arr_masked, lat_vals)
                    out_rows.append({"variable": var, "lead_time": D, "model": mdl,
                                     "region": reg, "value": val})
        return pd.DataFrame(out_rows)

    normal_cache = {
        "mean_sea_level_pressure": _classic_normal_by_region("mean_sea_level_pressure"),
        "2m_temperature":          _classic_normal_by_region("2m_temperature"),
        "10m_wind_speed":          _classic_normal_by_region("10m_wind_speed"),
    }

    def _pick_row(df_block: pd.DataFrame, value_col: str) -> pd.Series | None:
        if df_block.empty:
            return None
        s = df_block[value_col].astype(float)
        k = s.idxmin(skipna=True) if metric == "pc" else s.idxmax(skipna=True)
        if k is None or not np.isfinite(s.get(k, np.nan)):
            return None
        return df_block.loc[k]

    def _winners_matrix(var: str, tail: str | None, p_low: float | None) -> np.ndarray:
        M = np.empty((len(regions), len(lead_days)), dtype=object)
        if tail is None:
            dfN = normal_cache[var]
            for i, reg in enumerate(regions):
                for j, D in enumerate(lead_days):
                    row = _pick_row(dfN[(dfN["region"] == reg) & (dfN["lead_time"] == D)], "value")
                    M[i, j] = None if row is None else group_label.get(row["model"], None)
            return M

        tq = (1.0 - p_low) if tail == "upper" else p_low
        dfE = agg[
            (agg["variable"] == var) &
            (agg["tw_tail"] == tail) &
            (agg["t_quantile"].round(6) == round(tq, 6))
        ].copy()
        dfE["label"] = dfE["model"].map(group_label)

        for i, reg in enumerate(regions):
            for j, D in enumerate(lead_days):
                block = dfE[(dfE["region"] == reg) & (dfE["lead_time"] == D) & (dfE["label"].notna())]
                row = _pick_row(block, tw_col)
                M[i, j] = None if row is None else row["label"]
        return M

    # Load regional time series if significance enabled
    if compute_significance:
        if timeseries_csv is None:
            timeseries_csv = os.path.join(score_dir, f"regional_timeseries_{obs_source}.csv")
        if not os.path.exists(timeseries_csv):
            raise FileNotFoundError(f"Missing regional time-series CSV: {timeseries_csv}")
        df_ts_all = pd.read_csv(timeseries_csv, parse_dates=["time"])
        need_cols = {"obs_source","region","variable","lead_time_d","model","time","pc"}
        if not need_cols.issubset(df_ts_all.columns):
            missing = need_cols.difference(df_ts_all.columns)
            raise KeyError(f"Missing required columns in {timeseries_csv}: {sorted(missing)}")
        df_ts_all = df_ts_all[df_ts_all["obs_source"] == obs_source].copy()
    else:
        df_ts_all = None

    def _significant_win(label_winner: str | None, region: str, variable: str, lead_d: int,
                         row_idx: int, tail: str | None) -> bool | None:
        if not compute_significance or label_winner is None:
            return None

        models_win = label_to_models.get(label_winner, [])
        if not models_win:
            return None
        model_a = models_win[0]

        other_labels = [lab for lab in ("GC","PW","HRES") if lab != label_winner]
        rivals = [label_to_models[lab][0] for lab in other_labels if lab in label_to_models]
        if len(rivals) != 2:
            return None

        # choose correct time-series column for permutation tests
        if row_idx == 0 or tail is None:
            metric_col = "pc"
        else:
            p_low_map = {1: 0.10, 2: 0.05, 3: 0.01}
            p_low = p_low_map[row_idx]
            q = (1.0 - p_low) if tail == "upper" else p_low
            if q in (0.1, 0.9):
                metric_col = f"tw_pc_{q:.1f}"
            elif q in (0.05, 0.95, 0.01, 0.99):
                metric_col = f"tw_pc_{q:.2f}".rstrip("0").rstrip(".")
            else:
                metric_col = f"tw_pc_{q}"

        try:
            pvals = []
            for rb in rivals:
                base = df_ts_all[
                    (df_ts_all["region"] == region) &
                    (df_ts_all["variable"] == variable) &
                    (df_ts_all["lead_time_d"] == int(lead_d)) &
                    (df_ts_all["model"].isin([model_a, rb]))
                ][["time","model",metric_col]].dropna()
                if base.empty or base["model"].nunique() < 2:
                    return None
                wide = base.pivot_table(index="time", columns="model", values=metric_col, aggfunc="mean")
                if model_a not in wide or rb not in wide:
                    return None
                ab = wide[[model_a, rb]].dropna()
                if ab.shape[0] < 10:
                    return None

                key = f"{obs_source}|{region}|{variable}|{lead_d}|{model_a}|{rb}|{metric_col}"
                seed_offset = int(zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF)
                block_length = max(1, 2 * int(lead_d))

                p = float(block_permutation_test(
                    ab[model_a].to_numpy(np.float64),
                    ab[rb].to_numpy(np.float64),
                    seed_offset=seed_offset,
                    block_length=block_length,
                    b=b_perm,
                    root_seed=root_seed,
                ))
                pvals.append(p)
        except Exception:
            return None

        if not all(np.isfinite(p) for p in pvals):
            return None
        return bool(all(p < alpha for p in pvals))

    # Build figure
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=figsize, constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.91, bottom=0.1, wspace=0.20, hspace=0.20)

    for c, (var, col_title, tail_hint) in enumerate(col_specs):
        mats = [
            _winners_matrix(var, tail=None,        p_low=None),
            _winners_matrix(var, tail=tail_hint,   p_low=0.10),
            _winners_matrix(var, tail=tail_hint,   p_low=0.05),
            _winners_matrix(var, tail=tail_hint,   p_low=0.01),
        ]
        for r in range(4):
            ax = axes[r, c]
            mat = mats[r]
            row_tail = None if r == 0 else tail_hint

            rgb = np.zeros((len(regions), len(lead_days), 3), dtype=float)
            for i, reg in enumerate(regions):
                for j, D in enumerate(lead_days):
                    lab = mat[i, j]
                    if lab in ("GC","PW","HRES"):
                        is_sig = _significant_win(lab, reg, var, D, r, row_tail)
                        t = 1.0 if is_sig is True else float(fair_mix)
                        r_, g_, b_ = _blend_with_white({"GC":"#1f77b4","PW":"#ff7f0e","HRES":"#2ca02c"}[lab], t)
                        rgb[i, j, :] = (r_, g_, b_)
                    else:
                        rgb[i, j, :] = (0.898, 0.905, 0.914)  # light gray

            ax.imshow(rgb, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)

            # grid
            ax.set_xticks(np.arange(len(lead_days)) - 0.5, minor=True)
            ax.set_yticks(np.arange(len(regions)) - 0.5, minor=True)
            ax.grid(which="minor", color="black", linewidth=0.8)

            if r == 0:
                ax.set_title(col_title, fontsize=16, pad=6)
            if c == 0:
                ax.set_yticks(np.arange(len(regions)))
                ax.set_yticklabels([pretty_name.get(rn, rn).replace("_"," ") for rn in regions], fontsize=14)
                ax.set_ylabel(["PC (unweighted)","10% extremes","5% extremes","1% extremes"][r], fontsize=16)
            else:
                ax.set_yticks([])

            ax.set_xticks(np.arange(len(lead_days)))
            ax.set_xticklabels([str(d) for d in lead_days], fontsize=14)
            if r == 3:
                ax.set_xlabel("Lead Time [d]", fontsize=14)

    # Legend
    strong = {"GC":"#1f77b4", "PW":"#ff7f0e", "HRES":"#2ca02c"}
    pale   = {"GC":_blend_with_white(strong["GC"], fair_mix),
              "PW":_blend_with_white(strong["PW"], fair_mix),
              "HRES":_blend_with_white(strong["HRES"], fair_mix)}
    handles = [
        plt.Rectangle((0,0),1,1, facecolor=strong["GC"], label=f"{leg_gc} (significant)"),
        plt.Rectangle((0,0),1,1, facecolor=pale["GC"],   label=f"{leg_gc} (not significant)"),
        plt.Rectangle((0,0),1,1, facecolor=strong["PW"], label=f"{leg_pw} (significant)"),
        plt.Rectangle((0,0),1,1, facecolor=pale["PW"],   label=f"{leg_pw} (not significant)"),
        plt.Rectangle((0,0),1,1, facecolor=strong["HRES"], label="HRES (significant)"),
        plt.Rectangle((0,0),1,1, facecolor=pale["HRES"],   label="HRES (not significant)"),
    ]
    fig.legend(handles=handles,
               labels=[h.get_label() for h in handles],
               ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.02),
               frameon=False, fontsize=14,
               columnspacing=1.5)

    title_top = "Summary Scorecards – best Model by Region & Lead Time"
    title_sub = f"(Ground Truth={obs_source.upper()}, Metric=(tw){metric.upper()} | α={alpha}, b={b_perm})"
    fig.suptitle(f"{title_top}\n{title_sub}", y=0.97, fontsize=20)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    return fig

# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def run_permutation_pipeline(
    *,
    # ---------- SWITCHES ----------
    make_boxplots: bool = True,
    make_regional_scorecards: bool = True,

    # ---------- GENERAL (applies to BOTH boxplots & regional) ----------
    metric_base: str = "tw_pc",   # 'pc', 'tw_pc', 'qw_pc', or explicit like 'qw_pc_0.9'
    rg: float = 0.01,             # extremes fraction when tails are implied (no explicit q)
    b_perm: int = 1000,           # permutations for ALL block permutation tests
    root_seed: int = 42,          # global seed for reproducibility

    # ---------- COMPUTE (Dask) ----------
    scheduler: str = "threads",
    num_workers: int = 24,

    # ---------- REGIONAL-ONLY ----------
    scorecards_obs: str = "both", # choose 'era5', 'ifs', or 'both'
    alpha: float = 0.05,          # significance level for regional scorecards
    fair_mix: float = 0.45,       # blend toward white for non-significant tiles (0..1)
) -> None:
    """
    Orchestrate permutation-based evaluation and plotting:

      • Boxplots: grid-wise p-values (ERA5 & IFS), using b_perm/root_seed
      • Regional scorecards: significance-aware winner maps, using b_perm/alpha/fair_mix

    Notes
    -----
    - `metric_base='pc'` has no tails; for 'tw_pc'/'qw_pc' you can either give an explicit
      quantile (e.g. 'qw_pc_0.9') or leave it implicit and control tails via `rg`.
    - `b_perm` is used EVERYWHERE we compute permutation p-values (boxplots + regional).
    """
    # Only check Zarr stores if we actually need them for boxplots
    if make_boxplots:
        for p in (RAW_ERA5, RAW_IFS):
            if not (os.path.exists(p) and os.path.isdir(p)):
                raise FileNotFoundError(
                    f"Expected a Zarr store directory at: {p}\n"
                    "Make sure it contains valid Zarr metadata ('.zgroup', '.zattrs')."
                )

    # ---- interpret metric base & tails ----
    base, parsed_q = _parse_metric_base(metric_base)
    if parsed_q is not None:
        q_low, q_high = None, parsed_q
        tails_enabled = False
        metric_label_suffix = ""
    else:
        q_low, q_high = rg, (1.0 - rg)
        tails_enabled = base != "pc"
        metric_label_suffix = f" ({int(rg*100)}% extremes)" if tails_enabled else ""

    # -----------------------------------------------------------------
    # 1) BOX PLOTS (grid-wise p-values) — uses b_perm + root_seed
    # -----------------------------------------------------------------
    if make_boxplots:
        print("\n[boxplots] computing grid-wise p-values for boxplots …")

        # Build the list of (key, title, variable, metric_name)
        jobs = []
        for var, short in zip(VARIABLES_ALL, TITLES_ALL):
            tails = VAR_TO_TAILS[var] if (tails_enabled and parsed_q is None) else ("all",)
            for tail in tails:
                if tail == "low":
                    metric_name = _metric_name_for_tail(base, q_low)
                    title = f"{short} (low)"
                    key = f"{var}:low"
                elif tail == "high":
                    metric_name = _metric_name_for_tail(base, q_high)
                    title = f"{short} (high)"
                    key = f"{var}:high"
                else:
                    metric_name = _metric_name_for_tail(base, parsed_q) if parsed_q is not None else base
                    title = short
                    key = f"{var}:all"
                jobs.append((key, title, var, metric_name))

        p_vs_era5_by_key: dict[str, dict[str, xr.DataArray]] = {}
        p_vs_ifs_by_key:  dict[str, dict[str, xr.DataArray]] = {}

        for ji, (key, title, var, metric) in enumerate(jobs, start=1):
            print(f"\n[{ji}/{len(jobs)}] {title} — metric: {metric}")

            # ERA5 (reanalysis)
            print("  • ERA5: GC vs PW / GC vs HRES / PW vs HRES …")
            p_vs_era5_by_key[key] = {
                "GC_vs_PW":   run_block_permutation(
                                  RAW_ERA5, "graphcast", "pangu", metric, var,
                                  b=b_perm, root_seed=root_seed,
                                  scheduler=scheduler, num_workers=num_workers
                              ),
                "GC_vs_HRES": run_block_permutation(
                                  RAW_ERA5, "graphcast", "hres",  metric, var,
                                  b=b_perm, root_seed=root_seed,
                                  scheduler=scheduler, num_workers=num_workers
                              ),
                "PW_vs_HRES": run_block_permutation(
                                  RAW_ERA5, "pangu",     "hres",  metric, var,
                                  b=b_perm, root_seed=root_seed,
                                  scheduler=scheduler, num_workers=num_workers
                              ),
            }

            # IFS (operational)
            print("  • IFS: GC(op) vs PW(op) / GC(op) vs HRES / PW(op) vs HRES …")
            p_vs_ifs_by_key[key] = {
                "GC_vs_PW":   run_block_permutation(
                                  RAW_IFS, "graphcast_operational", "pangu_operational", metric, var,
                                  b=b_perm, root_seed=root_seed,
                                  scheduler=scheduler, num_workers=num_workers
                              ),
                "GC_vs_HRES": run_block_permutation(
                                  RAW_IFS, "graphcast_operational", "hres",              metric, var,
                                  b=b_perm, root_seed=root_seed,
                                  scheduler=scheduler, num_workers=num_workers
                              ),
                "PW_vs_HRES": run_block_permutation(
                                  RAW_IFS, "pangu_operational",     "hres",              metric, var,
                                  b=b_perm, root_seed=root_seed,
                                  scheduler=scheduler, num_workers=num_workers
                              ),
            }

        keys   = [k for (k, _, _, _) in jobs]
        titles = [t for (_, t, _, _) in jobs]
        metric_label = (base if parsed_q is None else f"{base}_{parsed_q}".rstrip("0").rstrip(".")) + metric_label_suffix
        out_box = os.path.join(PLOTS_DIR, f"boxplot_pvalues_{base}.png")
        print("\n  • plotting combined figure …")
        plot_p_value_boxplots_styled(
            p_vs_era5_by_key, p_vs_ifs_by_key,
            keys, titles, LEAD_TIMES,
            out_box,
            metric_label=metric_label,
        )
        print(f"  ✓ saved {out_box}")

    # -----------------------------------------------------------------
    # 2) REGIONAL SCORECARDS (significance-aware) — b_perm + alpha + fair_mix
    # -----------------------------------------------------------------
    if make_regional_scorecards:
        print("\n[regional scorecards] building 4×5 maps with significance shading …")

        if scorecards_obs not in {"era5", "ifs", "both"}:
            raise ValueError("scorecards_obs must be 'era5', 'ifs', or 'both'.")
        obs_list = ["era5", "ifs"] if scorecards_obs == "both" else [scorecards_obs]

        for obs in obs_list:
            save_pc  = os.path.join(PLOTS_DIR, f"regional_scorecard_{obs}_pc.png")
            save_pcs = os.path.join(PLOTS_DIR, f"regional_scorecard_{obs}_pcs.png")

            # PC (smaller is better)
            plot_regional_scorecard_4x5(
                score_dir=SCORE_DIR,
                obs_source=obs,
                metric="pc",
                savepath=save_pc,
                timeseries_csv=os.path.join(SCORE_DIR, f"regional_timeseries_{obs}.csv"),
                alpha=alpha,
                b_perm=b_perm,
                root_seed=root_seed,
                fair_mix=fair_mix,
                compute_significance=True,
            )
            print(f"  ✓ saved {save_pc}")

            # PCS (larger is better)
            plot_regional_scorecard_4x5(
                score_dir=SCORE_DIR,
                obs_source=obs,
                metric="pcs",
                savepath=save_pcs,
                timeseries_csv=os.path.join(SCORE_DIR, f"regional_timeseries_{obs}.csv"),
                alpha=alpha,
                b_perm=b_perm,
                root_seed=root_seed,
                fair_mix=fair_mix,
                compute_significance=True,
            )
            print(f"  ✓ saved {save_pcs}")


# ---------------------------------------------------------------------
# Main 
# ---------------------------------------------------------------------

def main() -> None:
    # ===================== SWITCHES =====================
    MAKE_BOXPLOTS = False
    MAKE_REGIONAL_SCORECARDS = False

    # ===================== GENERAL (applies to BOTH) =====================
    METRIC_BASE = "tw_pc"   # 'pc', 'tw_pc', 'qw_pc', or explicit like 'qw_pc_0.9'
    RG = 0.01               # extremes fraction when tails are implied (no explicit q)
    B_PERM = 1000           # permutations for block permutation tests
    ROOT_SEED = 42          # global seed for reproducibility

    # ===================== COMPUTE (Dask) =====================
    SCHEDULER = "threads"
    NUM_WORKERS = 24

    # ===================== REGIONAL-ONLY =====================
    SCORECARDS_OBS = "both" # choose 'era5', 'ifs', or 'both'
    ALPHA = 0.05            # significance level in regional scorecards
    FAIR_MIX = 0.45         # blend toward white for non-significant winners (0..1)

    run_permutation_pipeline(
        make_boxplots=MAKE_BOXPLOTS,
        make_regional_scorecards=MAKE_REGIONAL_SCORECARDS,

        metric_base=METRIC_BASE,
        rg=RG,
        b_perm=B_PERM,
        root_seed=ROOT_SEED,

        scheduler=SCHEDULER,
        num_workers=NUM_WORKERS,

        scorecards_obs=SCORECARDS_OBS,
        alpha=ALPHA,
        fair_mix=FAIR_MIX,
    )


if __name__ == "__main__":
    main()

