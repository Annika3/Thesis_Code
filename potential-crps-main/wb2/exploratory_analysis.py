# plotting_obs_diagnostics.py
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from dask.diagnostics import ProgressBar

# ------------------------------------------------------------
# Paths, variables, and metadata
# ------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))

# Input Zarrs 
data_dir = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# Output directories
plots_dir = os.path.join(script_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Observation datasets 
obs_keys = ["era5", "ifs_analysis"]
obs_labels = {"era5": "ERA5", "ifs_analysis": "IFS analysis"}

# Lead times (days) used across plots
lead_times_days = (1, 3, 5, 7, 10)

# Variables used across plots
variables = [
    "mean_sea_level_pressure",  # MSLP
    "2m_temperature",           # T2M
    "10m_wind_speed",           # WS10
]
var_titles = {
    "mean_sea_level_pressure": "MSLP",
    "2m_temperature": "T2M",
    "10m_wind_speed": "WS10",
}
var_units = {
    "mean_sea_level_pressure": "Pa",   # shown as hPa in QQ plots via conversion
    "2m_temperature": "K",
    "10m_wind_speed": "m s⁻¹",
}

# Rows for regional density plots (None → global distribution)
rows_regional = [
    ("Global", None),
    ("Canada", (55.0, -105.0)),
    ("Equator (Congo)", (0.0, 15.0)),
    ("Southern Ocean", (-60.0, 30.0)),
]

# Model mapping (per truth) used by scatter/QQ plots
MODEL_KEYS_BY_TRUTH = {
    "ifs_analysis": {  # IFS truth → use operational models
        "GC": "graphcast_operational",
        "PW": "pangu_operational",
        "HRES": "hres",
    },
    "era5": {          # ERA5 truth → use reanalysis models
        "GC": "graphcast",
        "PW": "pangu",
        "HRES": "hres",
    },
}

# ------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------

def load_obs(name: str) -> xr.Dataset:
    """
    Load an observation Zarr by key from ./data.
    """
    p = os.path.join(data_dir, f"{name}_64x32.zarr")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Zarr not found: {p}")
    return xr.open_zarr(p, consolidated=True)

def load_all() -> dict[str, xr.Dataset]:
    """
    Load all observations listed in obs_keys with a visible Dask progress bar.
    """
    with ProgressBar():
        obs = {k: load_obs(k) for k in obs_keys}
    return obs

# ------------------------------------------------------------
# Utility helpers (coords, selection, simple stats)
# ------------------------------------------------------------

def to_1d_clean(a: np.ndarray) -> np.ndarray:
    """
    Flatten an array and drop non-finite values.
    """
    x = np.asarray(a).ravel()
    return x[np.isfinite(x)]

def normalize_lon_for_dataset(ds_lon_vals: np.ndarray, lon_target: float) -> float:
    """
    Normalize a target longitude to the dataset’s longitude convention.
    """
    lon_vals = np.asarray(ds_lon_vals)
    if lon_vals.min() >= 0 and lon_vals.max() <= 360:
        return lon_target % 360.0
    return ((lon_target + 180) % 360) - 180

def pick_nearest_coord(ds: xr.Dataset, lat_t: float, lon_t: float) -> tuple[float, float]:
    """
    Return the nearest (lat, lon) grid coordinate in ds to a target coordinate.
    """
    lon_t = normalize_lon_for_dataset(ds.longitude.values, lon_t)
    da_ref = ds[variables[0]].sel(latitude=lat_t, longitude=lon_t, method="nearest")
    lat_sel = float(np.asarray(da_ref.latitude))
    lon_sel = float(np.asarray(da_ref.longitude))
    return lat_sel, lon_sel

def select_values_global(ds: xr.Dataset, var: str, time_mask=None) -> np.ndarray:
    """
    Select all values for a variable (optionally time-filtered) and return as 1D, finite-only array.
    """
    da = ds[var]
    if time_mask is not None and "time" in da.dims:
        da = da.sel(time=time_mask)
    return to_1d_clean(da.values)

def select_values_at(ds: xr.Dataset, var: str, lat_sel: float, lon_sel: float, time_mask=None) -> np.ndarray:
    """
    Select values at a specific grid point (optionally time-filtered) and return as 1D, finite-only array.
    """
    da = ds[var].sel(latitude=lat_sel, longitude=lon_sel)
    if time_mask is not None and "time" in da.dims:
        da = da.sel(time=time_mask)
    return to_1d_clean(da.values)

def kde_curve(x: np.ndarray, xmin: float, xmax: float, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a smooth density curve via Gaussian KDE; fall back to a histogram if SciPy is unavailable.
    """
    xx = np.linspace(xmin, xmax, n)
    try:
        from scipy.stats import gaussian_kde
        yy = gaussian_kde(x)(xx)
    except Exception:
        hist, edges = np.histogram(x, bins=min(512, max(32, int(np.sqrt(x.size)))), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        yy = np.interp(xx, centers, hist, left=0.0, right=0.0)
    return xx, yy

def fmt_lat(lat: float) -> str:
    """Return a human-readable latitude label."""
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'}"

def fmt_lon(lon: float) -> str:
    """Return a human-readable longitude label (normalized to [-180, 180])."""
    ll = ((lon + 180) % 360) - 180
    return f"{abs(ll):.2f}°{'E' if ll >= 0 else 'W'}"

def _mu_sigma(vals: np.ndarray) -> tuple[float, float]:
    """
    Return mean (μ) and population standard deviation (σ) for a 1D array.
    """
    if vals.size == 0:
        return np.nan, np.nan
    mu = float(np.mean(vals))
    sigma = float(np.std(vals))
    return mu, sigma

def _lead_label(ld: int) -> str:
    """
    Return a formatted lead-time label with singular/plural handling.
    """
    return "Lead Time = 1 day" if int(ld) == 1 else f"Lead Time = {ld} days"

# ------------------------------------------------------------
# Model loading helpers (for scatter / QQ)
# ------------------------------------------------------------

def load_model(name: str) -> xr.Dataset:
    """
    Load a forecast Zarr by model key from ./data.
    """
    p = os.path.join(data_dir, f"{name}_64x32.zarr")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Model zarr not found: {p}")
    return xr.open_zarr(p, consolidated=True, decode_timedelta=True)

def _xy_for_one_lead(model_ds: xr.Dataset, truth_ds: xr.Dataset, var: str, lead_days: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract truth and prediction series aligned at a given lead time (days) and return flattened arrays.
    """
    lead_td = np.timedelta64(int(lead_days), "D")
    pred = model_ds[var].sel(prediction_timedelta=lead_td, method="nearest")
    valid_time = pred.time + lead_td
    truth = truth_ds[var].sel(time=valid_time)

    x = to_1d_clean(truth.values)
    y = to_1d_clean(pred.values)
    n = min(x.size, y.size)
    return x[:n], y[:n]

def load_models_for_truth(ground_truth: str) -> dict[str, xr.Dataset]:
    """
    Load the trio of models appropriate for a given ground truth key.
    """
    if ground_truth not in MODEL_KEYS_BY_TRUTH:
        raise ValueError(f"ground_truth must be one of {list(MODEL_KEYS_BY_TRUTH)}")
    keys = MODEL_KEYS_BY_TRUTH[ground_truth]
    with ProgressBar():
        return {abbr: load_model(keys[abbr]) for abbr in ("GC", "PW", "HRES")}

# ------------------------------------------------------------
# Scatter plot: rows = lead times, cols = variables; X=truth, Y=model
# ------------------------------------------------------------

def plot_scatter_grid_by_lead(
    ground_truth: str = "era5",
    model_row: str = "GC",
    lead_times_days=lead_times_days,
) -> None:
    """
    Create a grid of scatter plots (rows = lead times, columns = variables).
    X-axis shows the selected ground truth; Y-axis shows the selected model.
    """
    obs = load_all()
    if ground_truth not in obs:
        raise ValueError(f"ground_truth must be one of {list(obs.keys())}")
    truth_ds = obs[ground_truth]

    if model_row not in ("GC", "PW", "HRES"):
        raise ValueError("model_row must be 'GC', 'PW', or 'HRES'")
    model_key = MODEL_KEYS_BY_TRUTH[ground_truth][model_row]
    print(f"[scatter-by-lead] loading model '{model_key}' for {model_row} with truth={ground_truth} …")
    model_ds = load_model(model_key)

    truth_short = "IFS" if ground_truth == "ifs_analysis" else "ERA5"
    model_label = f"{model_row}-{truth_short}"
    x_label_truth = "IFS analysis" if ground_truth == "ifs_analysis" else obs_labels[ground_truth]

    lt_days = list(lead_times_days)
    n_rows, n_cols = len(lt_days), len(variables)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 3.6 * n_rows), squeeze=False)

    total = n_rows * n_cols
    k = 0
    for i, ld in enumerate(lt_days):
        print(f"[scatter-by-lead] lead={ld} d")
        for j, var in enumerate(variables):
            ax = axes[i, j]
            print(f"   → {model_row} / {var} … ", end="", flush=True)

            x, y = _xy_for_one_lead(model_ds, truth_ds, var, ld)
            if x.size == 0 or y.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                print("no data")
                k += 1
                continue

            vmin = float(min(x.min(), y.min()))
            vmax = float(max(x.max(), y.max()))
            pad = 0.02 * (vmax - vmin + 1e-12)
            lo, hi = vmin - pad, vmax + pad

            ax.plot([lo, hi], [lo, hi], linestyle="--", color="0.4", linewidth=1.0)
            ax.scatter(x, y, s=2, alpha=0.5, edgecolors="none")

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

            if i == 0:
                ax.set_title(var_titles[var], fontsize=16, pad=8)
            if j == 0:
                ax.set_ylabel(f"{model_label}\n{_lead_label(ld)}", fontsize=14)
            if i == n_rows - 1:
                ax.set_xlabel(f"{x_label_truth} ({var_units[var]})", fontsize=14)

            k += 1
            print(f"done ({k}/{total})")

    fig.suptitle(f"Scatterplot: {model_label} forecasts vs. {x_label_truth}", fontsize=20, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_name = f"scatter_by_lead_{model_row}_{ground_truth}.png"
    out_path = os.path.join(plots_dir, out_name)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[scatter-by-lead] Saved: {out_path}")

# ------------------------------------------------------------
# QQ plots for extremes by lead (all models)
# ------------------------------------------------------------

def _unit_label_for_var(var: str) -> str:
    """Return a unit label suitable for axis annotations."""
    if var == "mean_sea_level_pressure":
        return "hPa"
    if var == "2m_temperature":
        return "K"
    if var == "10m_wind_speed":
        return "m s⁻¹"
    return ""

def _rescale_for_var(var: str, values: np.ndarray) -> np.ndarray:
    """Apply unit conversion where necessary (Pa → hPa)."""
    if var == "mean_sea_level_pressure":
        return values / 100.0
    return values

def plot_qq_extremes_grid_by_lead_allmodels(
    ground_truth: str = "ifs_analysis",
    lead_times_days=lead_times_days,
) -> None:
    """
    Create QQ-plot grids for extremes.
    Rows correspond to lead times; columns correspond to extreme tails per variable.
    MSLP is plotted in hPa; T2M and WS10 retain native units.
    """
    obs = load_all()
    if ground_truth not in obs:
        raise ValueError(f"ground_truth must be one of {list(obs.keys())}")
    truth_ds = obs[ground_truth]

    model_keys = MODEL_KEYS_BY_TRUTH[ground_truth]
    with ProgressBar():
        models = {abbr: load_model(model_keys[abbr]) for abbr in ("GC", "PW", "HRES")}

    truth_short = "IFS" if ground_truth == "ifs_analysis" else "ERA5"
    x_label_truth = truth_short
    title_label = f"Ground Truth: {truth_short}"

    cols = [
        ("mean_sea_level_pressure", "low"),
        ("mean_sea_level_pressure", "high"),
        ("2m_temperature", "low"),
        ("2m_temperature", "high"),
        ("10m_wind_speed", "high"),
    ]
    col_titles = [
        f"{var_titles['mean_sea_level_pressure']} – low extremes",
        f"{var_titles['mean_sea_level_pressure']} – high extremes",
        f"{var_titles['2m_temperature']} – low extremes",
        f"{var_titles['2m_temperature']} – high extremes",
        f"{var_titles['10m_wind_speed']} – high extremes",
    ]

    lt_days = list(lead_times_days)
    n_rows, n_cols = len(lt_days), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.8 * n_rows), squeeze=False)

    ref_style = dict(linestyle="-", color="0.4", linewidth=1.2)
    colors = {"GC": "C0", "PW": "C1", "HRES": "C2"}
    model_names = {"GC": f"GC-{truth_short}", "PW": f"PW-{truth_short}", "HRES": "HRES"}

    for i, ld in enumerate(lt_days):
        for j, (var, tail) in enumerate(cols):
            ax = axes[i, j]

            p_lo, p_hi = (0.0, 0.10) if tail == "low" else (0.90, 1.0)
            vmins, vmaxs = [], []

            for abbr in ("GC", "PW", "HRES"):
                x_series, y_series = _xy_for_one_lead(models[abbr], truth_ds, var, ld)
                x_series = _rescale_for_var(var, x_series)
                y_series = _rescale_for_var(var, y_series)
                n = min(x_series.size, y_series.size)
                if n < 5:
                    continue

                p = np.linspace(p_lo, p_hi, min(801, max(101, n // 2)))
                p = p[(p > 0.0) & (p < 1.0)]
                qx = np.quantile(x_series, p)
                qy = np.quantile(y_series, p)
                vmins += [qx.min(), qy.min()]
                vmaxs += [qx.max(), qy.max()]

                ax.plot(qx, qy, linestyle=":", linewidth=1.6,
                        color=colors[abbr], label=model_names[abbr])

            vmin, vmax = (0.0, 1.0) if not vmins else (float(min(vmins)), float(max(vmaxs)))
            pad = 0.02 * (vmax - vmin + 1e-12)
            lo, hi = vmin - pad, vmax + pad

            ax.plot([lo, hi], [lo, hi], **ref_style)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)

            if i == 0:
                ax.set_title(col_titles[j], fontsize=12, pad=8)
            if j == 0:
                ax.set_ylabel(f"{_lead_label(ld)}\n({_unit_label_for_var(var)})", fontsize=11)
            else:
                ax.set_ylabel(f"({_unit_label_for_var(var)})", fontsize=11)
            if i == n_rows - 1:
                ax.set_xlabel(f"{x_label_truth} ({_unit_label_for_var(var)})", fontsize=11)

    handles = [
        Line2D([0], [0], **ref_style, label="1:1 line"),
        Line2D([0], [0], color=colors["GC"], linestyle=":", linewidth=1.6, label=model_names["GC"]),
        Line2D([0], [0], color=colors["PW"], linestyle=":", linewidth=1.6, label=model_names["PW"]),
        Line2D([0], [0], color=colors["HRES"], linestyle=":", linewidth=1.6, label=model_names["HRES"]),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=14)

    fig.tight_layout(rect=[0, 0.03, 1, 0.965])
    fig.suptitle(f"Q–Q plot 10% most extreme values – {title_label}", fontsize=16, y=0.98)

    out_name = f"qq_extremes_grid_allmodels_{ground_truth}.png"
    out_path = os.path.join(plots_dir, out_name)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[qq-extremes-grid-allmodels] Saved: {out_path}")

# ------------------------------------------------------------
# Regional density plots (rows = predefined regions)
# ------------------------------------------------------------

def prepare_regional(obs: dict[str, xr.Dataset]):
    """
    Prepare coordinate choices and shared x-limits for regional density plots.
    """
    chosen_coords: dict[tuple[str, str], tuple[float, float] | None] = {}
    for row_label, region in rows_regional:
        for ds_key in obs_keys:
            if region is None:
                chosen_coords[(ds_key, row_label)] = None
            else:
                lat_sel, lon_sel = pick_nearest_coord(obs[ds_key], *region)
                chosen_coords[(ds_key, row_label)] = (lat_sel, lon_sel)

    # Report the selected grid points for reproducibility
    for row_label, region in rows_regional:
        if region is None:
            print(f"[{row_label}] uses all grid points (global).")
        else:
            print(f"[{row_label}] nearest grid points:")
            for ds_key in obs_keys:
                lat_sel, lon_sel = chosen_coords[(ds_key, row_label)]
                print(f"  - {obs_labels[ds_key]} -> ({fmt_lat(lat_sel)}, {fmt_lon(lon_sel)})")

    # Shared x-limits across datasets for each variable to improve comparability
    xlims = {v: [np.inf, -np.inf] for v in variables}
    for row_label, region in rows_regional:
        for var in variables:
            vals_all = []
            for ds_key in obs_keys:
                ds = obs[ds_key]
                if region is None:
                    vals = select_values_global(ds, var)
                else:
                    lat_sel, lon_sel = chosen_coords[(ds_key, row_label)]
                    vals = select_values_at(ds, var, lat_sel, lon_sel)
                if vals.size:
                    vals_all.append(vals)
            if vals_all:
                v = np.concatenate(vals_all)
                xlims[var][0] = min(xlims[var][0], float(np.min(v)))
                xlims[var][1] = max(xlims[var][1], float(np.max(v)))
    return rows_regional, chosen_coords, xlims

def plot_density_grid_regional(
    obs: dict[str, xr.Dataset],
    rows: list[tuple[str, tuple[float, float] | None]],
    chosen_coords: dict[tuple[str, str], tuple[float, float] | None],
    xlims: dict[str, list[float]],
    out_png: str = "obs_densities_regions.png",
) -> None:
    """
    Create regional density plots for the predefined rows.
    Mean (μ) and standard deviation (σ) are annotated per dataset in each cell.
    """
    n_rows, n_cols = len(rows), len(variables)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.2 * n_rows), squeeze=False)

    legend_fontsize = 14
    legend_handles = [
        Line2D([0], [0], color="C0", lw=1.6, label=obs_labels["era5"]),
        Line2D([0], [0], color="C1", lw=1.6, label=obs_labels["ifs_analysis"]),
        Line2D([0], [0], color="0.4", lw=1.2, linestyle="--", label="5th percentile"),
        Line2D([0], [0], color="0.4", lw=1.2, linestyle="-",  label="95th percentile"),
    ]
    legend_labels = [h.get_label() for h in legend_handles]

    box_kw = dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.3")

    for i, (row_label, region) in enumerate(rows):
        for j, var in enumerate(variables):
            ax = axes[i, j]

            # Collect per-dataset values for this panel
            data_by_src = {}
            for ds_key in obs_keys:
                ds = obs[ds_key]
                if region is None:
                    vals = select_values_global(ds, var)
                else:
                    lat_sel, lon_sel = chosen_coords[(ds_key, row_label)]
                    vals = select_values_at(ds, var, lat_sel, lon_sel)
                data_by_src[ds_key] = vals

            all_vals = np.concatenate([v for v in data_by_src.values() if v.size > 0])
            if all_vals.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                continue

            # Shared x-limits (precomputed across datasets/rows)
            xmin, xmax = xlims[var]
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
                xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))

            # Density curves (consistent colors w.r.t. legend handles)
            for k_ds, ds_key in enumerate(obs_keys):
                vals = data_by_src[ds_key]
                if vals.size == 0:
                    continue
                xx, yy = kde_curve(vals, xmin, xmax)
                ax.plot(xx, yy, linewidth=1.6, label=obs_labels[ds_key], color=f"C{k_ds}")

            # 5th/95th percentile markers (combined values for orientation)
            q5, q95 = np.percentile(all_vals, [5, 95])
            ax.axvline(q5,  linestyle="--", linewidth=1.2, color="0.4")
            ax.axvline(q95, linestyle="-",  linewidth=1.2, color="0.4")

            # μ/σ annotation (rounded to one decimal)
            lines = []
            for ds_key in obs_keys:
                vals = data_by_src[ds_key]
                if vals.size == 0:
                    continue
                mu, sigma = _mu_sigma(vals)
                lines.append(f"{obs_labels[ds_key]}: μ={mu:.1f}, σ={sigma:.1f}")

            if lines:
                x_txt, y_txt, ha, va = _stats_anchor(i, j, n_rows)
                ax.text(
                    x_txt, y_txt, "\n".join(lines),
                    transform=ax.transAxes, ha=ha, va=va,
                    fontsize=legend_fontsize, color="black",
                    bbox=box_kw, zorder=5,
                )

            # Axes cosmetics
            ax.set_xlim(xmin, xmax)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
            if i == 0:
                ax.set_title(var_titles[var], fontsize=14, pad=8)
            if j == 0:
                ax.set_ylabel(f"{row_label}\nDensity", fontsize=14)
            if i == n_rows - 1:
                ax.set_xlabel(var_units[var], fontsize=14)

    # Layout and legend
    plt.subplots_adjust(bottom=0.3)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.legend(
        legend_handles, legend_labels,
        loc="lower center",
        ncol=len(legend_labels),
        frameon=False,
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, 0.015)
    )

    out_path = os.path.join(plots_dir, out_png)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[regional-densities] Saved: {out_path}")

def _stats_anchor(i_row: int, j_col: int, n_rows: int) -> tuple[float, float, str, str]:
    """
    Return (x, y, ha, va) in axes coordinates for the μ/σ textbox.

    Rules:
      - Column 0: top-left, except bottom row → top-right
      - Column 1: top-left
      - Column 2: top-right
    """
    y, va = 0.96, "top"
    if j_col == 0:
        if i_row == n_rows - 1:
            return 0.98, y, "right", va
        else:
            return 0.02, y, "left", va
    elif j_col == 1:
        return 0.02, y, "left", va
    else:
        return 0.98, y, "right", va

# ------------------------------------------------------------
# Time-perspective density plots (global + quarters)
# ------------------------------------------------------------

def prepare_time(obs: dict[str, xr.Dataset]):
    """
    Prepare shared x-limits across quarters and datasets for global density plots.
    """
    rows_time = [
        ("Global", None),
        ("Q1 (Jan–Mar)", 1),
        ("Q2 (Apr–Jun)", 2),
        ("Q3 (Jul–Sep)", 3),
        ("Q4 (Oct–Dec)", 4),
    ]

    xlims = {v: [np.inf, -np.inf] for v in variables}
    for _, q in rows_time:
        for var in variables:
            vals_all = []
            for ds_key in obs_keys:
                ds = obs[ds_key]
                if q is None:
                    vals = select_values_global(ds, var)
                else:
                    time_mask = ds.time.dt.quarter == q
                    vals = select_values_global(ds, var, time_mask=time_mask)
                if vals.size:
                    vals_all.append(vals)
            if vals_all:
                v = np.concatenate(vals_all)
                xlims[var][0] = min(xlims[var][0], float(np.min(v)))
                xlims[var][1] = max(xlims[var][1], float(np.max(v)))
    return rows_time, xlims

def plot_density_grid_time(
    obs: dict[str, xr.Dataset],
    rows_time: list[tuple[str, int | None]],
    xlims: dict[str, list[float]],
    out_png: str = "obs_densities_time.png",
) -> None:
    """
    Create global density plots per calendar quarter (rows) across variables (columns).
    """
    n_rows, n_cols = len(rows_time), len(variables)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.2 * n_rows), squeeze=False)

    legend_handles, legend_labels = [], []
    added_quant = False

    for i, (label, q) in enumerate(rows_time):
        for j, var in enumerate(variables):
            ax = axes[i, j]

            data_by_src = {}
            for ds_key in obs_keys:
                ds = obs[ds_key]
                if q is None:
                    vals = select_values_global(ds, var)
                else:
                    time_mask = ds.time.dt.quarter == q
                    vals = select_values_global(ds, var, time_mask=time_mask)
                data_by_src[ds_key] = vals

            all_vals = np.concatenate([v for v in data_by_src.values() if v.size > 0])
            if all_vals.size == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_axis_off()
                continue

            xmin, xmax = xlims[var]
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
                xmin, xmax = float(np.min(all_vals)), float(np.max(all_vals))

            handles_this = []
            for ds_key in obs_keys:
                vals = data_by_src[ds_key]
                if vals.size == 0:
                    continue
                xx, yy = kde_curve(vals, xmin, xmax)
                (h,) = ax.plot(xx, yy, linewidth=1.6, label=obs_labels[ds_key])
                handles_this.append(h)

            q5, q95 = np.percentile(all_vals, [5, 95])
            ax.axvline(q5,  linestyle="--", linewidth=1.2, color="0.4")
            ax.axvline(q95, linestyle="-",  linewidth=1.2, color="0.4")

            if not legend_handles and handles_this:
                legend_handles = handles_this.copy()
                legend_labels = [h.get_label() for h in handles_this]
            if not added_quant:
                legend_handles += [
                    Line2D([0], [0], color="0.4", lw=1.2, linestyle="--", label="5th percentile"),
                    Line2D([0], [0], color="0.4", lw=1.2, linestyle="-",  label="95th percentile"),
                ]
                legend_labels += ["5th percentile", "95th percentile"]
                added_quant = True

            ax.set_xlim(xmin, xmax)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
            if i == 0:
                ax.set_title(var_titles[var], fontsize=14, pad=8)
            if j == 0:
                ax.set_ylabel(f"{label}\nDensity", fontsize=14)
            if i == n_rows - 1:
                ax.set_xlabel(var_units[var], fontsize=10)

    plt.subplots_adjust(bottom=0.3)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels, loc="lower center",
            ncol=len(legend_labels), frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.01)
        )

    out_path = os.path.join(plots_dir, out_png)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[time-densities] Saved: {out_path}")

# ------------------------------------------------------------
# Entrypoint with boolean switches 
# ------------------------------------------------------------

def make_density_plots(time_perspective: bool = False) -> None:
    """
    Orchestrate density plots:
      - If time_perspective=False: rows = selected regions.
      - If time_perspective=True:  rows = global per quarter (Q1..Q4).
    """
    obs = load_all()
    if not time_perspective:
        rows, chosen_coords, xlims = prepare_regional(obs)
        plot_density_grid_regional(obs, rows, chosen_coords, xlims, out_png="obs_densities_regions.png")
    else:
        rows_time, xlims = prepare_time(obs)
        plot_density_grid_time(obs, rows_time, xlims, out_png="obs_densities_time.png")

def main() -> None:
    """
    Run the desired plots based on boolean switches. This file is intended to be
    executed directly.
    """
    # -------------------- switches --------------------
    MAKE_DENSITY_PLOTS = False
    TIME_PERSPECTIVE = False          # False: regional; True: global-by-quarter

    PLOT_SCATTER_GRID = False
    SCATTER_GROUND_TRUTH = "ifs_analysis"   # {"ifs_analysis", "era5"}
    SCATTER_MODEL_ROW = "GC"              # {"GC", "PW", "HRES"}

    PLOT_QQ_EXTREMES_GRID = False
    QQ_GROUND_TRUTH = "era5"                # {"ifs_analysis", "era5"}

    # -------------------- execution -------------------
    if MAKE_DENSITY_PLOTS:
        make_density_plots(time_perspective=TIME_PERSPECTIVE)

    if PLOT_SCATTER_GRID:
        plot_scatter_grid_by_lead(
            ground_truth=SCATTER_GROUND_TRUTH,
            model_row=SCATTER_MODEL_ROW,
            lead_times_days=lead_times_days,
        )

    if PLOT_QQ_EXTREMES_GRID:
        plot_qq_extremes_grid_by_lead_allmodels(
            ground_truth=QQ_GROUND_TRUTH,
            lead_times_days=lead_times_days,
        )

if __name__ == "__main__":
    main()
