import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import inspect
from weatherbench2.metrics import _spatial_average

# --------------------------- Variables/ Definitions --------------------------------

SCORE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'score_data')

OBS_PATH       = 'data/ifs_analysis_64x32.zarr'   
# OBS_PATH     = 'data/era5_64x32.zarr'       
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

#def compute_lat_weighted_mean(df: pd.DataFrame, metric: str) -> float:
# use the _spatial_average function from the weatherbench2.metrics module    (before: data need to be converted into xarray)

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


def load_score_summary() -> pd.DataFrame: # später noch Spalten für weights function hinzufügen
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

    for each variable at a fixed lead_time, using only the specified forecast_model
    against obs_source.
    """
    # map CSV‐model keys to display names
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

    # determine how the model is named in the CSV
    if obs_source == 'ifs' and forecast_model in ('pangu','graphcast'):
        model_key = f"{forecast_model}_operational"
    else:
        model_key = forecast_model

    # helper to read & filter per metric
    def _load(var: str, metric: str, level: float|None) -> pd.DataFrame:
        fp = os.path.join(SCORE_DATA_DIR, f"{var}_lead{lead_time}d_{obs_source}.csv")
        df = pd.read_csv(fp)
        # restrict to the chosen model
        df = df[df['model'] == model_key]
        # drop NaNs in the metric column
        df = df.dropna(subset=[metric])
        # apply threshold/quantile filter if needed
        if metric == 'tw_pcs':
            df = df[df['t_quantile'] == t_level]
        elif metric == 'qw_pcs':
            df = df[df['q_value']    == q_level]
        return df

    # gather global vmin/vmax across all rows & cols
    all_vals = []
    for metric in ('pcs','tw_pcs','qw_pcs'):
        for var in VARIABLES:
            df_sub = _load(var, metric, None)
            if not df_sub.empty:
                all_vals.append(df_sub.groupby(['lat','lon'])[metric].mean().values)
    if not all_vals:
        raise RuntimeError("No data for that model/levels/lead_time.")
    flat = np.concatenate([v.flatten() for v in all_vals])
    vmin, vmax = np.nanmin(flat), np.nanmax(flat)

    # build the panel
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

    for i, (metric, label, lvl) in enumerate(row_info):
        for j, var in enumerate(VARIABLES):
            ax = axes[i, j]
            df_sub = _load(var, metric, lvl)
            if df_sub.empty:
                ax.set_facecolor('lightgray')
                ax.coastlines()
                ax.set_title('no data')
                continue

            dfm = df_sub.groupby(['lat','lon'])[metric].mean().reset_index()
            lons = np.sort(dfm['lon'].unique())
            lats = np.sort(dfm['lat'].unique())
            grid = dfm.pivot(index='lat', columns='lon', values=metric).values

            lon_g, lat_g = np.meshgrid(lons, lats, indexing='xy')
            mesh = ax.pcolormesh(
                lon_g, lat_g, grid,
                cmap=cmap,
                norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
                shading='nearest',
                transform=ccrs.PlateCarree()
            )
            ax.coastlines(linewidth=0.5)

            if i == 0:
                ax.set_title(NAME_MAP[var], fontsize=12)
            if j == 0:
                ax.text(-0.1, 0.5, label,
                        transform=ax.transAxes,
                        va='center', ha='right',
                        rotation=90, fontsize=10)

    # super‐title
    model_title = BASE_MODEL_TITLES.get(forecast_model, forecast_model)
    if obs_source == 'ifs' and forecast_model in ('pangu','graphcast'):
        model_title += '‑Operational'

    fig.suptitle(
        f"Lead {lead_time}d — {model_title} vs {obs_source.upper()}",
        y=0.98, fontsize=14
    )

    # shared colorbar
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(),
                        orientation='horizontal',
                        fraction=0.05, pad=0.02)
    cbar.set_label('PCS variants', fontsize=12)

    # save & show
    fname = (f"pcs_twq_panel_lead{lead_time}d_"
             f"{forecast_model}_vs_{obs_source}_"
             f"t{t_level:.2f}_q{q_level:.2f}.png")
    fig.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.show()

def plot_threshold_map(q_level: float = 0.9) -> None:
    """
    Plot and save a map of the grid‑point quantile thresholds for each variable
    at the specified q_level (e.g. 0.99 for the 99th percentile), indicating
    which observation source was used in the big title.
    """
    # determine ground‑truth name from OBS_PATH
    gt_name = os.path.basename(OBS_PATH).split('_')[0]

    # load & compute thresholds
    obs    = xr.open_zarr(OBS_PATH, decode_timedelta=True)
    period = slice('2020-01-01', '2020-12-31')
    thresh = {
        var: (
            obs[var]
            .sel(time=period)
            .chunk({'time': -1})
            .quantile(q_level, dim='time')
        )
        for var in VARIABLES
    }

    # set up figure
    proj   = ccrs.PlateCarree()
    nvars  = len(thresh)
    fig, axs = plt.subplots(
        1, nvars,
        figsize=(5.5 * nvars, 4.5),
        subplot_kw={'projection': proj}
    )
    if nvars == 1:
        axs = [axs]

    pct = int(q_level * 100)
    for ax, (var, t_val) in zip(axs, thresh.items()):
        lon = t_val['longitude'].values
        lat = t_val['latitude'].values
        lon_g, lat_g = np.meshgrid(lon, lat, indexing='xy')

        norm = mcolors.Normalize(
            vmin=float(np.nanmin(t_val)),
            vmax=float(np.nanmax(t_val))
        )

        mesh = ax.pcolormesh(
            lon_g, lat_g, t_val.values.T,
            cmap='viridis', norm=norm, shading='nearest',
            transform=proj
        )
        ax.coastlines()
        ax.set_title(f'{var}')

        cbar = plt.colorbar(
            mesh, ax=ax,
            orientation='horizontal',
            pad=0.05, shrink=0.8
        )
        cbar.ax.set_xlabel(t_val.attrs.get('units', ''))

        if var == 'mean_sea_level_pressure':
            vmin_pa = int(np.floor(norm.vmin / 1000)) * 1000
            vmax_pa = int(np.ceil(norm.vmax / 1000)) * 1000
            norm.vmin, norm.vmax = vmin_pa, vmax_pa
            mesh.set_norm(norm)
            ticks = np.arange(vmin_pa, vmax_pa + 1, 1000)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{t//1000:d}k' for t in ticks])

    # overall title includes quantile and ground‑truth source
    fig.suptitle(
        f'Grid‑point {pct}th‑percentile Thresholds (ground truth: {gt_name})',
        y=0.98
    )

    out_fname = f'threshold_map_{gt_name}_q{pct:d}.png'
    fig.savefig(os.path.join(PLOTS_DIR, out_fname), dpi=150, bbox_inches='tight')
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
    5×3 panel of global maps for `metric_base` (with optional `level`)
    of `forecast_model` against `obs_source`.

    metric_base:
      - 'pc' or 'pcs'       → pure score (ignores level)
      - 'tw_pc','tw_pcs'     → threshold‐weighted (requires level)
      - 'qw_pc','qw_pcs'     → quantile‐weighted   (requires level)

    level         : float threshold/quantile (e.g. 0.9, 0.95), or None for pure metrics
    obs_source    : 'ifs' or 'era5'
    forecast_model: 'hres', 'pangu', or 'graphcast'
    """
    # Only append “_operational” when comparing to IFS
    if obs_source == "ifs" and forecast_model in ("pangu", "graphcast"):
        model_key = f"{forecast_model}_operational"
    else:
        model_key = forecast_model

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        dfm = df[df["model"] == model_key]
        # pure base metrics: include all non‐NaN metric_base values
        if metric_base in ("pc", "pcs"):
            return dfm.dropna(subset=[metric_base])
        # weighted variants require level
        if level is None:
            raise ValueError(f"Must supply `level` for metric '{metric_base}'")
        if metric_base.startswith("tw_"):
            return dfm[dfm["t_quantile"] == level]
        if metric_base.startswith("qw_"):
            return dfm[dfm["q_value"]    == level]
        raise ValueError(f"Unknown metric_base '{metric_base}'")

    # gather vmin/vmax
    all_vals = []
    for lt in LEAD_TIMES:
        for var in VARIABLES:
            fn   = f"{var}_lead{lt}d_{obs_source}.csv"
            path = os.path.join(SCORE_DATA_DIR, fn)
            if not os.path.exists(path):
                continue
            sub = _filter(pd.read_csv(path))
            if not sub.empty:
                all_vals.append(sub.groupby(["lat","lon"])[metric_base].mean().values)

    if not all_vals:
        suffix = "" if metric_base in ("pc","pcs") else f" (level={level})"
        raise RuntimeError(f"No data for {metric_base.upper()}{suffix} with model_key='{model_key}'")

    flat = np.concatenate([v.flatten() for v in all_vals])
    vmin, vmax = np.nanmin(flat), np.nanmax(flat)

    # build panel
    fig, axes = plt.subplots(
        len(LEAD_TIMES), len(VARIABLES),
        figsize=(4*len(VARIABLES), 3*len(LEAD_TIMES)),
        subplot_kw={"projection": projection},
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    for i, lt in enumerate(LEAD_TIMES):
        for j, var in enumerate(VARIABLES):
            ax = axes[i, j]
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
            lon_g, lat_g = np.meshgrid(
                np.sort(sub["lon"].unique()),
                np.sort(sub["lat"].unique())
            )
            mesh = ax.pcolormesh(
                lon_g, lat_g, grid,
                transform=ccrs.PlateCarree(),
                cmap=cmap, vmin=vmin, vmax=vmax,
                shading="nearest"
            )
            ax.coastlines(linewidth=0.5)
            if i == 0:
                ax.set_title(NAME_MAP[var], fontsize=12)
            if j == 0:
                ax.text(-0.1, 0.5, f"Lead {lt}d",
                        transform=ax.transAxes, va="center",
                        ha="right", rotation=90, fontsize=10)

    # super‐title & colorbar
    lvl_str    = "" if metric_base in ("pc","pcs") else f"={level}"
    mt         = BASE_MODEL_TITLES[forecast_model]
    if obs_source == "ifs" and forecast_model in ("pangu","graphcast"):
        mt += "-Operational"
    fig.suptitle(
        f"{metric_base.upper()}{lvl_str} for {mt} vs {obs_source.upper()}",
        y=0.98, fontsize=14
    )
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.05, pad=0.02)
    cbar.set_label(metric_base.upper(), fontsize=12)

    # save & show
    suffix = "" if metric_base in ("pc","pcs") else f"_{level}"
    out    = f"panel_{metric_base}{suffix}_{forecast_model}_vs_{obs_source}.png"
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
    Line‐panel: rows = [
        BASE,
        TW_BASE @ t_level,
        QW_BASE @ q_level
    ],
    cols = VARIABLES, one line per model, filtered to obs_source.

    base_metric: 'pc' or 'pcs'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # build the exact column names (uppercase to match summary_df)
    BASE = base_metric.upper()                  # e.g. "PC" or "PCS"
    TW   = f"tw_{BASE}_{t_level}"               # e.g. "tw_PC_0.95"
    QW   = f"qw_{BASE}_{q_level}"               # e.g. "qw_PC_0.95"

    metrics    = [BASE, TW, QW]
    row_labels = [
        BASE,
        f"TW_{BASE} (t={t_level})",
        f"QW_{BASE} (q={q_level})"
    ]

    # filter to this ground-truth
    df = summary_df[summary_df["obs_source"] == obs_source]

    # sanity check
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        print(f"⚠️ Missing metrics in summary_df: {missing}")
        return

    models = sorted(df["model"].unique())
    cmap   = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models)}

    name_map = {
        "mean_sea_level_pressure": "MSLP",
        "2m_temperature":           "T2M",
        "10m_wind_speed":           "WS10",
    }

    n_rows, n_cols = len(metrics), len(VARIABLES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True,
        sharey=False,               # <-- no shared y
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

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
            if i == n_rows - 1:
                ax.set_xlabel("Lead Time [d]")
            if j == 0:
                ax.set_ylabel(row_labels[i])
            if i == 0:
                ax.set_title(name_map.get(var, var))
            ax.grid(alpha=0.3)

    # single legend
    handles = [
        Line2D([], [], marker="o", ls="-", ms=6, color=colors[m])
        for m in models
    ]
    labels = [
        m.replace("_operational","")
         .replace("graphcast","GraphCast")
         .replace("pangu","Pangu-Weather")
         .replace("hres","IFS-HRES")
        for m in models
    ]
    fig.legend(
        handles, labels,
        ncol=min(len(labels), 6),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False
    )

    fig.suptitle(
        f"Model Comparison – {BASE} variants\n"
        f"(obs: {obs_source.upper()}, t={t_level}, q={q_level})",
        y=0.98
    )

    out_fname = (
        f"model_panel_{BASE}_t{t_level}_q{q_level}_{obs_source}.png"
    )
    fig.savefig(os.path.join(PLOTS_DIR, out_fname),
                dpi=150, bbox_inches="tight")
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
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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
        bbox_to_anchor=(0.5, -0.02),
        frameon=False
    )

    # Title and save
    fig.suptitle(
        f"Sensitivity of {BASE} to {weight_type.upper()} levels {levels}\n"
        f"(obs: {obs_source.upper()})",
        y=0.98
    )
    out = (
        f"sensitivity_{BASE}_{weight_type}_"
        f"{'_'.join(f'{lvl:.2f}' for lvl in levels)}"
        f"_vs_{obs_source}.png"
    )
    fig.savefig(os.path.join(PLOTS_DIR, out), dpi=150, bbox_inches="tight")
    plt.show()


def plot_performance_diff_panel(
    summary_df: pd.DataFrame,
    obs_source: str,
    ai_model: str,
    base_metric: str = "pc",
    metric_type: str = "tw",
    diff_type: str = "absolute"  # or 'percent'
) -> None:
    """
    3-row panel: per variable columns, lead_time on x-axis.
    Row 1: BASE metric curves (HRES vs AI).
    Row 2: BASE metric difference (HRES vs AI) as specified by diff_type ('absolute' or 'percent').
    Row 3: Differences for each level of metric_type, overlaid with legend, by diff_type.
    Rows 2 and 3 share identical y-axis limits for comparability.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # validate inputs
    BASE = base_metric.lower()
    if BASE not in ("pc", "pcs"):
        raise ValueError("base_metric must be 'pc' or 'pcs'")
    BASE_UP = BASE.upper()

    if metric_type not in ("tw", "qw"):
        raise ValueError("metric_type must be 'tw' or 'qw'")

    if diff_type not in ("absolute", "percent"):
        raise ValueError("diff_type must be 'absolute' or 'percent'")

    # determine model keys
    hres_key = "hres"
    if obs_source == "ifs" and ai_model in ("pangu", "graphcast"):
        ai_key = f"{ai_model}_operational"
    else:
        ai_key = ai_model

    ai_label = BASE_MODEL_TITLES.get(ai_model, ai_model.title())

    # filter to obs_source
    df = summary_df[summary_df["obs_source"] == obs_source]

    # detect levels dynamically
    prefix = f"{metric_type}_{BASE_UP}_"
    levels = sorted({float(col.split("_")[-1]) for col in df.columns if col.startswith(prefix)})

    # setup panel
    n_rows, n_cols = 3, len(VARIABLES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True,
        constrained_layout=True
    )
    axes = np.atleast_2d(axes)

    # Row 1: BASE curves
    for j, var in enumerate(VARIABLES):
        ax = axes[0, j]
        sub = df[df["variable"] == var]
        for model_key, label in [(hres_key, BASE_MODEL_TITLES['hres']), (ai_key, ai_label)]:
            seg = sub[sub['model'] == model_key]
            if seg.empty:
                continue
            ax.plot(seg['lead_time'], seg[BASE_UP], marker='o', ms=5, label=label)
        if j == 0:
            ax.set_ylabel(BASE_UP)
        ax.set_title(NAME_MAP.get(var, var))
        ax.grid(alpha=0.3)
        if j == n_cols - 1:
            ax.legend(loc='best')

    # prepare diff data
    diff_values = []
    base_diffs = {}
    level_diffs = {lvl: {} for lvl in levels}

    # compute diffs
    for var in VARIABLES:
        sub = df[df['variable'] == var]
        # base metric
        h = sub[sub['model'] == hres_key][['lead_time', BASE_UP]]
        a = sub[sub['model'] == ai_key][['lead_time', BASE_UP]]
        m_base = h.merge(a, on='lead_time', suffixes=('_hres','_ai')).dropna()
        if not m_base.empty:
            if diff_type == 'absolute':
                if BASE == 'pc':
                    m_base['diff'] = m_base[f'{BASE_UP}_hres'] - m_base[f'{BASE_UP}_ai']
                else:
                    m_base['diff'] = m_base[f'{BASE_UP}_ai'] - m_base[f'{BASE_UP}_hres']
            else:  # percent
                if BASE == 'pc':
                    m_base['diff'] = ((m_base[f'{BASE_UP}_hres'] - m_base[f'{BASE_UP}_ai']) / m_base[f'{BASE_UP}_hres']) * 100
                else:
                    m_base['diff'] = ((m_base[f'{BASE_UP}_ai'] - m_base[f'{BASE_UP}_hres']) / m_base[f'{BASE_UP}_hres']) * 100
            base_diffs[var] = m_base
            diff_values.extend(m_base['diff'].values)

        # levels
        for lvl in levels:
            col = f"{metric_type}_{BASE_UP}_{lvl}"
            h_l = sub[sub['model'] == hres_key][['lead_time', col]]
            a_l = sub[sub['model'] == ai_key][['lead_time', col]]
            m_lvl = h_l.merge(a_l, on='lead_time', suffixes=('_hres','_ai')).dropna()
            if not m_lvl.empty:
                if diff_type == 'absolute':
                    if BASE == 'pc':
                        m_lvl['diff'] = m_lvl[f'{col}_hres'] - m_lvl[f'{col}_ai']
                    else:
                        m_lvl['diff'] = m_lvl[f'{col}_ai'] - m_lvl[f'{col}_hres']
                else:
                    if BASE == 'pc':
                        m_lvl['diff'] = ((m_lvl[f'{col}_hres'] - m_lvl[f'{col}_ai']) / m_lvl[f'{col}_hres']) * 100
                    else:
                        m_lvl['diff'] = ((m_lvl[f'{col}_ai'] - m_lvl[f'{col}_hres']) / m_lvl[f'{col}_hres']) * 100
                level_diffs[lvl][var] = m_lvl
                diff_values.extend(m_lvl['diff'].values)

    # common y-limits
    y_min, y_max = (min(diff_values), max(diff_values)) if diff_values else (None, None)

    # Row 2: BASE metric difference
    for j, var in enumerate(VARIABLES):
        ax = axes[1, j]
        m = base_diffs.get(var)
        ylabel = f"{BASE_UP} difference"
        if m is None or m.empty:
            ax.set_facecolor('lightgray')
            ax.set_title('no data')
        else:
            ax.plot(m['lead_time'], m['diff'], marker='o', ms=5, label=ylabel)
            ax.legend(loc='best')
        if j == 0:
            ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if y_min is not None:
            ax.set_ylim(y_min, y_max)
        if 1 == n_rows - 1:
            ax.set_xlabel('Lead Time [d]')

    # Row 3: differences for each level
    for j, var in enumerate(VARIABLES):
        ax = axes[2, j]
        for lvl, diffs in level_diffs.items():
            m = diffs.get(var)
            if m is None or m.empty:
                continue
            label = f"{metric_type.upper()}={lvl}"
            ax.plot(m['lead_time'], m['diff'], marker='o', ms=5, label=label)
        if j == 0:
            ax.set_ylabel(ylabel)
        ax.set_xlabel('Lead Time [d]')
        ax.grid(alpha=0.3)
        if y_min is not None:
            ax.set_ylim(y_min, y_max)
        ax.legend(loc='best')

    # title & save
    fig.suptitle(
        f"Performance Comparison – {BASE_UP} + {diff_type} difference of {ai_label} over IFS-HRES\n"
        f"(obs: {obs_source.upper()}, metric: {metric_type.upper()})",
        y=0.98
    )
    out_filename = f"perf_diff_panel_{BASE}_{metric_type}_{diff_type}_{ai_model}_vs_hres_{obs_source}.png"
    fig.savefig(os.path.join(PLOTS_DIR, out_filename), dpi=150, bbox_inches='tight')
    plt.show()




# --------------------------- Main -------------------------

if __name__ == '__main__':

    ## PANELS
    RUN_PANEL_AVERAGED_GRIDPOINTS = False
    RUN_PANEL_SENSITIVITY = False
    RUN_PERF_DIFF_PANEL = False

    ## MAPS    
    RUN_THRESHOLD_MAP   = False    # threshold-map
    THRESHOLD_LEVEL = 0.9 # level to be depicted in threshold map

    RUN_3_SCORES_HEATMAP = False      # 3×3-PCS-Panel (comparison )

    RUN_FIXED_SCORE_HEATMAP = True  # 5×3-PCS-Panel with all variables & lead times

    ## DATA
    LOAD_SUMMARY = False  # recompute summary or load existing 

    if LOAD_SUMMARY:
            # recompute & overwrite
            summary_df = load_score_summary()
    else:
            # load previously saved summary
            csv_path = os.path.join(SCORE_DATA_DIR, 'score_summary.csv')
            summary_df = pd.read_csv(csv_path)

    ## MAPS Plotting

    if RUN_THRESHOLD_MAP:
        plot_threshold_map(THRESHOLD_LEVEL)

    # Choose which column to plot:
    #   - base metrics:   "pc" or "pcs"        (level is ignored)
    #   - quantile‑metrics: "qw_pc" or "qw_pcs" (must supply level)
    #   - threshold‑metrics: "tw_pc" or "tw_pcs" (must supply level)
    COLUMN     = "tw_pcs"
    LEVEL      = 0.95       # e.g. 0.90, 0.95, 0.99; ignored if COLUMN is "pc"/"pcs"
    OBS_SOURCE = "ifs"      # or "era5"

    if RUN_FIXED_SCORE_HEATMAP:
        plot_spatial_map(
        metric_base   = "tw_pcs", # "pcs", "tw_pcs", "qw_pcs"
        level          = 0.95,
        obs_source     = "era5",
        forecast_model = "graphcast"
    )

    # extra Funktion für ERA5 als PC-ref value 
    # (hier PCS Spalten überschreiben von den alten csv Dateien aus Ordner) -> Output soll Heatmap sein

    if RUN_3_SCORES_HEATMAP:
        plot_pcs_twq_map(
            lead_time       = 5,
            t_level         = 0.95,
            q_level         = 0.95,
            obs_source      = "era5"   ,       # "ifs" or "era5"
            forecast_model  = "graphcast" ,  # "hres", "pangu", or "graphcast"
        )

    ## PANELS Plotting
    
    if RUN_PANEL_AVERAGED_GRIDPOINTS:
        plot_model_panel(
            summary_df  = summary_df,
            obs_source  = "era5"   , # ifs or "era5"
            t_level     = 0.99,
            q_level     = 0.99,
            base_metric = "pcs"    # "pc" or "pcs"
        )

    if RUN_PANEL_SENSITIVITY:
        plot_sensitivity_panel(
            summary_df  = summary_df,
            obs_source  = "era5",  # ifs or "era5"
            base_metric = "pcs",    # or "pcs"
            weight_type = "tw"     # or "tw"
        )
    

    if RUN_PERF_DIFF_PANEL:
        # choose obs (“ifs” or “era5”), AI model (“graphcast” or “pangu”),
        # base_metric (“pc” or “pcs”) and metric_type (“tw” or “qw”)
        plot_performance_diff_panel(
            summary_df=summary_df,
            obs_source="era5", # “ifs” or “era5”
            ai_model="graphcast", # “graphcast” or “pangu”
            base_metric="pc", #“pc” or “pcs”
            metric_type="qw", # “tw” or “qw”
            diff_type = "absolute"  # "absolute" or "percent"
        )