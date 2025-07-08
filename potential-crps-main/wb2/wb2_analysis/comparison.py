import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
# -----------------------------------------------------------
SCORE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'score_data')

OBS_PATH       = 'data/ifs_analysis_64x32.zarr'   
# OBS_PATH     = 'data/era5_64x32.zarr'       
PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

OBS_SOURCES    = ['era5', 'ifs']
VARIABLES      = ['2m_temperature', 'mean_sea_level_pressure', '10m_wind_speed']
LEAD_TIMES     = [1, 3, 5, 7, 10]
SCORE_COLS     = ['pc', 'pcs', 'tw_pc', 'tw_pcs', 'qw_pc', 'qw_pcs']


def load_score_summary() -> pd.DataFrame:
    rows = []
    for obs in OBS_SOURCES:
        for var in VARIABLES:
            for lt in LEAD_TIMES:
                csv = f'{var}_lead{lt}d_{obs}.csv'
                path = os.path.join(SCORE_DATA_DIR, csv)
                if not os.path.exists(path):
                    print(f'⚠  Missing file: {path}')
                    continue
                df = pd.read_csv(path)
                means = (
                    df.groupby('model')[SCORE_COLS].mean()
                      .reset_index()
                      .assign(obs_source=obs, variable=var, lead_time=lt)
                )
                rows.append(means)
    return pd.concat(rows, ignore_index=True)

def load_thresholds() -> dict[str, xr.DataArray]:
    obs     = xr.open_zarr(OBS_PATH, decode_timedelta=True)
    period  = slice('2020-01-01', '2020-12-31')
    thresh  = {}
    for var in VARIABLES:
        arr = obs[var].sel(time=period).chunk({'time': -1}).quantile(0.9, 'time')
        thresh[var] = arr
    return thresh


# ------------------ plot-functions -----------------------
def plot_threshold_map(thresholds: dict[str, xr.DataArray]) -> None:
    proj   = ccrs.PlateCarree()
    nvars  = len(thresholds)
    fig, axs = plt.subplots(1, nvars, figsize=(5.5*nvars, 4.5),
                            subplot_kw={'projection': proj})
    if nvars == 1:
        axs = [axs]

    for ax, (var, t_val) in zip(axs, thresholds.items()):
        lon, lat = t_val['longitude'].values, t_val['latitude'].values
        lon_g, lat_g = np.meshgrid(lon, lat, indexing='xy')
        norm = mcolors.Normalize(vmin=float(np.nanmin(t_val)),
                                 vmax=float(np.nanmax(t_val)))
        mesh = ax.pcolormesh(lon_g, lat_g, t_val.values.T,
                             cmap='viridis', norm=norm, shading='nearest',
                             transform=proj)
        ax.coastlines(); ax.set_title(f'90-percentile {var}')
        cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal',
                            pad=0.05, shrink=0.8)
        cbar.ax.set_xlabel(t_val.attrs.get('units', ''))
        if var == 'mean_sea_level_pressure':
            # Tick-Labels 
            vmin_pa = int(np.floor(norm.vmin / 1000))*1000
            vmax_pa = int(np.ceil (norm.vmax / 1000))*1000
            norm.vmin, norm.vmax = vmin_pa, vmax_pa
            mesh.set_norm(norm)
            ticks = np.arange(vmin_pa, vmax_pa+1, 1000)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{t//1000:d}k' for t in ticks])

    fig.suptitle('Grid-point 0.9 Quantile Thresholds', y=0.98)
   
    fig.savefig(os.path.join(PLOTS_DIR, 'threshold_map.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

def plot_model_panel(summary_df: pd.DataFrame) -> None:
    metrics = ['pcs', 'tw_pcs', 'qw_pcs']
    models  = sorted(summary_df['model'].unique())
    cmap    = plt.get_cmap('tab10')
    colors  = {m: cmap(i % 10) for i, m in enumerate(models)}

    fig, axs = plt.subplots(len(metrics), len(VARIABLES),
                            figsize=(16, 11),
                            sharex=True, sharey='row')
    name_map = {'2m_temperature':'T2M', 'mean_sea_level_pressure':'MSLP', '10m_wind_speed':'WS10'}

    for r, metric in enumerate(metrics):
        for c, var in enumerate(VARIABLES):
            ax     = axs[r, c]
            subset = summary_df.query("variable == @var")
            for model in models:
                seg = subset.query("model == @model")
                if seg.empty: continue
                ax.plot(seg['lead_time'], seg[metric],
                        marker='o', ms=5, color=colors[model], label=model)
            if r == len(metrics)-1: ax.set_xlabel('Lead Time [d]')
            if c == 0:              ax.set_ylabel(metric.upper())
            if r == 0:              ax.set_title(name_map.get(var, var))
            ax.grid(alpha=0.3)

    # Legende
    handles = [Line2D([],[], marker='o', ls='-', ms=6, color=colors[m]) for m in models]
    labels  = [m.replace('_operational','').replace('graphcast','GraphCast')
                  .replace('pangu','Pangu-Weather').replace('hres','IFS-HRES')
               for m in models]
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, top=0.90)
    fig.legend(handles, labels, ncol=min(len(labels),6),
               loc='lower center', bbox_to_anchor=(0.5,0.02), frameon=False)
    fig.suptitle('Model comparison – PCS variants', y=0.96)

    fig.savefig(os.path.join(PLOTS_DIR, 'pcs_panel.png'),
                dpi=150, bbox_inches='tight')
    plt.show()

# --------------------------- main -------------------------
if __name__ == '__main__':
    
    RUN_MAP   = True      # 90%-threshold-map
    RUN_PANEL = True      # 3×3-PCS-Panel
  
    # only load data when needed
    summary_df = load_score_summary() if RUN_PANEL else None
    thresholds = load_thresholds()    if RUN_MAP   else None

    if RUN_MAP:
        plot_threshold_map(thresholds)

    if RUN_PANEL:
        plot_model_panel(summary_df)