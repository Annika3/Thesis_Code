import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors


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
    One-sided block permutation test on 1D score arrays.
    """
    if score_a.shape != score_b.shape or score_a.ndim != 1:
        raise ValueError('Inputs must be 1-D arrays of identical length')

    d = score_a - score_b
    n = d.size
    d_mean = d.mean()

    n_blocks = int(np.ceil((n + block_length - 1) / block_length))
    rng = np.random.default_rng(root_seed + int(seed_offset))
    block_signs = rng.choice((-1, 1), size=(b, n_blocks), replace=True)

    offsets = np.arange(b, dtype=int) % block_length
    block_indices = (np.arange(n)[None, :] + offsets[:, None]) // block_length

    signs = block_signs[np.arange(b)[:, None], block_indices]
    m = (signs * d[None, :]).mean(axis=1)
    p = np.mean(m <= d_mean)
    return p.astype(np.float32)


def run_block_permutation(
    raw_zarr_path: str,
    model_a: str,
    model_b: str,
    metric: str,
    variable: str,
    *,
    b: int = 1000,
    root_seed: int = 42,
) -> xr.DataArray:
    """
    Compute block-permutation p-values for two models and a metric.
    """
    ds = xr.open_zarr(raw_zarr_path, decode_timedelta=True)
    da_a = ds[metric].sel(model=model_a, variable=variable)
    da_b = ds[metric].sel(model=model_b, variable=variable)

    lon = da_a['longitude']
    lat = da_a['latitude']
    leads = da_a['prediction_timedelta']
    seed_vals = np.arange(lon.size * lat.size * leads.size, dtype='uint32')
    seed_da = xr.DataArray(
        seed_vals.reshape(lon.size, lat.size, leads.size),
        dims=('longitude','latitude','prediction_timedelta'),
        coords={'longitude': lon, 'latitude': lat, 'prediction_timedelta': leads},
    )

    lead_days = (leads / np.timedelta64(1,'D')).astype(int)
    block_length_da = (lead_days * 2).rename('block_length').broadcast_like(seed_da)

    p_da = xr.apply_ufunc(
        block_permutation_test,
        da_a, da_b, seed_da, block_length_da,
        input_core_dims=[['time'], ['time'], [], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32],
        kwargs={'b': b, 'root_seed': root_seed},
    )
    return p_da


def plot_p_value_map(
    p_da: xr.DataArray,
    model_pair: str,
    metric: str,
    variable: str,
    cmap: str = 'viridis',
    projection: ccrs.CRS = ccrs.Robinson()
) -> None:
    """
    Plot global map of p-values for each lead and save figure.
    """
    lons = p_da['longitude'].values
    lats = p_da['latitude'].values
    lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='xy')
    n_leads = p_da.sizes['prediction_timedelta']

    fig, axes = plt.subplots(
        1, n_leads,
        figsize=(4 * n_leads, 4),
        subplot_kw={'projection': projection},
        constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    vmin, vmax = 0, 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i, lead in enumerate(p_da['prediction_timedelta'].values):
        ax = axes[i]
        data = p_da.sel(prediction_timedelta=lead).transpose('latitude','longitude')
        mesh = ax.pcolormesh(
            lon_grid, lat_grid, data,
            cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(),
            shading='nearest'
        )
        ax.coastlines()
        day = int(np.timedelta64(int(lead), 'D') / np.timedelta64(1,'D'))
        ax.set_title(f'{variable} p-value\n{day}d, {metric}')

    cbar = fig.colorbar(mesh, ax=axes, orientation='horizontal', fraction=0.05, pad=0.04)
    cbar.set_label('p-value')
    fname = f'pmap_{model_pair}_{metric}_{variable}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ----------------------------- Boxplots of p-values ----------------------------- #

def plot_p_value_boxplots(
    p_vs_era5: dict[str, xr.DataArray],
    p_vs_ifs:  dict[str, xr.DataArray],
    lead_times: list[np.timedelta64],
    variables: list[str],
    titles:    list[str],
    output_file: str
) -> None:
    """
    Create boxplots of p-values for three model pairs under two ground truths.

    p_vs_era5: mapping 'label'->p_da for ERA5 truth
    p_vs_ifs:  mapping 'label'->p_da for IFS Analysis truth
    lead_times: list of lead timedelta values
    variables: list of variable names
    titles:    list of variable titles (same order)
    output_file: filename to save figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_vars = len(variables)
    fig, axes = plt.subplots(2, n_vars, figsize=(16, 10), sharey='row')

    offsets = {'GC_vs_PW': -0.3, 'GC_vs_HRES': 0.0, 'PW_vs_HRES': 0.3}
    colors = {'GC_vs_PW': 'tab:green', 'GC_vs_HRES': 'tab:blue', 'PW_vs_HRES': 'tab:orange'}

    # Top row: ERA5
    for j, var in enumerate(variables):
        ax = axes[0, j]
        x_ticks = []
        data = {label: [] for label in p_vs_era5}
        for lt in lead_times:
            x = int(lt / np.timedelta64(1,'D'))
            x_ticks.append(x)
            for label, p_da in p_vs_era5.items():
                vals = p_da.sel(prediction_timedelta=lt).values.flatten()
                data[label].append(vals)
        for label, vals_list in data.items():
            pos = [x + offsets[label] for x in x_ticks]
            ax.boxplot(vals_list, positions=pos,
                       widths=0.2, patch_artist=True,
                       boxprops=dict(facecolor=colors[label]),
                       medianprops=dict(color='black'),
                       showfliers=False)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])
        ax.set_title(titles[j], fontsize=22)
        if j==0: ax.set_ylabel('p-Value', fontsize=18)
        ax.set_xlabel('Lead Time [d]', fontsize=18)
        ax.tick_params(labelsize=16)

    # Bottom row: IFS
    for j, var in enumerate(variables):
        ax = axes[1, j]
        x_ticks = []
        data = {label: [] for label in p_vs_ifs}
        for lt in lead_times:
            x = int(lt / np.timedelta64(1,'D'))
            x_ticks.append(x)
            for label, p_da in p_vs_ifs.items():
                vals = p_da.sel(prediction_timedelta=lt).values.flatten()
                data[label].append(vals)
        for label, vals_list in data.items():
            pos = [x + offsets[label] for x in x_ticks]
            ax.boxplot(vals_list, positions=pos,
                       widths=0.2, patch_artist=True,
                       boxprops=dict(facecolor=colors[label]),
                       medianprops=dict(color='black'),
                       showfliers=False)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(x) for x in x_ticks])
        if j==0: ax.set_ylabel('p-Value', fontsize=18)
        ax.set_xlabel('Lead Time [d]', fontsize=18)
        ax.tick_params(labelsize=16)

    # Legends
    patches = [Patch(facecolor=colors[l], label=l.replace('_',' vs ')) for l in offsets]
    fig.legend(handles=patches, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=18)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)



if __name__ == '__main__':
    raw = 'score_data/raw_scores.zarr'
    lead_times = [np.timedelta64(d,'D') for d in (1,3,5,7,10)]
    variables = ['mean_sea_level_pressure', '2m_temperature', '10m_wind_speed']
    titles    = ['MSLP','T2M','WS10']

    # Loop over each variable and produce boxplots
    for var, title in zip(variables, titles):
        # Compute p-values for each model pair under ERA5 truth
        p_vs_era5 = {
            'GC_vs_PW': run_block_permutation(raw, 'graphcast', 'pangu', 'crps', var),
            'GC_vs_HRES': run_block_permutation(raw, 'graphcast', 'hres', 'crps', var),
            'PW_vs_HRES': run_block_permutation(raw, 'pangu',   'hres', 'crps', var),
        }
        # Compute p-values under IFS-Analysis truth
        p_vs_ifs = {
            'GC_vs_PW': run_block_permutation(raw, 'graphcast_operational', 'pangu_operational', 'crps', var),
            'GC_vs_HRES': run_block_permutation(raw, 'graphcast_operational', 'hres',  'crps', var),
            'PW_vs_HRES': run_block_permutation(raw, 'pangu_operational','hres','crps', var),
        }
        # Plot and save boxplots for this variable
        output_file = f'boxplot_pvalues_{var}.png'
        plot_p_value_boxplots(
            p_vs_era5, p_vs_ifs,
            lead_times,
            [var], [title],
            output_file
        )
        print(f"Saved boxplots to {output_file}")
