# WeatherBench 2 Case Study (`wb2/`)

This folder contains the implementation of the case study described in Chapter 5 of the thesis.  

---

## Workflow

The scripts are organized sequentially. Running them in order reproduces the complete pipeline:

1. **01_load_data.py**  
   Loads WeatherBench 2 forecasts and reference data.  
   Two options are available and can be set in the `__main__` block:
   - `load_classic_data` loads the standard WeatherBench 2 evaluation datasets (forecasts + IFS/ERA5 references)  
   - `build_era5_climatology` additionally constructs ERA5 climatology forecasts as an alternative reference. The climatology construction is based on code taken directly from the original repository: https://github.com/tobiasbiegert/potential-crps/blob/main/wb2/construct_era5_climatology_forecasts.py  

2. **02_scores.py**  
   Computes the Potential CRPS (PC), Potential CRPS Skill Score (PCS), and their threshold- and quantile-weighted variants (twPC, qwPC, twPCS, qwPCS).  
   The script evaluates forecasts from GraphCast, Pangu-Weather, and HRES against ERA5 and IFS references over all grid points, variables, and lead times.  
   The implementation fits isotonic distributional regression (IDR) at each grid point, generates probabilistic forecasts, and derives scores for both lower and upper distributional tails.  
   Several modes are available:
   - **Classical scores**: gridpoint-level PC/PCS, twPC/qwPC with fixed thresholds and quantiles.  
   - **Gaussian-CDF weighted scores**: smooth weighting functions around thresholds.  
   - **Regional threshold scores**: pooled regional quantiles used as thresholds for tail evaluation.  
   - **Raw time-resolved scores**: per-initialization and per-gridpoint outputs required for block permutation testing.  
   Results are written as CSV, PKL, or Zarr files in the `score_data/` directory, with naming conventions reflecting variable, lead time, observation source, and scoring mode.  

3. **03_block_permutation_test.py**  
   Performs one-sided **block permutation tests** to assess significance of model score differences under temporal dependence and generates (i) grid-wise **p-value boxplots** and (ii) **regional scorecards** with significance shading.  We closely followed the approach in https://github.com/tobiasbiegert/potential-crps/blob/main/wb2/create_plots.py, which creates the boxplots used in the original analysis.  
   Switches in `main()`:  
   - `MAKE_BOXPLOTS=True` (requires `score_data/raw_scores_for_permutation_{era5,ifs}.zarr`)  
   - `MAKE_REGIONAL_SCORECARDS=True` (requires `score_data/regional_timeseries_{era5,ifs}.csv`)  

4. **03_exploratory_analysis.py**  
   Generates observation diagnostics and sanity plots: (i) **regional/global density** comparisons of ERA5 vs. IFS analysis, (ii) **scatter grids** (truth vs. model by lead), and (iii) **QQ plots** for extremes across variables and lead times.  
   Inputs are Zarr stores in `./data/{era5,ifs_analysis}_64x32.zarr`; figures are written to `./plots/`.  
   Switches in `main()`:  
   - `MAKE_DENSITY_PLOTS=True`  
   - `PLOT_SCATTER_GRID=True`, plus `SCATTER_GROUND_TRUTH` and `SCATTER_MODEL_ROW`  
   - `PLOT_QQ_EXTREMES_GRID=True`, plus `QQ_GROUND_TRUTH`  

5. **03_sensitivity_analysis.py**  
   Generates sensitivity/interpretation plots for (TW & QW) PC/PCS and BS/QS at a grid point.  
   Requires Zarr files in `./data`.  
   Switches in `__main__`:  
   - `RUN_QUANTILE_GRID=True` with `SCORE`  
   - `RUN_INTERPRETATION_GRID=True`  
   - `RUN_INTERPRETATION_GRID_ZOOM_LAST10=True`  
   - `RUN_INTERPRETATION_GRID_SMALL=True`  
   - `RUN_INTERPRETATION_GRID_SMALL_ZOOM=True`  
   - `RUN_BS_QS_GRIDPOINT=True` with `SP_SCORE_TYPE`, `SP_GROUND_TRUTH`, and location  
   - `RUN_BOTH_SCORES_GRIDPOINT_FIXED_LEAD=True` with `FL_*` parameters  

6. **03_visual_analysis.py** / **xx_visual_analysis.py**  
   Generates global maps and lead-time panels for PC/PCS and their tail-weighted variants, with optional **climatology-referenced** (ERA5 rolling) PCS.  
   Caches reference PC(0) per gridpoint in `score_data/era5_ref_pc/` to avoid recomputation.  

   **Requires**  
   - Zarrs in `./data`: `era5_64x32.zarr`, `ifs_analysis_64x32.zarr`, `era5_climatology_forecasts.zarr`  
   - Score CSVs in `./score_data/*.csv`  
   - Outputs saved to `./plots/`  

   Switches in `__main__`:  
   - `RUN_ERA5_CLIM_PCS_HEATMAP=True` (with `metric_base`, `obs_source`, `forecast_model`)  
   - `RUN_FIXED_SCORE_HEATMAP=True` (with `metric_base`, `level`)  
   - `RUN_3_SCORES_HEATMAP=True` (with `lead_time`, thresholds)  
   - `RUN_THRESHOLD_MAP=True` (with `THRESHOLD_LEVEL`, `obs_source`)  
   - `RUN_PANEL_AVERAGED_GRIDPOINTS=True` (with `base_metric`, thresholds)  
   - `RUN_PANEL_SENSITIVITY=True` (with `base_metric`, `weight_type`)  
   - `RUN_PERF_DIFF_PANEL=True` (with `ai_model`, `base_metric`, `metric_type`, `diff_type`)  
   - `RUN_PERF_DIFF_SUMMARY_PANEL=True` (with `ai_model`, `diff_type`)  
   - `RUN_GAUSSIAN_VS_TW_PANEL=True` (with `t_level`, `obs_source`, `score`)  
   - `RUN_GAUSSIAN_EPS_VS_TW_PANEL=True` (with Gaussian parameters)  

---

## Supporting Script

- **metric_functions.py**  
  Contains the implementation of PC, PCS, twPC, qwPC, and the corresponding skill scores, including unconditional and conditional climatology references.  
