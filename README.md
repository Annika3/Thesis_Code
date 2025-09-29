# Fair Benchmarking of AI and Numerical Weather Models for Extremes  

This repository contains the code and resources accompanying the bachelor’s thesis *“Fair Benchmarking of AI and Numerical Models for Extreme Weather Events”* (KIT, 2025). The thesis extends the **Potential Continuous Ranked Probability Score (CRPS)** to **threshold- and quantile-weighted variants**, enabling fair evaluation of **extreme weather forecasts**.  

The project compares **AI-based weather prediction (AIWP)** models (GraphCast, Pangu-Weather) with the **ECMWF high-resolution model (HRES)**. Evaluations cover **mean sea level pressure (MSLP)**, **2-meter temperature (T2M)**, and **10-meter wind speed (WS10)** across multiple lead times and reference datasets (ERA5, IFS analysis).  


## Repository Structure  

- **simulation_study/**  
  Contains the simulation study introduced in Chapter 4 of the thesis.  
  - `run_experiments.py` → runs the synthetic experiments and produces evaluation metrics  
  - `plots/` → stores figures generated from the simulation study  

- **wb2/**  
  Application of the extended Potential CRPS framework to the WeatherBench 2 dataset (Chapter 5 of the thesis).  
  - `load_data.py` → data loading routines for WeatherBench 2 forecasts and references  
  - `metric_functions.py` → implementation of PC, PCS, and weighted variants (twPC, twPCS, qwPC, qwPCS)  
  - `scores.py` → computation of potential scores for different models  
  - `block_permutation_test.py` → statistical significance testing 
  - `exploratory_analysis.py` → diagnostic plots and data exploration  
  - `sensitivity_analysis.py` → evaluation of sensitivity to threshold and quantile choices  
  - `visual_analysis.py` → aggregated, spatial and regional performance plots
  - `plots/` → generated figures for thesis results  

- **visualization_thesis.py**  
  Script to reproduce CRPS figure included in the thesis.  

- **requirements.txt**  
  List of pinned Python packages required to run the experiments and analyses.  


## Getting Started
1. Clone the repository:  
   git clone https://github.com/Annika3/Thesis_Code.git  
   cd Thesis_Code/potential-crps-main  

2. Install dependencies (Python ≥ 3.10):  
   pip install -r requirements.txt  

3. Reproduce results:  
   - `simulation_study/` → run synthetic experiments  
   - `wb2/` → reproduce WeatherBench 2 case study  
   - `visualization_thesis.py` → regenerate CRPS plot used in the thesis  
