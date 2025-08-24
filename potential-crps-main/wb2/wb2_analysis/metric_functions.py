import numpy as np
import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from isodisreg import idr
import os
import pickle
from joblib import Parallel, delayed
import math
from typing import Callable


def pc(prob_pred, y):
    mean_crps = np.mean(prob_pred.crps(y))
    return mean_crps

def pcs(pc_val, y):
    # crps of the climatological forecast
    pc_ref = np.mean(np.abs(np.tile(y, (len(y), 1)) - np.tile(y, (len(y), 1)).transpose())) / 2

    return (pc_ref - pc_val) / pc_ref

# threshold weighted
def tw_crps(self, obs, t):
    
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    def get_points(pred):
        return np.array(pred.points)
    def get_cdf(pred):
        return np.array(pred.ecdf)
    def modify_points(cdf):
        return np.hstack([cdf[0], np.diff(cdf)])

    def tw_crps0(y, p, w, x, t):
        x = np.maximum(x, t)
        y = np.maximum(y, t)
        return 2 * np.sum(w * ((y < x).astype(float) - p + 0.5 * w) * (x - y))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(modify_points, p))

    T = [t] * len(y)   
    return list(map(tw_crps0, y, p, w, x, T))

def tw_crps_small(self, obs, t):
    """
    Threshold-weighted CRPS focusing on the lower tail:
    apply min(., t) to both support points x and observations y, then integrate CRPS.
    """
    predictions = self.predictions
    y = np.array(obs)

    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    def get_points(pred):
        return np.array(pred.points)

    def get_cdf(pred):
        return np.array(pred.ecdf)

    def modify_points(cdf):
        # Convert CDF knots to bin-weights
        return np.hstack([cdf[0], np.diff(cdf)])

    def tw_crps0_small(y_i, p_i, w_i, x_i, t_i):
        # Lower-tail truncation: cap values from above at t (min)
        x_cap = np.minimum(x_i, t_i)
        y_cap = np.minimum(y_i, t_i)
        return 2.0 * np.sum(w_i * ((y_cap < x_cap).astype(float) - p_i + 0.5 * w_i) * (x_cap - y_cap))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(modify_points, p))
    T = [t] * len(y)
    return list(map(tw_crps0_small, y, p, w, x, T))

def tw_pc(prob_pred, y, t): 

    type(prob_pred).tw_crps = tw_crps # monkey-patch the tw_crps method into the prediction object

    tw_crps_scores = prob_pred.tw_crps(y, t)
    mean_tw_crps = np.mean(tw_crps_scores)
    return mean_tw_crps

def tw_pc_small(prob_pred, y, t):
    """Mean threshold-weighted CRPS for lower tail."""
    type(prob_pred).tw_crps_small = tw_crps_small
    tw_scores = prob_pred.tw_crps_small(y, t)
    return np.mean(tw_scores)

def tw_pcs(tw_pc_val, y, t):

    y_thresh = np.maximum(y, t)
    pc_ref = np.mean(np.abs(np.tile(y_thresh, (len(y_thresh), 1)) - np.tile(y_thresh, (len(y_thresh), 1)).transpose())) / 2

    # pc_model = tw_pc(prob_pred, y, t)

    pcs = (pc_ref - tw_pc_val) / pc_ref 
   
    return pcs


def tw_pcs_small(tw_pc_val_small, y, t):
    """
    Threshold-weighted skill score for lower tail using pairwise MAD
    on lower-capped observations as reference.
    """
    y_cap = np.minimum(y, t)
    pc_ref_small = np.mean(np.abs(np.tile(y_cap, (len(y_cap), 1)) - np.tile(y_cap, (len(y_cap), 1)).T)) / 2.0
    return (pc_ref_small - tw_pc_val_small) / pc_ref_small

# currently not explicitly needed
def pc_ref_formula_small(y, t):
    """Lower-tail TW reference (pairwise MAD on y capped from above at t)."""
    y_cap = np.minimum(y, t)
    return np.mean(np.abs(np.tile(y_cap, (len(y_cap), 1)) - np.tile(y_cap, (len(y_cap), 1)).T)) / 2.0

# currently not explicitly needed
def climatological_tw_pc_small(y, t):
    """
    Lower-tail TW PC using a pooled-ECDF (climatological) forecast via IDR
    with a constant predictor and the lower-tail TW integrand.
    """
    x_dummy = np.zeros_like(y)
    fit = idr(y, pd.DataFrame({"x": x_dummy}))
    prob_pred = fit.predict(pd.DataFrame({"x": x_dummy}), digits=12)
    type(prob_pred).tw_crps_small = tw_crps_small
    tw_scores = prob_pred.tw_crps_small(y, t)
    return np.mean(tw_scores)

# --- quantile weighted -------------------------------------------------------------------
def qw_crps0(y, w, x, q):
        c_cum = np.cumsum(w)
        c_cum_prev = np.hstack(([0], c_cum[:-1]))
        c_cum_star = np.maximum(c_cum, q)
        c_cum_prev_star = np.maximum(c_cum_prev, q)
        indicator = (x >= y).astype(float)
        terms = indicator * (c_cum_star - c_cum_prev_star) - 0.5 * (c_cum_star**2 - c_cum_prev_star**2)
        return 2 * np.sum(terms * (x - y))
    
def qw_crps0_small(y, w, x, q):
    """
    QW-CRPS lower-tail transform for a single observation y against discrete support x:
    replace F by min(F, q) to weight the lower tail (F <= q).
    """
    c_cum = np.cumsum(w)
    c_prev = np.hstack(([0.0], c_cum[:-1]))
    c_star = np.minimum(c_cum, q)
    c_prev_star = np.minimum(c_prev, q)
    indicator = (x >= y).astype(float)
    terms = indicator * (c_star - c_prev_star) - 0.5 * (c_star**2 - c_prev_star**2)
    return 2.0 * np.sum(terms * (x - y))

def qw_crps(self, obs, q=0.9):
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    def get_points(pred):
        return np.array(pred.points)
    def get_cdf(pred):
        return np.array(pred.ecdf)
    def get_weights(cdf):
        return np.hstack([cdf[0], np.diff(cdf)])

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(get_weights, p))
    Q = [q] * len(y)
    return list(map(qw_crps0, y, w, x, Q))

def qw_crps_small(self, obs, q=0.1):
    """
    Quantile-weighted CRPS focusing on the lower tail:
    replace F by min(F, q). Choose q in (0,1).
    """
    predictions = self.predictions
    y = np.array(obs)

    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    def get_points(pred):
        return np.array(pred.points)

    def get_cdf(pred):
        return np.array(pred.ecdf)

    def get_weights(cdf):
        return np.hstack([cdf[0], np.diff(cdf)])

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(get_weights, p))
    Q = [q] * len(y)
    return list(map(qw_crps0_small, y, w, x, Q))

def qw_pc(prob_pred, y, q=0.9):
    
    type(prob_pred).qw_crps = qw_crps
    qwcrps_scores = prob_pred.qw_crps(y, q=q)
    return np.mean(qwcrps_scores)

def qw_pc_small(prob_pred, y, q=0.1):
    """Mean lower-tail QW-CRPS."""
    type(prob_pred).qw_crps_small = qw_crps_small
    qw_scores = prob_pred.qw_crps_small(y, q=q)
    return np.mean(qw_scores)

def qw_pc0(y, q):
    y = np.sort(np.asarray(y))
    n = len(y)
    w = np.full(n, 1.0 / n)  # uniform weights

    pc_ref = sum(qw_crps0(y_i, w, y, q) for y_i in y) / n
    return pc_ref

def qw_pc0_small(y, q):
    """
    Lower-tail QW reference PC for a climatological forecast using the empirical support.
    """
    y_sorted = np.sort(np.asarray(y))
    n = len(y_sorted)
    w = np.full(n, 1.0 / n)
    return np.mean([qw_crps0_small(y_i, w, y_sorted, q) for y_i in y_sorted])

def climatological_qw_pc(y, q):
        """
        Compute PCRPS using a climatological forecast: fit IDR on y, predict same pooled ECDF for all y.
        (This avoids overfitting)
        """
        x_dummy = np.zeros_like(y)  # All predictors the same -> 1 pooled distribution
        fitted_idr = idr(y, pd.DataFrame({'x': x_dummy}))
        prob_pred = fitted_idr.predict(pd.DataFrame({'x': x_dummy}))

        type(prob_pred).qw_crps = qw_crps # monkey-patch 

        qw_crps_scores = prob_pred.qw_crps(y, q)
        mean_qw_crps = np.mean(qw_crps_scores)
        return mean_qw_crps
    
def climatological_qw_pc_small(y, q):
    """
    Lower-tail QW PC using a pooled-ECDF (climatological) forecast via IDR
    with the lower-tail QW integrand.
    """
    x_dummy = np.zeros_like(y)
    fit = idr(y, pd.DataFrame({"x": x_dummy}))
    prob_pred = fit.predict(pd.DataFrame({"x": x_dummy}), digits=12)
    type(prob_pred).qw_crps_small = qw_crps_small
    qw_scores = prob_pred.qw_crps_small(y, q=q)
    return np.mean(qw_scores)

def qw_pcs(qw_pc_val, y, q):

    # pc_model = qw_pc(prob_pred, y, q)
    pc_ref = qw_pc0(y, q)
    # pc_ref = climatological_qw_pc(y, q)
    # diff = pc_ref - pc_faster
    # print ("difference in pc_crps0 and pc_climatological: ", diff)

    pcs = (pc_ref - qw_pc_val) / pc_ref   
    return pcs


def qw_pcs_small(qw_pc_val_small, y, q):
    """
    Lower-tail QW skill score using the empirical lower-tail QW reference.
    """
    ref_small = qw_pc0_small(y, q)
    return (ref_small - qw_pc_val_small) / ref_small


# --- Gaussian CDF weighted TW-CRPS ------------------------------------------
# twCRPS with w(z)=Phi_{mu, sigma}(z) via chaining v(z)=(z-mu)*Phi + sigma^2*phi

import math
from typing import Callable

def _norm_pdf(z, mu, sigma):
    u = (z - mu) / sigma
    return (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * np.exp(-0.5 * u * u)

def _norm_cdf(z, mu, sigma):
    u = (z - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(u))

def gaussian_cdf_chaining(mu: float, sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    """Chaining v for w(z)=Phi_{mu,sigma}(z): v(z)=(z-mu)*Phi + sigma^2*phi."""
    if not np.isfinite(mu):
        raise ValueError("mu must be finite.")
    if not (np.isfinite(sigma) and sigma > 0):
        raise ValueError("sigma must be finite and > 0.")
    def v_func(z_in):
        z = np.asarray(z_in, dtype=float)
        Phi = np.vectorize(_norm_cdf)(z, mu, sigma)
        phi = np.vectorize(_norm_pdf)(z, mu, sigma)
        return (z - mu) * Phi + (sigma ** 2) * phi
    return v_func

def _twcrps_from_discrete(x_points: np.ndarray, x_weights: np.ndarray,
                          y_val: float, vfunc: Callable[[np.ndarray], np.ndarray]) -> float:
    """Discrete twCRPS = sum w_i|v(x_i)-v(y)| - 0.5*sum_{ij} w_i w_j |v(x_i)-v(x_j)|."""
    vx = vfunc(x_points)
    vy = float(vfunc(np.array([y_val]))[0])
    term1 = np.sum(x_weights * np.abs(vx - vy))
    diff = np.abs(vx[:, None] - vx[None, :])
    term2 = 0.5 * np.sum((x_weights[:, None] * x_weights[None, :]) * diff)
    return float(term1 - term2)

def tw_crps_gaussian_cdf(self, obs, mu: float, sigma: float):
    """
    TW-CRPS using Gaussian CDF weight (via chaining).
    Returns a list of twCRPS values (one per case).
    """
    predictions = self.predictions
    y = np.array(obs)
    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains NaN values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    vfunc = gaussian_cdf_chaining(mu=mu, sigma=sigma)

    def get_points(pred): return np.array(pred.points, dtype=float)
    def get_cdf(pred):    return np.array(pred.ecdf,  dtype=float)
    def get_masses(cdf):
        w = np.hstack([cdf[0], np.diff(cdf)])
        w[w < 0] = 0.0
        s = w.sum()
        return w / s if s > 0 else w

    xs  = list(map(get_points, predictions))
    cdf = list(map(get_cdf, predictions))
    ws  = list(map(get_masses, cdf))

    if y.size == 1:
        y_rep = [y.item()] * len(xs)
    else:
        y_rep = y.tolist()

    return [_twcrps_from_discrete(xi, wi, yi, vfunc) for xi, wi, yi in zip(xs, ws, y_rep)]

def tw_pc_gaussian_cdf(prob_pred, y, mu: float, sigma: float) -> float:
    """Mean Gaussian-CDF TW-CRPS."""
    type(prob_pred).tw_crps_gaussian_cdf = tw_crps_gaussian_cdf
    vals = prob_pred.tw_crps_gaussian_cdf(y, mu=mu, sigma=sigma)
    return float(np.mean(vals))

def tw_pcs_gaussian_cdf(tw_pc_val: float, y: np.ndarray, mu: float, sigma: float) -> float:
    """
    Skill score vs. climatology computed in v-space:
    twPC^(0) = (1/(2n^2)) * sum_{i,j} | v(y_i) - v(y_j) |.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be 1-D")
    v = gaussian_cdf_chaining(mu=mu, sigma=sigma)(y)
    pc_ref = 0.5 * np.mean(np.abs(v[:, None] - v[None, :]))
    return (pc_ref - tw_pc_val) / pc_ref

