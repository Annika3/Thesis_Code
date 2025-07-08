import numpy as np
import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.isotonic import isotonic_regression

from tqdm import tqdm
import timeit
import os

from isodisreg import idr       # to compute isotonic regression fit
import urocc                    # to compute CPA

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))          # one level up
save_plots = os.path.join(BASE_DIR, 'simulation_study', 'plots')

# define possible error measures ----------------------------------------------------

def rmse(pred, y):
    # root mean squared error
    return np.sqrt(np.mean((pred - y)**2)) 

def acc(pred, y):
    # anomaly correlation coefficient corresponds simply to the Pearson correlation
    # numpy just evaluates correlation matrix, i.e., correlation coefficient is at [0,1] or at [1,0]
    return np.corrcoef(pred, y)[0, 1]

def mae(pred, y):
    # mean absolute loss
    return np.mean(np.abs(pred - y))

def ql90(pred, y):
    # quantile loss at level 0.9
    return np.mean(((pred > y) - 0.9) * (pred - y))

def cpa(pred, y):
    # coefficient of predictive ability : average AUC values for all possible binarized problems
    return urocc.cpa(y, pred)

def auc(pred, y):
    # calculate AUC
    return metrics.roc_auc_score(y, pred)

def auc_plus(pred, y):
    # calculate AUC after replacing ROC curve by its concave hull
    pair_array = np.array(list(zip(pred, -1 * y)), dtype=[('x', float), ('-y', float)])
    # sort x increasing, and in case of ties use y to determine decreasing order
    order = np.argsort(pair_array, order=('x', '-y'))
    recalibrate = isotonic_regression(y[order])
    return metrics.roc_auc_score(y[order], recalibrate)

def pc(pred, y):
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    # "predict" the estimated cdfs again, so that we can use the crps function
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))
    mean_crps = np.mean(prob_pred.crps(y))
    return mean_crps

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

    def qw_crps0(y, p, w, x, q):
        c_cum = np.cumsum(w)
        c_cum_prev = np.hstack(([0], c_cum[:-1]))
        c_cum_star = np.maximum(c_cum, q)
        c_cum_prev_star = np.maximum(c_cum_prev, q)
        indicator = (x >= y).astype(float)
        terms = indicator * (c_cum_star - c_cum_prev_star) - 0.5 * (c_cum_star**2 - c_cum_prev_star**2)
        return 2 * np.sum(terms * (x - y))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(get_weights, p))
    Q = [q] * len(y)
    return list(map(qw_crps0, y, p, w, x, Q))

def tw_pc(pred, y, t): 
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))

    type(prob_pred).tw_crps = tw_crps # monkey-patch the tw_crps method into the prediction object

    tw_crps_scores = prob_pred.tw_crps(y, t)
    mean_tw_crps = np.mean(tw_crps_scores)
    return mean_tw_crps

def qw_pc(pred, y, q=0.9):
    
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))
    type(prob_pred).qw_crps = qw_crps
    qwcrps_scores = prob_pred.qw_crps(y, q=q)
    return np.mean(qwcrps_scores)


def my_crps(x, cum_weights, y):
    weights = cum_weights - np.hstack((np.zeros(((y.size, 1))), cum_weights[:, :-1]))
    # the formula is simply extracted from idr predict crps function
    return 2 * np.sum(weights * (np.array((y < x)) - cum_weights + 0.5 * weights) * np.array(x - y), axis=1)

def pc_time(pred, y):
    time1 = timeit.default_timer()
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    time2 = timeit.default_timer()
    mean_crps = np.mean(my_crps(np.reshape(fitted_idr.thresholds, (1, -1)), # reshape to make it braodcast-able
                                fitted_idr.ecdf,
                                np.concatenate(fitted_idr.y.values).reshape((-1, 1))))
    time3 = timeit.default_timer()
    print(mean_crps)
    return np.array([time2 - time1, time3 - time2])

def pcs(pred, y):
    # crps of the climatological forecast
    pc_ref = np.mean(np.abs(np.tile(y, (len(y), 1)) - np.tile(y, (len(y), 1)).transpose())) / 2

    return (pc_ref - pc(pred, y)) / pc_ref

def tw_pcs(pred, y, t):

    y_thresh = np.maximum(y, t)
    pc_ref = np.mean(np.abs(np.tile(y_thresh, (len(y_thresh), 1)) - np.tile(y_thresh, (len(y_thresh), 1)).transpose())) / 2

    pc_model = tw_pc(pred, y, t)

    pcs = (pc_ref - pc_model) / pc_ref
    
    return pcs

def qw_pcs(pred, y, q):

    pc_model = qw_pc(pred, y, q)
    
    def climatological_qw_pc(y, q):
        """
        Compute PCRPS using a climatological forecast: fit IDR on y, predict same pooled ECDF for all y.
        This avoids overfitting and returns the proper CRPS reference value.
        """
        x_dummy = np.zeros_like(y)  # All predictors the same -> 1 pooled distribution
        fitted_idr = idr(y, pd.DataFrame({'x': x_dummy}))
        prob_pred = fitted_idr.predict(pd.DataFrame({'x': x_dummy}))

        type(prob_pred).qw_crps = qw_crps # monkey-patch 

        qw_crps_scores = prob_pred.tw_crps(y, q)
        mean_qw_crps = np.mean(qw_crps_scores)
        return mean_qw_crps
    pc_ref = climatological_qw_pc(y, q)
    pcs = (pc_ref - pc_model) / pc_ref    
    return pcs




# simulation routines ----------------------------------------------------------------

def get_data(n = 1000, seed = 1):
    np.random.seed(seed)
    w = np.random.uniform(low=0.0, high=10.0, size=n)

    my_shape = np.sqrt(w)
    my_scale = np.minimum(np.maximum(w, 1), 6)
    y = np.random.gamma(shape=my_shape, scale=my_scale, size=n)

    return pd.DataFrame({'y': y,
                         'f1': w,
                         'f2': my_shape * my_scale,
                         'f3': gamma.ppf(0.5, my_shape, scale=my_scale),
                         'f4': gamma.ppf(0.9, my_shape, scale=my_scale)})


def run_simulation_example_1(n=1000, square_y=False):
    pred_data = get_data(n=n)
    plot_data(pred_data)

    y = pred_data['y']
    add_name = ''
    if square_y:
        y = pred_data['y']**2
        add_name = '_squared'

    loss_fcts = {'RMSE': rmse, 'MAE': mae, 'QL90': ql90, 'PC': pc, 'ACC': acc, 'CPA': cpa, 'PCS': pcs, 'tw_PC': lambda pred, y: tw_pc(pred, y, t=24), 'tw_PCS': lambda pred, y: tw_pcs(pred, y, t=24), 'qw_PC': lambda pred, y: qw_pc(pred, y, q=0.9), 'qw_PCS': lambda pred, y: qw_pcs(pred, y, q=0.9)}

    fcsts = pred_data.columns[pred_data.columns != 'y']
    loss_vals = np.zeros((len(fcsts), len(loss_fcts)))

    for i in range(len(fcsts)):
        for j, loss in enumerate(loss_fcts.values()):
            loss_vals[i, j] = loss(pred_data[fcsts[i]], y)

    res = pd.DataFrame(loss_vals, columns=list(loss_fcts.keys()))
    res.insert(0, 'Model', fcsts)
    res.to_csv(os.path.join(save_plots, 'loss_values' + add_name + '.csv'))
    print(res)


def run_simulation_example_2(thresh_list=[10], add_name=''):
    df_all = pd.DataFrame()

    pred_data = get_data(n=1000)
    loss_fcts = {'PC': pc, 'PCS': pcs, 'AUC': auc, 'AUC+': auc_plus}
    fcsts = pred_data.columns[pred_data.columns != 'y']
    loss_vals = np.zeros((len(fcsts), len(loss_fcts)))

    for t in tqdm(thresh_list):
        for i in range(len(fcsts)):
            for j, loss in enumerate(loss_fcts.values()):
                loss_vals[i, j] = loss(pred_data[fcsts[i]], 1 * (pred_data['y'] >= t))

        res = pd.DataFrame(loss_vals, columns=list(loss_fcts.keys()))
        res.insert(0, 'Threshold', np.repeat(t, res.shape[0]))
        res.insert(0, 'Model', fcsts)

        df_all = pd.concat((df_all, res))
    print(df_all)
    df_all.to_csv(os.path.join(save_plots, 'loss_values_bin' + add_name + '.csv'))



# print and plot  ----------------------------------------------------------------

def print_results():
   #  column_order = ['RMSE', 'MAE', 'QL90', 'PC', 'ACC', 'CPA', 'PCS', 'tw_PC']
    column_order = ['RMSE', 'MAE', 'QL90', 'PC', 'ACC', 'CPA', 'PCS']
    t = pd.read_csv(os.path.join(save_plots, 'loss_values.csv'))
    # print(t.loc[:, column_order].round({'RMSE': 2, 'MAE': 2, 'QL90': 2, 'PC': 2, 'ACC': 3, 'CPA': 3, 'PCS': 3, 'tw_PC': 3}))
    print(t.loc[:, column_order].round({'RMSE': 2, 'MAE': 2, 'QL90': 2, 'PC': 2, 'ACC': 3, 'CPA': 3, 'PCS': 3}))
    t = pd.read_csv(os.path.join(save_plots, 'loss_values_squared.csv'))
    # print(t.loc[:, column_order].round({'RMSE': 0, 'MAE': 0, 'QL90': 0, 'PC': 0, 'ACC': 3, 'CPA': 3, 'PCS': 3, 'tw_PC': 3}))
    print(t.loc[:, column_order].round({'RMSE': 0, 'MAE': 0, 'QL90': 0, 'PC': 0, 'ACC': 3, 'CPA': 3, 'PCS': 3}))

def plot_data(df):
    plt.figure()
    plt.title('Pairwise Scatter Plot')
    pd.plotting.scatter_matrix(df, figsize=(10, 10))
    plt.savefig(os.path.join(save_plots, 'pairwise_plot.png'))
    plt.close()

    df_sorted = df.sort_values(by='f1')
    plt.figure()
    plt.scatter(df_sorted['f1'], df_sorted['y'], label='Y', c='black', s=5)
    plt.plot(df_sorted['f1'], df_sorted['f2'], label='Cond. mean')
    plt.plot(df_sorted['f1'], df_sorted['f3'], label='Cond. median')
    plt.plot(df_sorted['f1'], df_sorted['f4'], label='Cond. 90%-quantile')
    plt.legend()
    plt.savefig(os.path.join(save_plots, 'scatter_plot.png'))
    plt.close()


def plot_thresh_graph():
    df = pd.read_csv(os.path.join(save_plots, 'loss_values_bin_graph.csv'), index_col=0)
    # as all forecaster have same statistic filter one of them
    df = df.loc[df['Model'] == 'f1']
    stat_list = ['PC', 'PCS', 'AUC']    # take 'AUC+' out
    df_long = pd.melt(df, id_vars='Threshold', value_vars=stat_list, var_name='Stat')

    sns.set_theme(style='whitegrid')
    g = sns.lineplot(data=df_long, x='Threshold', y='value', hue='Stat', palette='husl')
    g.figure.set_size_inches(6.5, 4.5)
    g.set(xlabel='Threshold c', ylabel='', title='')
    plt.legend(title='')
    g.get_figure().savefig(os.path.join(save_plots, 'stat_by_threshold.png'))


def analyze_runtime():
    n_vals = 10**np.arange(2,5)

    time_measures = np.zeros((4 * len(n_vals), 2))

    for s, n in enumerate(n_vals):
        pred_data = get_data(n=n)
        fcsts = pred_data.columns[pred_data.columns != 'y']
        for i in range(len(fcsts)):
            time_measures[s * 4 + i] = pc_time(pred_data[fcsts[i]], pred_data['y'])

    res = pd.DataFrame(time_measures, columns=['IDR Fit', 'CRPS'])
    res.insert(0, 'n', np.repeat(n_vals, len(fcsts)))
    res.insert(0, 'Model', np.tile(fcsts, len(n_vals)))
    res.to_csv(os.path.join(save_plots, 'runtime.csv'))
    print(res)


def plot_runtime():
    df = pd.read_csv(os.path.join(save_plots, 'runtime.csv'), index_col=0)
    df_plot = df.groupby(['n', 'Model'])[['IDR Fit', 'CRPS']].mean().reset_index()
    df_long = pd.melt(df_plot, id_vars=['n', 'Model'], value_vars=['IDR Fit', 'CRPS'], var_name='Routine')

    sns.set_theme(style='whitegrid')
    g = sns.lineplot(data=df_long, x='n', y='value', hue='Routine', style='Model', palette='husl')
    g.figure.set_size_inches(6.5, 4.5)
    g.set(xlabel='Input size [n]', ylabel='Runtime [s]', title='')
    g.set(xscale='log', yscale='log')
    plt.legend(title='')
    plt.tight_layout()
    g.get_figure().savefig(os.path.join(save_plots, 'runtime.png'))


def test_of_new_functions():
    # Generate test data
    pred_data = get_data(n=1000, seed=1)
    y = pred_data['y']
    fcst = pred_data['f1']

    # threshold < all data
    t_test = min(np.min(y), np.min(fcst)) - 1.0

    # Compute regular & thresholded CRPS
    pc_val = pc(fcst, y)
    tw_pc_val = tw_pc(fcst, y, t_test)

    print("PC (CRPS):", pc_val)
    print("tw_PC at threshold=min-1:", tw_pc_val)
    print("Absolute difference:", abs(pc_val - tw_pc_val))

    # Compute quantile-weighted CRPS
    lowest_q = 0.0000001
    qw_pc_val = qw_pc(fcst, y, q=lowest_q)

    print(f"qw_PC (quantiles > {lowest_q}):", qw_pc_val)
    print(f"Absolute difference (PC - qw_PC): {abs(pc_val - qw_pc_val)}")

   # PCS: pc_ref manual vs. pc(y, y)
    pc_ref_manual = np.mean(np.abs(np.tile(y, (len(y), 1)) - np.tile(y, (len(y), 1)).transpose())) / 2
    def climatological_pc(y):
        """
        Compute PCRPS using a climatological forecast: fit IDR on y, predict same pooled ECDF for all y.
        This avoids overfitting and returns the proper CRPS reference value.
        """
        x_dummy = np.zeros_like(y)  # All predictors the same → 1 pooled distribution
        fitted_idr = idr(y, pd.DataFrame({'x': x_dummy}))
        prob_pred = fitted_idr.predict(pd.DataFrame({'x': x_dummy}))

        return np.mean(prob_pred.crps(y))
    pc_ref_func = climatological_pc(y)

    print("\nVergleich pc_ref Varianten:")
    print("Result manual pc_ref: ", pc_ref_manual)
    print("Result using sort of pc(y): ", pc_ref_func)
    print("Difference: ", np.abs(pc_ref_manual - pc_ref_func))


if __name__ == '__main__':
    if not os.path.exists(save_plots):
        print(f'Please create directory {save_plots}')
    else:
         run_simulation_example_1(n=1000, square_y=False)
         test_of_new_functions()
        # run_simulation_example_1(n=1000, square_y=True)
        # run_simulation_example_2(thresh_list=np.linspace(1, 40, 40), add_name='_graph')
        # run_simulation_example_2(thresh_list=[30], add_name='_graph')
        # plot_thresh_graph()
        # print_results()
        # analyze_runtime()
        # plot_runtime()

# archive ----------------------------------------------------------------
def tw_pc_masked(pred, y, t): 
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))

    type(prob_pred).tw_crps_masked = tw_crps_masked # monkey-patch the tw_crps_masked method into the prediction object

    tw_crps_scores = prob_pred.tw_crps_masked(y, t)
    mean_tw_crps = np.mean(tw_crps_scores)
    return mean_tw_crps

def tw_crps_masked(self, obs, t):
    
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
        mask = (x >= t).astype(float)
        w_masked = w * mask
        return 2 * np.sum(w_masked * ((y < x).astype(float) - p + 0.5 * w_masked) * (x - y))

    x = list(map(get_points, predictions))
    p = list(map(get_cdf, predictions))
    w = list(map(modify_points, p))

    T = [t] * len(y)   
    return list(map(tw_crps0, y, p, w, x, T))


def pc0(y):
    n = len(y)
    y_sorted = np.sort(y)
    ranks = np.arange(1, n + 1)          
    weights = 2 * ranks - n - 1  

    return np.sum(weights * y_sorted) / (n**2)

def qw_pc_approx(pred, y, q=0.9):
    
    fitted_idr = idr(y, pd.DataFrame({"x": pred}, columns=["x"]))
    prob_pred = fitted_idr.predict(pd.DataFrame({"x": pred}, columns=["x"]))
    type(prob_pred).qw_crps_approx = qw_crps_approx 
    qwcrps_scores = prob_pred.qw_crps_approx(y, q=q)
    return np.mean(qwcrps_scores)


def qw_crps_approx_paper (self, obs, q=0.9):

    predictions = self.predictions
    y = np.array(obs)
    J = 1999

    if y.ndim > 1:
        raise ValueError("obs must be a 1-D array")
    if np.isnan(np.sum(y)):
        raise ValueError("obs contains nan values")
    if y.size != 1 and len(y) != len(predictions):
        raise ValueError("obs must have length 1 or the same length as predictions")

    alphas = np.arange(1, J) / J  # j/J, j=1,...,J-1
    weights = (alphas > q).astype(float)
    qf = self.qpred(alphas)  # shape (n_samples, J-1)
    def qw_crps_single(y_i, qf_i):
        pinball = 2 * ((y_i < qf_i).astype(float) - alphas) * (qf_i - y_i)
        return np.sum(weights * pinball) / (J - 1)
    
    return [qw_crps_single(y[i], qf[i, :]) for i in range(len(y))]

    
    

def qw_crps_approx(self, obs, q=0.9):
    
    q_levels = np.linspace(0.005, 0.995, 199)
    y = np.array(obs)
    qf = self.qpred(q_levels)
    weights = (q_levels > q).astype(float)
    d_alpha = q_levels[1] - q_levels[0] 

    def qw_crps_approx_single(y_i, qf_i):
        indicator = (qf_i >= y_i).astype(float)
        return 2 * np.sum(weights * (indicator - q_levels) * (qf_i - y_i) * d_alpha)

    return [qw_crps_approx_single(y[i], qf[i, :]) for i in range(len(y))]

