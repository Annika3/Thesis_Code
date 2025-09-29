# CRPS variants visualization for thesis
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from PIL import Image
import os
def qw_tw_crps_visualization():
    # ----- Common setup -----
    x = np.linspace(-3, 3, 1000)           # x-grid
    F = norm.cdf(x)                        # predictive CDF F(x)
    y = 0.6                                # observation (step location)
    t = 0.8                                # threshold for twCRPS (must be > y for this illustration)
    q = 0.8                                # quantile cutoff for qwCRPS (horizontal line at level q)

    # Step function H(x - y) = 1{x >= y}
    H = (x >= y).astype(float)

    # Font sizes
    label_fs = 16     # axis labels
    title_fs = 16     # panel titles
    annot_fs = 14     # x=t and F(x)=q labels
    tick_fs  = 12     # tick labels 

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    # Panel 1
    ax = axes[0]
    ax.plot(x, F, linestyle='-', zorder=2)
    ax.plot(x, H, linestyle='-', color='black', zorder=3)
    ax.fill_between(x, F, H, alpha=0.25, zorder=1)
    ax.set_title("CRPS", fontsize=title_fs)
    ax.set_xlabel("z", fontsize=label_fs)
    ax.set_ylabel(r"$F(z)\ (=\alpha)$", fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)

    # Panel 2
    ax = axes[1]
    ax.plot(x, F, linestyle='-', zorder=2)
    ax.plot(x, H, linestyle='-', color='black', zorder=3)
    mask_tw = x >= t
    ax.fill_between(x[mask_tw], F[mask_tw], H[mask_tw], alpha=0.25, zorder=1)
    ax.axvline(t, linestyle='-', color='green', zorder=4)
    ax.text(t + 0.05, 0.5, r"$z=t$", color='green',
            rotation=0, rotation_mode='anchor',
            va='center', ha='left', fontsize=annot_fs)
    ax.set_title(r"twCRPS: $w(z)=\mathbf{1}\{z\geq t\}$", fontsize=title_fs)
    ax.set_xlabel("z", fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)

    # Panel 3
    ax = axes[2]
    ax.plot(x, F, linestyle='-', zorder=2)
    ax.plot(x, H, linestyle='-', color='black', zorder=3)
    mask_right = x >= y
    lower = np.maximum(F, q)
    upper = np.ones_like(x)
    ax.fill_between(x[mask_right], lower[mask_right], upper[mask_right], alpha=0.25, zorder=1)
    ax.axhline(q, linestyle='-', color='green', zorder=4)
    ax.text(x.min() + 0.1, q + 0.03, r"$F(z)=q$", color='green',
            va='bottom', ha='left', fontsize=annot_fs)
    ax.set_title(r"qwCRPS: $w(\alpha)=\mathbf{1}\{\alpha\geq q\}$", fontsize=title_fs)
    ax.set_xlabel("z", fontsize=label_fs)
    ax.tick_params(axis='both', labelsize=tick_fs)

    for ax in axes:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, "plots")
    combo_path = os.path.join(save_dir, "crps_twcrps_qwcrps_1x3.png")
    fig.savefig(combo_path, dpi=200)

    plt.show()


if __name__ == '__main__':
    qw_tw_crps_visualization()