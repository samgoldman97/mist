""" fp_boxplot.py

Create boxplot of fingerprints

Example:

python3 analysis/fp_preds/fp_boxplot.py --fp-pred-files data/paired_spectra/csi2022/prev_results/spectra_encoding_csi2022_Fold_0.p results/2022_08_22_mist_best_aug_lr/fp_preds_csi2022.p results/2022_08_22_ffn_binned/fp_preds_csi2022.p --model-names CSI:FingerID MIST FFN --save-name /Users/samgoldman/Desktop/fp_boxplot.png

"""
import argparse
from pathlib import Path
import pickle
from functools import partial
import pandas as pd
from itertools import product

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter

from mist import utils
from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp-pred-files", nargs="+")
    parser.add_argument("--model-names", nargs="+")
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--png", default=False, action="store_true")
    return parser.parse_args()


def ll(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    log = partial(utils.clamped_log_np, _min=-5)
    ll = targ * log(pred) + (1 - targ) * log(1 - pred)
    return ll


def cos_sim(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    pred

    sim = cosine_similarity(pred, targ)
    sim = np.diag(sim)
    return sim[:, None]


def tani(pred, targ):
    """tani.

    Args:
        pred:
        targ:
    """
    pred = np.copy(pred)
    above_thresh = pred >= 0.5
    pred[above_thresh] = 1.0
    pred[~above_thresh] = 0.0

    pred, targ = pred.astype(bool), targ.astype(bool)
    denom = np.logical_or(pred, targ).sum(-1)
    num = np.logical_and(pred, targ).sum(-1)
    res = num / denom
    return res[:, None]


def get_metric(metric):
    """get_metric.

    Args:
        metric:
    """
    """get_metric.

    Args:
        metric:
    """
    return {
        "LL": ll,
        "Cosine": cos_sim,
        "Tani": tani,
    }[metric]


def set_size(w, h, ax=None):
    """w, h: width, height in inches"""
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    pred_files = args.fp_pred_files
    pred_names = args.model_names

    assert len(pred_names) == len(pred_files)

    if save_name is None:
        pred_file = pred_files[0]
        output_dir = Path(pred_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)

        ext = "png" if png else "pdf"
        # save_name = f"{model_name}_CSI_{metric}_{pool_method}.{ext}"
        save_name = f"csi_ffn_boxplot.{ext}"
        save_name = output_dir / save_name

    metric_val_dict = {
        "Tanimoto": ("Tani", "spectra"),
        "Cosine": ("Cosine", "spectra"),
        "NLL (spec)": ("LL", "spectra"),
        "NLL (bit)": ("LL", "bit"),
    }

    metric_names = list(metric_val_dict.keys())
    metric_vals = [metric_val_dict[i] for i in metric_names]

    metric_vals = [
        ("Tani", "spectra"),
        ("Cosine", "spectra"),
        ("LL", "spectra"),
        ("LL", "bit"),
    ]
    metric_names = [
        "Tanimoto",
        "Cosine",
        "Log likelihood\n(spectra)",
        "Log likelihood\n(bits)",
    ]

    # Get preds and sort
    preds = []
    for pred_file in pred_files:
        fp_preds = pickle.load(open(pred_file, "rb"))
        p, t = (np.array(fp_preds["preds"]), np.array(fp_preds["targs"]))
        preds.append((p, t))

    # Next compute matrix predictions
    # Start with likelihoods
    out_list = []
    res_pairs = preds
    res_names = pred_names
    for (preds, targs), method_name in zip(res_pairs, res_names):
        for (metric, pool_method), metric_name in zip(metric_vals, metric_names):

            val_fn = get_metric(metric)
            res = val_fn(preds, targs)

            if pool_method == "spectra":
                res = res.mean(1)
            elif pool_method == "bit":
                res = res.mean(0)

            out_dicts = [
                {"Metric": metric_name, "Val": i, "Name": method_name}
                for i in res.flatten()
            ]
            out_list.extend(out_dicts)

    # Seaborn plot
    plt_df = pd.DataFrame(out_list)

    # Get statistics
    std_df = plt_df.pivot_table(
        index="Name", values="Val", columns="Metric", aggfunc="std"
    )
    count_df = np.sqrt(
        plt_df.pivot_table(
            index="Name", values="Val", columns="Metric", aggfunc="count"
        )
    )
    se_df = std_df / count_df
    mean_df = plt_df.pivot_table(
        index="Name", values="Val", columns="Metric", aggfunc="mean"
    )
    print("SE df:", se_df)
    print("Mean df:", mean_df)

    ax_sizes = (1.52, 1.76)
    if False:
        colors = ["#88B7BC", "#EFC6B7", "#FAEED1"]
        colors = ["#6F84AE", "#88B7BC", "#EFC6B7"]
        fig = plt.figure(figsize=(2 * len(metric_vals), 10))
        ax = fig.gca()
        sns.boxplot(
            x="Metric",
            y="Val",
            hue="Name",
            data=plt_df,
            ax=ax,
            hue_order=pred_names,
            order=metric_names,
            palette=colors,
        )
        ax.set_ylabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
        legend = ax.legend(
            loc="lower left", frameon=False, facecolor="none", fancybox=False
        )
        legend.get_frame().set_facecolor("none")
        fig.savefig(save_name, bbox_inches="tight")
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=ax_sizes)

        positions = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]
        rotation = 30
        x1_ticks = [2, 6]
        x1_labels = metric_names[:2]
        x2_ticks = [10, 14]
        x2_labels = metric_names[2:]
        colors = ["#6F84AE", "#88B7BC", "#EFC6B7"]
        colors = ["#EFC6B7", "#6F84AE", "#88B7BC"]
        method_to_color = dict(zip(pred_names, colors))

        for ind, (metric, method) in enumerate(
            (list(product(metric_names, pred_names)))
        ):
            df_mask = np.logical_and(
                plt_df["Metric"] == metric, plt_df["Name"] == method
            )
            values = plt_df[df_mask]["Val"].values
            if ind // 3 >= 2:
                ax = ax2
            else:
                ax = ax1
            bplot = ax.boxplot(
                values,
                positions=[positions[ind]],
                widths=0.8,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(linewidth=0.3),
                medianprops=dict(linewidth=0.3),
                whiskerprops=dict(linewidth=0.3),
                capprops=dict(linewidth=0.3),
            )

            color = method_to_color[method]
            for i in bplot["boxes"]:
                i.set_facecolor(color)
            for median in bplot["medians"]:
                median.set_color("black")

        ax1.set_xticks(x1_ticks)
        ax1.set_xticklabels(x1_labels, rotation=rotation)
        ax2.set_xticks(x2_ticks)
        ax2.set_xticklabels(x2_labels, rotation=rotation)

        legend_handles = [
            Patch(facecolor=method_to_color[method], label=method)
            for method in pred_names
        ]
        legend = ax2.legend(
            handles=legend_handles,
            loc=(0, 0),
            frameon=False,
            facecolor="none",
            fancybox=False,
            handletextpad=0.2,
        )

        legend.get_frame().set_facecolor("none")
        ax2.yaxis.tick_right()
        ax2.set_ylim([-0.25, 0.01])

        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        fig.text(0.5, -0.07, "Metric", ha="center", color="black")
        plt.subplots_adjust(
            left=0.1, bottom=0.16, right=0.9, top=0.9, wspace=0.04, hspace=0.4
        )
        set_size(*ax_sizes, ax1)
        fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
