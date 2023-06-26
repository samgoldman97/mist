""" model_ablation_plot.py

Create retrieval classification barplot

"""
import argparse
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
import scipy.stats as st

from mist import utils

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

import seaborn as sns

from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-file",
        default="results/2022_09_08_canopus_ablations_lr_fixed/fp_pred_summary.tsv",
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--metric", default="cos_sim")
    parser.add_argument("--png", default=False, action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    model_ablation_file = Path(args.ablation_file)
    metric = args.metric

    if save_name is None:
        output_dir = Path(model_ablation_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"model_ablation_plot_{metric}.{ext}"

    plt_df = pd.read_csv(model_ablation_file, sep="\t")

    keys = pd.unique(plt_df["method"])
    title_map = {
        "ffn": "FFN",
        "no_aug": r"MIST $-$ simulated",
        "no_pairwise": r"MIST $-$ pairwise",
        "no_magma": r"MIST $-$ MAGMa",
        "no_growing": r"MIST $-$ unfolding",
        "full_model": r"MIST",
    }
    keys = ["ffn", "no_aug", "no_pairwise", "no_magma", "no_growing", "full_model"]

    method_to_mean = {}
    for k in keys:
        subset = plt_df[plt_df["method"] == k]
        # Sort by cos sim
        mean = np.mean(subset["cos_sim"])
        method_to_mean[k] = mean
    method_to_mean["full_model"] = 10000

    method_order = sorted(keys, key=lambda x: method_to_mean[x], reverse=True)

    pal = sns.light_palette("#6F84AE", n_colors=len(keys) + 1)  # , as_cmap=True

    y_name_map = dict(cos_sim="Cosine similarity", ll="Log likelihood")

    positions = np.arange(len(keys))
    figsize = (0.8, 1.5)
    fig = plt.figure(figsize=(figsize), dpi=400)
    ax = fig.gca()
    plotted_bars = []
    for ind, key in enumerate(method_order):
        subset = plt_df[plt_df["method"] == key]
        mean = np.mean(subset[metric])
        color = pal[ind]
        sem = st.sem(subset[metric])
        sem_95 = st.norm.interval(alpha=0.95, loc=mean, scale=sem)
        plotted_bars.append(mean)

        #     ax.bar(x = positions[ind], height=mean, color=color,
        #           edgecolor="black", linewidth=0.3)
        ax.errorbar(
            x=positions[ind],
            y=mean,
            yerr=np.abs(mean - np.array(sem_95)[:, None]),
            marker="o",
            ms=4,
            mec="black",
            color="black",
            linewidth=0.8,
            capsize=1.2,
            capthick=0.8,
            mew=0.4,
            mfc="#1D2452",  # color
        )  # marker='s', #mfc='red',
        # mec='green', ms=20, mew=4)

    ax.set_ylabel(y_name_map[metric])
    ax.set_xticks(positions)
    ax.set_xticklabels([title_map.get(i) for i in method_order], rotation=90)

    # lb, ub = np.min(plotted_bars) - np.std(plotted_bars), np.max(plotted_bars) + np.std(plotted_bars)
    # ax.set_ylim([lb,ub])
    set_size(*figsize, ax)
    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
