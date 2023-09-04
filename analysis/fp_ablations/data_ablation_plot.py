""" data_ablation_plot.py

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
import matplotlib as mpl

from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-file",
        default="results/2022_09_08_canopus_data_ablation/fp_pred_summary.tsv",
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--metric", default="cos_sim")
    parser.add_argument("--png", default=False, action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    data_ablation_file = Path(args.ablation_file)
    metric = args.metric

    if save_name is None:
        output_dir = Path(data_ablation_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"data_ablation_plot_{metric}.{ext}"

    plt_df = pd.read_csv(data_ablation_file, sep="\t")
    title_map = lambda x: f"{x}%"
    keys = [20, 40, 60, 80, 100]

    y_name_map = dict(cos_sim="Cosine similarity", ll="Log likelihood")

    mist_color = "#6F84AE"
    positions = np.arange(len(keys))
    figsize = (0.8, 1.5)
    fig = plt.figure(figsize=(figsize), dpi=400)
    ax = fig.gca()
    means = []
    for ind, key in enumerate(keys):
        subset = plt_df[plt_df["split_frac"] == key]
        mean = np.mean(subset[metric])
        means.append(mean)
        sem = st.sem(subset[metric])
        sem_95 = st.norm.interval(alpha=0.95, loc=mean, scale=sem)
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
            mew=0.3,
            mfc="#1D2452",  # mist_color,
        )  # marker='s', #mfc='red',
        # mec='green', ms=20, mew=4)

    ax.plot(
        positions, means, linewidth=0.8, linestyle="--", color="#1D2452"
    )  # mist_color)
    ax.set_ylabel(y_name_map[metric])
    ax.set_xticks(positions)
    ax.set_xticklabels([title_map(i) for i in keys], rotation=60)
    # ax.set_ylim([0.5, 0.8])
    ax.set_xlabel("Dataset fraction")
    set_size(*figsize, ax)

    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
