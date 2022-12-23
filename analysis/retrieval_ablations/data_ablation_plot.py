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


def convert_to_lineplot(inds, max_num=1000):
    sample_nums = np.arange(0, max_num + 1)
    out = np.zeros_like(sample_nums).astype(float)
    for i in sample_nums:
        out[i] = (inds <= i).mean()
    return out


title_map = lambda x: f"{x}%"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-file",
        default="results/2022_09_13_canopus_data_retrieval_ablation/ind_found_collective.p",
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--png", default=False, action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    data_ablation_file = Path(args.ablation_file)
    out = pickle.load(open(data_ablation_file, "rb"))

    if save_name is None:
        output_dir = Path(data_ablation_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"data_ablation_plot_retrieval.{ext}"

    save_name_to_inds = {}
    for i in out:
        keys = i.keys()
        save_dir = Path(i["file"])
        save_int = int(save_dir.parent.name)
        print(save_int)
        if save_int in save_name_to_inds:
            save_name_to_inds[save_int].extend(i["ind_found"])
        else:
            save_name_to_inds[save_int] = i["ind_found"].tolist()

    for i, j in save_name_to_inds.items():
        save_name_to_inds[i] = np.array(j)

    all_outs = []
    for i, j in save_name_to_inds.items():

        ret_accs = convert_to_lineplot(j, max_num=1000)
        ranks = np.arange(ret_accs.shape[0])
        temp = [dict(k=k, acc=l, name=i) for k, l in zip(ranks, ret_accs)]
        all_outs.extend(temp)
    df = pd.DataFrame(all_outs)
    all_names = np.unique(df["name"].values)
    name_sort = sorted(all_names)

    out_table = df.pivot_table(index="name", columns="k", values="acc")[
        [1, 5, 10, 20, 50, 100, 200]
    ]
    print(out_table)

    k_vals = [1, 20, 50, 100]
    offset = 1
    pal = sns.light_palette("#6F84AE", n_colors=len(k_vals) + offset)[
        offset:
    ]  # , as_cmap=True
    num_to_color = dict(zip(k_vals, pal[:]))

    figsize = (1.8, 1.5)
    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.gca()

    for k in k_vals:
        subset = df[df["k"] == k]
        acc_values = subset["acc"].values
        frac_values = subset["name"].values
        color = num_to_color[k]
        argsort = np.argsort(frac_values)
        acc_values = acc_values[argsort]
        frac_values = frac_values[argsort]
        ax.plot(
            frac_values,
            acc_values,
            color=color,
            zorder=0,
            linestyle="--",
            linewidth=0.8,
        )
        ax.scatter(
            frac_values, acc_values, color=color, edgecolor="black", linewidth=0.5
        )

    ax.set_xticks(frac_values)
    ax.set_xticklabels([title_map(i) for i in frac_values], rotation=60)
    # ax.set_ylim(0.3, 0.4)
    ax.set_ylabel("Accuracy")

    # # Build legend
    patch_legend_handles = [
        Patch(
            facecolor=num_to_color[k],
            label=f"Top {k}",
            edgecolor="black",
            linewidth=0.3,
        )
        for k in sorted(k_vals)
    ]
    legend_handles = [*patch_legend_handles]
    legend = ax.legend(
        handles=legend_handles,
        loc=(1.0, 0.5),  # (0.1,0.0),
        frameon=False,
        facecolor="none",
        fancybox=False,
    )  # bbox_to_anchor = "center")
    ax.set_ylim([0, 1])
    ax.set_xlabel("Dataset fraction")

    set_size(*figsize, ax)
    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
