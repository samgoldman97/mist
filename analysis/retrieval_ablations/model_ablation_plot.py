""" model_ablation_plot.py

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
    inds = [i if i is not None else 99999999999 for i in inds]
    for i in sample_nums:
        out[i] = (inds <= i).mean()
    return out


title_map = lambda x: f"{x}%"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-file",
        default="results/2022_09_13_canopus_model_retrieval_ablation/ind_found_collective.p",
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--png", default=False, action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    model_ablation_file = Path(args.ablation_file)
    out = pickle.load(open(model_ablation_file, "rb"))

    if save_name is None:
        output_dir = Path(model_ablation_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"model_ablation_plot_retrieval.{ext}"

    save_name_to_inds = {}
    for i in out:
        keys = i.keys()
        save_dir = Path(i["file"])
        saved_file = save_dir.parent.name
        print(saved_file)
        if saved_file in save_name_to_inds:
            save_name_to_inds[saved_file].extend(i["ind_found"])
        else:
            save_name_to_inds[saved_file] = i["ind_found"].tolist()

    for i, j in save_name_to_inds.items():
        save_name_to_inds[i] = np.array(j)

    save_name_to_retrieved = {}
    for i, j in save_name_to_inds.items():
        save_name_to_retrieved[i] = convert_to_lineplot(j)

    all_names = np.unique(list(save_name_to_inds.keys())).tolist()
    name_sort = sorted(
        all_names, key=lambda x: save_name_to_retrieved[x][5], reverse=True
    )
    names_map = {
        "retrieval-ffn-fp": "FFN fingerprint",
        "ffn": "FFN contrastive",
        "mist-contrastive": "MIST contrastive",
        "mist-no-pretrain": "MIST contrastive \n- pretrain",
        "retrieval-mist-fp": "MIST fingerprint",
        "mist-contrast-fp": "MIST contrastive \n+ fingerprint",
    }

    new_df = []
    for old_name in name_sort:
        new_name = names_map[old_name]
        cur_dict = save_name_to_retrieved[old_name]
        for k, val in enumerate(cur_dict):
            new_df.append({"name": new_name, "k": k, "acc": val})
    new_df = pd.DataFrame(new_df)
    out_table = new_df.pivot_table(index="name", columns="k", values="acc")[
        [1, 5, 10, 20, 50, 100, 200]
    ]
    out_table = out_table.sort_values(by=1)
    print(out_table.to_string())

    bars = [1, 20, 50, 100]
    offset = 1
    pal = sns.light_palette("#6F84AE", n_colors=len(bars) + offset)[
        offset:
    ]  # , as_cmap=True
    bar_to_color = dict(zip(bars, pal[:]))

    figsize = (1.8, 1.5)
    fig = plt.figure(figsize=figsize, dpi=400)
    ax = fig.gca()

    x_pos = np.arange(len(name_sort) * len(bars) + len(name_sort))
    for ind, j in enumerate(name_sort):

        pos_base = x_pos[ind] * len(bars)
        for bar_ind, bar in enumerate(bars):
            x = pos_base + bar_ind + ind
            retrieved = save_name_to_retrieved[j][bar]
            ax.bar(
                x,
                height=retrieved,
                color=bar_to_color[bar],
                width=1.0,
                edgecolor="black",
                linewidth=0.3,
            )

    x_pos = np.arange(len(name_sort)) * (len(bars) + 1) + (len(bars) / 2 - 0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([names_map.get(i) for i in name_sort], rotation=90)
    # ax.set_ylim(0.3, 0.4)
    ax.set_ylabel("Accuracy")

    # Build legend
    patch_legend_handles = [
        Patch(
            facecolor=bar_to_color[bar],
            label=f"Top {bar}",
            edgecolor="black",
            linewidth=0.3,
        )
        for bar in sorted(bars)
    ]
    legend_handles = [*patch_legend_handles]
    legend = ax.legend(
        handles=legend_handles,
        loc=(1.0, 0.5),  # (0.1,0.0),
        frameon=False,
        facecolor="none",
        fancybox=False,
    )  # bbox_to_anchor = "center")

    #                    handletextpad=0.2)
    # ax.set_ylim([0,1])

    set_size(*figsize, ax)
    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
