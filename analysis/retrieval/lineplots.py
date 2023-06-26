""" lineplots.py

Create lineplots of retrieval

Example:

"""
import argparse
from pathlib import Path
import pickle
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-files",
        nargs="+",
        default=[
            "results/2022_08_31_retrieval_fig/csi_retrieval_cosine_0.p",
            "results/2022_08_31_retrieval_fig/csi_retrieval_Bayes.p",
            "results/2022_08_31_retrieval_fig/retrieval_contrast_pubchem_with_csi_retrieval_db_csi2022_cosine_0_ind_found.p",
            "results/2022_08_31_retrieval_fig/retrieval_contrast_pubchem_with_csi_retrieval_db_csi2022_cosine_0_merged_dist_0_7_ind_found.p",
        ],
    )
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--png", default=False, action="store_true")
    # parser.add_argument("--model-names", nargs="+")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    ext = "png" if png else "pdf"
    save_dir = args.save_dir
    retrieval_files = args.retrieval_files

    if save_dir is None:
        pred_file = retrieval_files[-1]
        output_dir = Path(pred_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        save_dir = output_dir
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

    save_name_20 = f"csi_lineplot_20.{ext}"
    save_name_20 = save_dir / save_name_20

    save_name_200 = f"csi_lineplot_200.{ext}"
    save_name_200 = save_dir / save_name_200

    model_names = ["CSI:FingerID", "CSI:FingerID", "MIST", "MIST"]
    dist_names = ["Cosine", "Bayes", "Cosine", "Contrastive"]
    assert len(retrieval_files) == len(dist_names)

    # Extract rankings from file
    ret_names, ret_inds = [], []
    for i in retrieval_files:
        with open(i, "rb") as fp:
            a = pickle.load(fp)
            ind_found, names = np.array(a["ind_found"]), np.array(a["names"])
            sort_order = np.argsort(names)
            names = names[sort_order]
            ind_found = ind_found[sort_order]

            ret_names.append(names)
            ret_inds.append(ind_found)

    # Calc common inds and subset
    common_inds = None
    for i, j in zip(ret_names, ret_inds):
        i = i[~np.isnan(j.astype(float))]
        temp_names = set(i)
        if common_inds is None:
            common_inds = temp_names
        else:
            common_inds = common_inds.intersection(temp_names)

    # Re-mask each based upon common inds
    new_names, new_inds = [], []
    for ret_name, ret_ind in zip(ret_names, ret_inds):
        mask = [i in common_inds for i in ret_name]
        new_names.append(ret_name[mask])
        new_inds.append(ret_ind[mask])
    ret_inds = new_inds
    ret_names = new_names

    # Create top k
    k_vals = np.arange(0, 1001)
    # max_k = np.max(k_vals) + 1
    top_k_x, top_k_y = [], []
    for ret_ind in ret_inds:
        new_x, new_y = [], []
        for k in k_vals:
            new_x.append(k)
            new_y.append(np.mean(ret_ind <= k))
        top_k_x.append(new_x), top_k_y.append(new_y)

    # Start making plots
    model_colors = {"CSI:FingerID": "#EFC6B7", "MIST": "#6F84AE"}

    dist_styles = {
        "Cosine": "-",
        "Bayes": ":",
        "Contrastive": "--",
    }

    ax_figsize = (1.25, 1.85)
    fig = plt.figure(figsize=(ax_figsize))
    ax = fig.gca()
    for x, y, model, dist in zip(top_k_x, top_k_y, model_names, dist_names):

        color, style = model_colors.get(model), dist_styles.get(dist)
        ax.step(x[0:], y[0:], c=color, linestyle=style, linewidth=0.8)

    ax.set_xlim([0, 20])
    ax.set_ylim([0.35, 1.0])
    ax.set_xlabel("Top K")
    ax.set_ylabel("Accuracy")

    # Build legend
    patch_legend_handles = [
        Patch(facecolor=model_colors[model], label=model) for model in set(model_names)
    ]
    patch_legend_handles.append(
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    )

    patch_legend_handles.insert(
        0,
        Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False, label="Model"
        ),
    )
    line_legend_handles = [
        Line2D(
            [0, 1],
            [0, 1],
            linestyle=dist_styles[dist],
            linewidth=0.8,
            color="black",
            label=dist,
        )
        for dist in set(dist_names)
    ]
    line_legend_handles.insert(
        0,
        Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False, label="Distance"
        ),
    )
    legend_handles = [*patch_legend_handles, *line_legend_handles]

    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",  # (0.1,0.0),
        frameon=False,
        facecolor="none",
        fancybox=False,  # ncol=2,
        columnspacing=0.5,
    )
    set_size(*ax_figsize)
    fig.savefig(save_name_20, bbox_inches="tight", dpi=400, transparent=True)

    ## Top 200
    fig = plt.figure(figsize=(ax_figsize))
    ax = fig.gca()
    for x, y, model, dist in zip(top_k_x, top_k_y, model_names, dist_names):

        color, style = model_colors.get(model), dist_styles.get(dist)
        ax.step(x[0:], y[0:], c=color, linestyle=style, linewidth=0.8)

    ax.set_xlim([1, 200])
    ax.set_ylim([0.7, 1.0])

    ax.set_xlabel("Top K")
    ax.set_ylabel("Accuracy")

    # Build legend
    patch_legend_handles = [
        Patch(facecolor=model_colors[model], label=model) for model in set(model_names)
    ]

    patch_legend_handles.append(
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)
    )

    patch_legend_handles.insert(
        0,
        Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False, label="Model"
        ),
    )
    line_legend_handles = [
        Line2D(
            [0, 1],
            [0, 1],
            linestyle=dist_styles[dist],
            linewidth=0.8,
            color="black",
            label=dist,
        )
        for dist in set(dist_names)
    ]
    line_legend_handles.insert(
        0,
        Rectangle(
            (0, 0), 1, 1, fill=False, edgecolor="none", visible=False, label="Distance"
        ),
    )
    legend_handles = [*patch_legend_handles, *line_legend_handles]
    legend = ax.legend(
        handles=legend_handles,
        loc="lower right",  # (0.1,0.0),
        frameon=False,
        facecolor="none",
        fancybox=False,  # ncol=2,
        columnspacing=0.5,
    )
    set_size(*ax_figsize)
    fig.savefig(save_name_200, bbox_inches="tight", dpi=400, transparent=True)
    print(save_name_20, save_name_200)


if __name__ == "__main__":
    main()
