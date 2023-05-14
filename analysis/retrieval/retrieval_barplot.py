""" retrievl_barplot.py

Create retrieval classification barplot

"""
import argparse
from pathlib import Path
import pickle
import pandas as pd
from collections import Counter

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.patches import Patch, Rectangle
import matplotlib.ticker as mtick
from matplotlib.lines import Line2D
from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval-files",
        nargs="+",
        default=[
            "data/paired_spectra/csi2022/prev_results/csi_retrieval_Bayes_012.p",
            "results/2022_10_07_contrastive_best_csi/merged_retrieval/merged_contrast_only_retrieval_ind_found.p",
        ],
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument(
        "--spec-to-cls-file", default="data/unpaired_mols/bio_mols/new_smi_to_classes.p"
    )
    parser.add_argument("--png", default=False, action="store_true")
    # parser.add_argument("--model-names", nargs="+")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    retrieval_files = args.retrieval_files
    smi_to_cls_file = args.spec_to_cls_file
    smi_to_cls = pickle.load(open(smi_to_cls_file, "rb"))

    if save_name is None:
        pred_file = retrieval_files[-1]
        output_dir = Path(pred_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"retrieval_cls_barplot.{ext}"

    model_names = ["CSI:FingerID", "MIST"]
    dist_names = ["Bayes", "Contrastive"]
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

    # Start processing
    examples = []
    all_examples = []
    for name, old_method, new_method in zip(ret_names[0], ret_inds[0], ret_inds[1]):
        superclasses = smi_to_cls[name]["superclass_results"]
        for superclass in superclasses:
            new_entry = {
                "Superclass": superclass,
                "Name": name,
                "old_method": old_method,
                "new_method": new_method,
                "new_better": new_method < old_method,
                "old_better": old_method < new_method,
                "same": old_method == new_method,
            }
            examples.append(new_entry)
        new_entry_all = {
            "Superclass": superclasses[0] if len(superclasses) > 0 else "unknown",
            "Name": name,
            "old_method": old_method,
            "new_method": new_method,
            "new_better": new_method < old_method,
            "old_better": old_method < new_method,
            "same": old_method == new_method,
        }
        all_examples.append(new_entry_all)

    plt_df = pd.DataFrame(examples)
    all_df = pd.DataFrame(all_examples)
    print("Total num old better: ", all_df["old_better"].sum())
    print("Total num new better: ", all_df["new_better"].sum())
    print("Total num same: ", all_df["same"].sum())
    temp_df = all_df[(all_df["Superclass"] == "Sphingolipids")]
    temp_df = temp_df[temp_df["same"]]
    print("for sphingolipids, count of ranks:")
    print(Counter(temp_df["new_method"].values))
    print("for sphingolipids, total ranks:")
    print(len(temp_df["new_method"].values))

    summary_df = plt_df.groupby(["Superclass"])[
        ["new_better", "old_better", "same"]
    ].sum()
    summary_df["total"] = (
        summary_df["new_better"] + summary_df["old_better"] + summary_df["same"]
    )
    summary_df["diff"] = summary_df["new_better"] - summary_df["old_better"]
    summary_df["%"] = (summary_df["diff"] / summary_df["total"]) * 100
    summary_df["us_frac"] = (summary_df["new_better"] / summary_df["total"]) * 100
    summary_df["them_frac"] = (summary_df["old_better"] / summary_df["total"]) * 100
    summary_df["same_frac"] = (summary_df["same"] / summary_df["total"]) * 100

    summary_df = summary_df[summary_df["total"] > 40]
    summary_df = summary_df.sort_values(by="us_frac", ascending=False).reset_index()

    ax_sizes = (2.85, 0.8)
    fig = plt.figure(figsize=ax_sizes, dpi=400)

    ax = fig.gca()

    same_col = "#DFE1EC"
    us_col = "#6E85AE"
    them_col = "#F0C7B8"

    same, them, us = (
        summary_df["same_frac"].values,
        summary_df["them_frac"].values,
        summary_df["us_frac"].values,
    )
    x = summary_df["Superclass"].values
    x = [xx.strip() for xx in x]
    ax.bar(
        np.arange(len(x)),
        us,
        color=us_col,
        edgecolor="black",
        linewidth=0.3,
        label=r"MIST",
    )
    ax.bar(
        np.arange(len(x)),
        same,
        bottom=us,
        color=same_col,
        edgecolor="black",
        label=r"Tied",
        linewidth=0.3,
    )
    ax.bar(
        np.arange(len(x)),
        them,
        color=them_col,
        edgecolor="black",
        linewidth=0.3,
        bottom=same + us,
        label=r"CSI:FingerID",
    )

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, rotation=90)
    ax.set_xlim([-0.8, len(x) - 0.2])

    fmt = "%.0f%%"
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_ylabel("Class $\Delta$", labelpad=1.0)

    # ax.set_ylim([0,100]
    # Re-assign second ax labels
    set_size(*ax_sizes, ax)
    print(save_name)
    ax.legend(frameon=False, facecolor="none", fancybox=False, ncol=3, loc=(0.18, 1.0))

    fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)


if __name__ == "__main__":
    main()
