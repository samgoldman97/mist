"""cls_barplot.py
Barplot of chemical classes

"""
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

from mist import utils
from rdkit import Chem
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

from mist.utils.plot_utils import *

set_style()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred-file", default="results/2022_08_22_mist_best_aug_lr/fp_preds_csi2022.p"
    )
    parser.add_argument(
        "--labels-file",
        default="data/paired_spectra/csi2022/labels.tsv",
        help="Labels to help map back to smiles",
    )
    parser.add_argument(
        "--baseline-file",
        default="data/paired_spectra/csi2022/prev_results/spectra_encoding_csi2022_Fold_0.p",
    )
    parser.add_argument(
        "--spec-to-cls-file", default="data/unpaired_mols/bio_mols/new_smi_to_classes.p"
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--png", default=False, action="store_true")
    return parser.parse_args()


def cos_sim(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    sim = cosine_similarity(pred, targ)
    sim = np.diag(sim)
    return sim[:, None]


def main():
    args = get_args()
    smi_to_cls_file = args.spec_to_cls_file
    pred_file = args.pred_file
    baseline_file = args.baseline_file
    labels = args.labels_file
    save_name = args.save_name
    debug = args.debug
    if save_name is None:
        output_dir = Path(pred_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "pdf" if not args.png else "png"
        save_name = f"chem_cls_heatmap.{ext}"
        save_name = output_dir / save_name

    labels = pd.read_csv(labels, sep="\t")

    # Get preds and sort
    fp_preds = pickle.load(open(pred_file, "rb"))
    a_names, a_preds, a_targs = fp_preds["names"], fp_preds["preds"], fp_preds["targs"]
    a_names = np.array(a_names)
    a_keep_set = set(a_names)

    # Get baselines
    b_preds = pickle.load(open(baseline_file, "rb"))
    b_names, b_preds, b_targs = b_preds["names"], b_preds["preds"], b_preds["targs"]
    b_names, b_preds, b_targs = np.array(b_names), np.array(b_preds), np.array(b_targs)
    b_keep_set = set(b_names)

    # Get set overlap of names
    keep_set = b_keep_set.intersection(a_keep_set)

    # Filter both down to overlap
    b_keep = [i in keep_set for i in b_names]
    b_names, b_preds, b_targs = b_names[b_keep], b_preds[b_keep], b_targs[b_keep]

    a_keep = [i in keep_set for i in a_names]
    a_names, a_preds, a_targs = a_names[a_keep], a_preds[a_keep], a_targs[a_keep]

    a_sort = np.argsort(a_names)
    b_sort = np.argsort(b_names)
    a_names, a_preds, a_targs = a_names[a_sort], a_preds[a_sort], a_targs[a_sort]
    b_names, b_preds, b_targs = b_names[b_sort], b_preds[b_sort], b_targs[b_sort]

    assert np.all(a_targs == b_targs)

    if debug:
        a_names, a_preds, a_targs = a_names[:100], a_preds[:100], a_targs[:100]
        b_names, b_preds, b_targs = b_names[:100], b_preds[:100], b_targs[:100]

    # Next compute matrix predictions
    # Start with likelihoods
    val_fn = cos_sim
    a_res = val_fn(a_preds, a_targs)
    b_res = val_fn(b_preds, b_targs)

    a_res = a_res.mean(-1)
    b_res = b_res.mean(-1)

    name_to_smi = dict(labels[["spec", "smiles"]].values)
    mol_list = [Chem.MolFromSmiles(name_to_smi[i]) for i in a_names]
    smi_list = [name_to_smi[i] for i in a_names]

    smi_to_cls = pickle.load(open(smi_to_cls_file, "rb"))
    examples = []
    for res_ind, (res_name, r_a, r_b) in enumerate(zip(a_names, a_res, b_res)):
        superclasses = smi_to_cls[res_name]["superclass_results"]
        for superclass in superclasses:
            new_entry = {
                "Superclass": superclass,
                "Name": res_name,
                "Smi": name_to_smi[res_name],
                "res_a": r_a,
                "res_b": r_b,
                "a>b": r_a > r_b,
                "a<b": r_a < r_b,
            }
            examples.append(new_entry)

    cls_df = pd.DataFrame(examples)
    gr_ct = cls_df.groupby(["Superclass"])[["a<b", "a>b"]].sum()
    gr_ct["total"] = gr_ct["a<b"] + gr_ct["a>b"]
    gr_ct["diff"] = gr_ct["a>b"] - gr_ct["a<b"]
    gr_ct["%"] = (gr_ct["diff"] / gr_ct["total"]) * 100

    gr_ct = gr_ct[gr_ct["total"] > 40]
    gr_ct_sorted = gr_ct.sort_values(by="%", ascending=False).reset_index()

    cool_warm = sns.color_palette("coolwarm", as_cmap=True)
    ax_sizes = (3.8, 0.5)
    fig = plt.figure(figsize=ax_sizes, dpi=400)
    ax = fig.gca()
    x, y = gr_ct_sorted["Superclass"].values, gr_ct_sorted["%"].values
    color = [cool_warm(yy / 200 + 0.5) for yy in y]
    x = [xx.strip() for xx in x]
    out = ax.bar(
        np.arange(len(x)), np.abs(y), color=color, edgecolor="black", linewidth=0.3
    )

    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, rotation=90)
    ax.set_xlim([-0.8, len(x) - 0.2])

    fmt = "%.0f%%"
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.set_ylabel("Class $\Delta$", labelpad=1.0)

    neg_ax = ax.twinx()
    neg_ax.set_yticks(ax.get_yticks())
    neg_ax.set_ylim(ax.get_ylim())

    fmt = "-%.0f%%"
    yticks = mtick.FormatStrFormatter(fmt)
    neg_ax.yaxis.set_major_formatter(yticks)
    set_size(*ax_sizes, ax)

    # Reset after
    neg_ax.set_yticks(ax.get_yticks())
    neg_ax.set_ylim(ax.get_ylim())

    fig.savefig(save_name, bbox_inches="tight", dpi=400)

    # Pull out smiles in top class that we do well on
    # two smiles in top class
    top_k = 4
    for k in range(top_k):
        cls = gr_ct_sorted["Superclass"][k]

        is_cls = cls_df["Superclass"] == cls
        is_better = cls_df["a>b"]
        mask = np.logical_and(is_cls, is_better)
        smis = np.random.choice(cls_df[mask]["Smi"], size=5)
        print(f"Strong examples in class {cls}")
        print("\n".join(smis))
        print("\n")
        # display(Draw.MolsToGridImage([Chem.MolFromSmiles(i) for i in smis], molsPerRow=5))


if __name__ == "__main__":
    main()
