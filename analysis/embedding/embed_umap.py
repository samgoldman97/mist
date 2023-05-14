""" embed_umap.py

Create barplot

"""
import argparse
from pathlib import Path
import pickle
import pandas as pd
import umap
from collections import Counter

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
        "--umap-embeddings",
        default="results/2022_10_07_contrastive_best_csi/2022_10_07-1835_891915_8bdcd80687294392e00d01ae2eec1cad/embed/embed_csi2022_0.p",
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
    smi_to_cls_file = args.spec_to_cls_file
    smi_to_cls = pickle.load(open(smi_to_cls_file, "rb"))
    embed_file = Path(args.umap_embeddings)

    if save_name is None:
        output_dir = embed_file.parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"embed_umap.{ext}"

    embeddings = pickle.load(open(embed_file, "rb"))
    embeds, names = embeddings["embeds"], embeddings["names"]
    # Extract classes
    classes = []
    for name in names:
        superclasses = smi_to_cls[name]["superclass_results"]
        if len(superclasses) == 0:
            classes.append("Unknown")
        else:
            classes.append(np.random.choice(superclasses))

    classes = np.array(classes)
    # Lower min dist and lwoer n neighbors makes it a ltitle more clumped
    reducer = umap.UMAP(metric="cosine", min_dist=0.1, n_neighbors=15, spread=1.0)
    umap_embeds = reducer.fit_transform(
        embeds,
    )

    # Calculate the order of classes
    uniq_classes = list(set(classes))
    num_classes = len(uniq_classes)

    class_counts = Counter(classes)
    top_class_order = np.argsort([class_counts[i] for i in uniq_classes])[::-1]
    top_classes = np.array(uniq_classes)[top_class_order]

    # Build plot gruops
    k = 15
    plot_groups = []
    plot_group_names = []
    for j in top_classes[:k]:
        inds = np.where(classes == j)[0]
        plot_groups.append(inds)
        plot_group_names.append(j)

    # Build colors
    colors = sns.color_palette("husl", n_colors=len(plot_groups))

    # Resort to organize
    plot_group_order = [
        "Unknown",
        "Oligopeptides",
        "Small peptides",
        "Lysine alkaloids",
        "Tyrosine alkaloids",
        "Tryptophan alkaloids",
        "Nicotinic acid alkaloids",
        "Anthranilic acid alkaloids",
        "Ornithine alkaloids",
        "Pseudoalkaloids",
        "Flavonoids",
        "Coumarins",
        "Steroids",
        "Triterpenoids",
        "Sesquiterpenoids",
    ]
    priority_dict = dict(zip(plot_group_order, np.arange(len(plot_group_order))))
    argsort_basis = [priority_dict.get(i, 100) for i in plot_group_names]
    new_order = np.argsort(argsort_basis)
    plot_groups = np.array(plot_groups, dtype=object)[new_order]
    plot_group_names = np.array(plot_group_names)[new_order]

    name_to_color = dict(zip(plot_group_names, colors))

    # name_to_color['Unknown'] = "white"
    name_to_alpha = {i: 0.9 for i in name_to_color}
    name_to_alpha["Unknown"] = 0.8
    name_to_edge_color = {i: "none" for i in name_to_color}
    name_to_edge_color["Unknown"] = "#EFC7B8"  # name_to_color['Unknown']
    name_to_color["Unknown"] = "none"

    ax_size = (1.8, 2.0)

    def make_plots(ax_size=(1.8, 1.8)):
        fig = plt.figure(figsize=ax_size)
        ax = fig.gca()
        for inds, name in zip(plot_groups, plot_group_names):
            out = ax.scatter(
                umap_embeds[inds, 0],
                umap_embeds[inds, 1],
                label=name,
                color=name_to_color[name],
                facecolors=name_to_color[name],
                edgecolors=name_to_edge_color[name],
                s=0.6,
                linewidth=0.35,
                alpha=name_to_alpha[name],
            )

        ax.legend(
            loc="upper left",
            frameon=False,
            facecolor="none",
            fancybox=False,
            markerscale=3,
            bbox_to_anchor=(0.91, 0.95),
            labelspacing=0.3,
            # bbox_to_anchor=(-0.5, 0.95)  , #
            ncol=1,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_xlabel("UMAP 1")
        # ax.set_ylabel("UMAP 2")
        ax.axis("off")
        set_size(*ax_size)
        print(save_name)
        fig.savefig(save_name, bbox_inches="tight", dpi=400, transparent=True)

    # Hide in function for debugging
    make_plots(ax_size)


if __name__ == "__main__":
    main()
