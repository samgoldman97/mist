""" tani_lineplot.py

Create ground truth tanimoto lineplot

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
from sklearn.metrics.pairwise import cosine_similarity

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import AllChem, DataStructs

from mist.utils.plot_utils import *

set_style()


def get_morgan_fp(mol: Chem.Mol, nbits: int = 2048) -> np.ndarray:
    """get_morgan_fp."""

    if mol is None:
        return None

    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits)

    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-file", default="data/paired_spectra/csi2022/labels.tsv"
    )
    parser.add_argument(
        "--s2v-embed-file", 
        default="results/2023_04_30_embed_matchms/spec2vec_out.p",
        #default="results/2022_10_10_embedded_outs/spec2vec_out.p"
    )
    parser.add_argument(
        "--ms2deep-embed-file", default="results/2023_04_30_embed_matchms/ms2deepscore_out.p"
    )
    parser.add_argument(
        "--mist-embed-file",
        default="results/2022_10_27_contrastive_best_ensemble/embeds/embed_csi2022_0.p"
    )
    parser.add_argument(
        "--mod-cos-file", 
        default="results/2023_04_30_embed_matchms/cosine_out.p",
        #default="results/2022_10_10_embedded_outs/cos_out.p"
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--png", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")

    # parser.add_argument("--model-names", nargs="+")
    return parser.parse_args()


def main():
    args = get_args()
    png = args.png
    save_name = args.save_name
    debug = args.debug

    mist_embed_file = args.mist_embed_file
    s2v_embed_file = args.s2v_embed_file
    ms2deep_embed_file = args.ms2deep_embed_file
    mod_cos_file = args.mod_cos_file
    label_file = args.labels_file

    labels = pd.read_csv(label_file, sep="\t")
    name_to_smi = dict(labels[["spec", "smiles"]].values)

    if save_name is None:
        output_dir = Path(mist_embed_file).parent / "plots"
        output_dir.mkdir(exist_ok=True)
        ext = "png" if png else "pdf"
        save_name = output_dir / f"spec2vec_compare_revision.{ext}"

    # MIST embeds
    mist_embeddings = pickle.load(open(mist_embed_file, "rb"))
    mist_embeds, mist_names = mist_embeddings["embeds"], mist_embeddings["names"]

    if debug:
        mist_embeds = mist_embeds[:100]
        mist_names = mist_names[:100]

    argsort_names = np.argsort(mist_names)
    mist_names = mist_names[argsort_names]
    mist_embeds = mist_embeds[argsort_names]

    # S2V embeds
    s2v = pickle.load(open(s2v_embed_file, "rb"))
    s2v_embeds, s2v_names = s2v["embeds"], np.array(s2v["names"])

    # Extract and subset s2v
    s2v_name_sort = np.argsort(s2v_names)
    s2v_embeds = s2v_embeds[s2v_name_sort]
    s2v_names = s2v_names[s2v_name_sort]
    embed_names_set = set(mist_names)
    s2v_mask = [i in embed_names_set for i in s2v_names]
    s2v_embeds = s2v_embeds[s2v_mask]
    s2v_names = s2v_names[s2v_mask]


    # MS2Deep embeds
    ms2deep = pickle.load(open(ms2deep_embed_file, "rb"))
    ms2deep_embeds, ms2deep_names = ms2deep["embeds"], np.array(ms2deep["names"])

    # Extract and subset ms2deep
    ms2deep_name_sort = np.argsort(ms2deep_names)
    ms2deep_embeds = ms2deep_embeds[ms2deep_name_sort]
    ms2deep_names = ms2deep_names[ms2deep_name_sort]
    embed_names_set = set(mist_names)
    ms2deep_mask = [i in embed_names_set for i in ms2deep_names]
    ms2deep_embeds = ms2deep_embeds[ms2deep_mask]
    ms2deep_names = ms2deep_names[ms2deep_mask]


    # Get modified cosine
    mod_cos_out = pickle.load(open(mod_cos_file, "rb"))
    mod_cos_names, mod_cos_dists = mod_cos_out["names"], np.array(
        mod_cos_out["pairwise_cos"]
    )

    subsetted = [i in embed_names_set for i in mod_cos_names]
    mod_cos_names = mod_cos_names[subsetted]
    mod_cos_dists = mod_cos_dists[subsetted, :][:, subsetted]

    resorted = np.argsort(mod_cos_names)
    mod_cos_names = mod_cos_names[resorted]
    mod_cos_dists = mod_cos_dists[:, resorted][resorted, :]

    # Morgan fingerprint
    # Get all morgan fingerprints
    mols_list = [Chem.MolFromSmiles(name_to_smi.get(i)) for i in mist_names]
    stacked_fp = np.vstack([get_morgan_fp(i) for i in mols_list])

    # All by all dist
    stacked_fp_prev = stacked_fp
    stacked_fp = stacked_fp_prev
    # Compute tanimoto similarity ofall true fingerprints
    einsum_intersect = np.einsum("x i, y i -> xy", stacked_fp, stacked_fp)
    einsum_union = stacked_fp.sum(-1)[None, :] + stacked_fp.sum(-1)[:, None]
    einsum_union_less_intersect = einsum_union - einsum_intersect
    tani_pairwise = einsum_intersect / einsum_union_less_intersect

    # Zero diagonal
    num_items = tani_pairwise.shape[0]
    tani_pairwise[np.arange(num_items), np.arange(num_items)] = 0
    all_pairs_flat = tani_pairwise.flatten()

    # Get x plot coordinates
    if debug:
        interval = 1
        percentile = 1
    else:
        interval = 100
        percentile = 0.001

    total_pairs = all_pairs_flat.shape[0]
    num_pairs_consider = int(total_pairs * percentile)
    pair_cutoffs = np.arange(1, num_pairs_consider)[interval::interval]
    pair_cutoff_percentages = (pair_cutoffs / total_pairs) * 100

    print("Done with loading, computing theoretical max")

    # Get theoretical max
    sorted_by_max = np.argsort(all_pairs_flat)
    vals_by_max_sort = all_pairs_flat[sorted_by_max[::-1]]
    avg_percentiles_max = [np.mean(vals_by_max_sort[:i]) for i in pair_cutoffs]

    print("Computing mist dists")

    # Get MIST
    mist_all_by_all = cosine_similarity(mist_embeds, mist_embeds)
    mist_all_by_all[np.arange(num_items), np.arange(num_items)] = -10
    flattened_mist = mist_all_by_all.flatten()
    sorted_by_mist = np.argsort(flattened_mist)
    vals_by_mist_sort = all_pairs_flat[sorted_by_mist[::-1]]
    avg_percentiles_mist = [np.mean(vals_by_mist_sort[:i]) for i in pair_cutoffs]

    print("Computing s2v dists")

    # Get spec2vec
    num_items_s2v = s2v_embeds.shape[0]
    s2v_all_by_all = cosine_similarity(s2v_embeds, s2v_embeds)
    s2v_all_by_all[np.arange(num_items_s2v), np.arange(num_items_s2v)] = -10
    flattened_s2v = s2v_all_by_all.flatten()
    sorted_by_s2v = np.argsort(flattened_s2v)
    vals_by_s2v_sort = all_pairs_flat[sorted_by_s2v[::-1]]
    avg_percentiles_s2v = [np.mean(vals_by_s2v_sort[:i]) for i in pair_cutoffs]

    print("Computing ms2deep dists")

    # Get spec2vec
    num_items_ms2deep = ms2deep_embeds.shape[0]
    ms2deep_all_by_all = cosine_similarity(ms2deep_embeds, ms2deep_embeds)
    ms2deep_all_by_all[np.arange(num_items_ms2deep), np.arange(num_items_ms2deep)] = -10
    flattened_ms2deep = ms2deep_all_by_all.flatten()
    sorted_by_ms2deep = np.argsort(flattened_ms2deep)
    vals_by_ms2deep_sort = all_pairs_flat[sorted_by_ms2deep[::-1]]
    avg_percentiles_ms2deep = [np.mean(vals_by_ms2deep_sort[:i]) for i in pair_cutoffs]


    print("Computing rnd dists")

    # random baseline
    random_inds = np.arange(len(all_pairs_flat))
    np.random.shuffle(random_inds)
    vals_by_rnd_sort = all_pairs_flat[random_inds[::-1]]
    avg_percentiles_rnd = [np.mean(vals_by_rnd_sort[:i]) for i in pair_cutoffs]

    print("Computing mod cos dists")

    # mod cos baseline
    num_items = mod_cos_dists.shape[0]
    mod_cos_dists[np.arange(num_items), np.arange(num_items)] = -10
    mod_cos_dists_flat = mod_cos_dists.flatten()
    sorted_by_mod_cos = np.argsort(mod_cos_dists_flat)
    vals_by_mod_cos_sort = all_pairs_flat[sorted_by_mod_cos[::-1]]
    avg_percentiles_mod_cos = [np.mean(vals_by_mod_cos_sort[:i]) for i in pair_cutoffs]

    # Plotting code
    colors = ["#841619", "#EECCBE", "#98BFC4", "#7D8FB9", "#245F92"]
    colors = ["#1D2452", "#6F84AE",  "#69B367", "#98BFC4",
              "#EECCBE", "#F2CA9A"]
    ax_figsize = (1.25, 1.85)
    fig = plt.figure(figsize=(ax_figsize))
    ax = fig.gca()
    for name, percentiles, color in zip(
        ["Upper bound", "MIST",  "MS2DeepScore", "Spec2Vec",
         "Modified Cosine", "Random"],
        [
            avg_percentiles_max,
            avg_percentiles_mist,
            avg_percentiles_ms2deep,
            avg_percentiles_s2v,
            avg_percentiles_mod_cos,
            avg_percentiles_rnd,
        ],
        colors,
    ):
        ax.plot(
            pair_cutoff_percentages, percentiles, label=name, linewidth=0.8, color=color
        )
        print(f"Cutoff for {name} at {pair_cutoff_percentages[-1]}: {percentiles[-1]}")

    ax.legend(frameon=False, facecolor="none", fancybox=False)
    ax.set_xlabel("Spectral similarity percentile")
    ax.set_ylabel("Structural similarity")
    set_size(*ax_figsize)
    print(save_name)
    fig.savefig(save_name, bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    main()
