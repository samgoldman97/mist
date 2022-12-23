""" 10_add_forward_decoys.py

For contrastive learning, this adds a set of forward decoys and hdf files to
the subset


"""
import argparse
from pathlib import Path
import pickle
import h5py
import numpy as np
import pandas as pd
from functools import partial
from tqdm import tqdm

from mist import utils
from mist.data import featurizers

debug = False
k = 256
CHUNKSIZE = 128
max_cpu = 32

rnd_subset = 100000


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-file",
        default="data/paired_spectra/csi2022/forward_preds_out/labels.tsv",
    )
    parser.add_argument(
        "--fp-names",
        action="store",
        nargs="+",
        help="List of fp names for pred",
        default=["morgan4096"],
        choices=[
            "morgan512",
            "morgan1024",
            "morgan2048",
            "morgan_project",
            "morgan4096",
            "morgan4096_3",
            "maccs",
            "map2048",
            "maccs-cdk",
            "klekota-roth",
            "cdk",
            "pubchem",
            "FP3",
            "contextgin",
            "csi",
        ],
    )
    parser.add_argument(
        "--output-folder", default="data/paired_spectra/csi2022/forward_preds_out"
    )
    return parser.parse_args()


args = get_args()
labels_file = args.labels_file
output_file = Path(args.output_folder) / "decoy_weight_inds.p"
dataset_name = Path(labels_file).parent.parent.stem
fp_names = args.fp_names

fp_names_str = "-".join(fp_names)
output_h5 = Path(args.output_folder) / f"{fp_names_str}_fp.hdf"

df = pd.read_csv(labels_file, sep="\t")
inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
inchikey_list = df["inchikey"].values
smiles_list = df["smiles"].values


# 2. Featurize smiles

batches = utils.batches(inchikey_list, 4)

# Inchikey list
if "csi" in fp_names:
    featurizer = featurizers.FingerprintFeaturizer(
        fp_names=fp_names, dataset_name=dataset_name
    )
    all_fps = np.vstack([featurizer.featurize_smiles(i) for i in tqdm(smiles_list)])
else:

    def fingerprint_smi(smi):
        """fingerprint_smi"""
        featurizer = featurizers.FingerprintFeaturizer(
            fp_names=fp_names,
        )
        return featurizer.featurize_smiles(smi)

    # all_fps = np.vstack([fingerprint_smi(i) for i in smiles_list])
    all_fps = np.vstack(utils.chunked_parallel(smiles_list, fingerprint_smi))

with h5py.File(output_h5, "w") as h5_out:
    fps_dset = h5_out.create_dataset(
        "fingerprints",
        all_fps.shape,
        chunks=(CHUNKSIZE * 32, all_fps.shape[-1]),
        dtype="int8",
    )
    fps_dset[:] = all_fps
    fps_dset.resize(all_fps.shape[0], axis=0)


# 3. Compute decoy distances one at a time
def extract_top_k(fps, all_fps, k):
    """extract_top_k."""

    subset_amt = min(rnd_subset, len(all_fps))
    new_inds = np.random.choice(len(all_fps), subset_amt, replace=False)
    all_fps = all_fps[new_inds]

    # Batched tanimoto
    intersect = np.einsum("ij, kj -> ik", fps, all_fps)
    union = fps.sum(-1)[:, None] + all_fps.sum(-1)[None, :] - intersect

    # Calc sim and also set the true fp to 1e-12
    sim = intersect / union

    # Cosine similarity
    sim[sim == 1] = 1e-12
    order = np.argsort(sim, -1)[:, ::-1]

    # top k closest
    top_k_inds = order[:, :k]
    sim_closest = np.take_along_axis(sim, top_k_inds, axis=1)

    top_k_inds = new_inds[top_k_inds]

    return (top_k_inds, sim_closest)


partial_fn = partial(extract_top_k, all_fps=all_fps, k=k)
# compute batched tani similarity
fp_batches = utils.batches(all_fps, CHUNKSIZE)
fp_batches = [np.vstack(i) for i in fp_batches]
output_list = utils.chunked_parallel(
    fp_batches,
    partial_fn,
    max_cpu=max_cpu,
    chunks=max_cpu * 3,
    timeout=100000,
    max_retries=3,
)
inds = [j for i in output_list for j in i[0]]
weights = [j for i in output_list for j in i[1]]
ind_weight_pairs = list(zip(inds, weights))
output = dict(zip(inchikey_list, ind_weight_pairs))
with open(output_file, "wb") as fp:
    inchikey_to_weights = dict(zip(inchikey_list, ind_weight_pairs))
    pickle.dump(inchikey_to_weights, fp)
input = pickle.load(open(output_file, "rb"))
