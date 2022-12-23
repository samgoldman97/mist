""" 07_get_rank_order.py

Transform hdf file candidates into a rank ordering of indices

"""
import pandas as pd
from pathlib import Path
import pickle
import h5py
from tqdm import tqdm
import numpy as np

from mist import utils

import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.DEBUG)

debug = False
num_workers = 30

my_dir = Path("data/paired_spectra/csi2022/retrieval_hdf/")
# my_dir = Path("/nfs/ccoleylab001/samlg/MassSpec/csi_decoy_db")

in_dir = Path("data/paired_spectra/csi2022/retrieval_hdf/")

# First uncompress for speed
labels_file = "data/paired_spectra/csi2022/labels.tsv"
features_hdf = "fingerprints/precomputed_fp/cache_csi_csi2022.hdf5"
features_p = "fingerprints/precomputed_fp/cache_csi_csi2022_index.p"

input_hdf = in_dir / "pubchem_with_csi_retrieval_db.hdf5"
input_p = in_dir / "pubchem_with_csi_retrieval_db_index.p"
output_p = my_dir / "pubchem_with_csi_retrieval_db_ranked.p"

# Steps
# 1. Load labels
df = pd.read_csv(labels_file, sep="\t")
inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
inchikey_list = list(set(list(df["inchikey"].values)))

# 2. Load featurizations
features_pickle_file = pickle.load(open(features_p, "rb"))
features_hdf = h5py.File(features_hdf, "r")["features"]

# 3. Load hdf and pickle file
pubchem_inds = pickle.load(open(input_p, "rb"))
pubchem_hdf = h5py.File(input_hdf, "r")["fingerprints"]

# 3. Load compute output
# Iterating over the dataset, create batches of size chunk_size
# where each chunk is tuples of true mol fingerprint, decoy
# fingerprints
# Do this because we can't pickle the hdf itself, and we can't load
# it all into numpy memory

if debug:
    inchikey_list = inchikey_list[:30]

# Read in chunks of size 500
hdf_chunk_size = 500
parallel_chunk_size = 5
if debug:
    hdf_chunk_size = 10
    parallel_chunk_size = 3
res_sample_weights = []
res_true_inds = []

# Loop over the hdf
iterator = utils.batches(inchikey_list, hdf_chunk_size)
num_batches = len(inchikey_list) // hdf_chunk_size + 1
for index, mol_batch in enumerate(tqdm(iterator, total=num_batches, unit="batch")):
    inds = np.arange(index * hdf_chunk_size, index * hdf_chunk_size + len(mol_batch))
    true_fps, decoy_fps, fp_lens, offsets = [], [], [], []

    # Make a list of all things we actually want
    for ind, mol in tqdm(list(zip(inds, mol_batch))):
        true_mol_form = inchikey_to_formula.get(mol)

        ind_dict = pubchem_inds.get(true_mol_form)

        offset, length = ind_dict["offset"], ind_dict["length"]

        # featurize
        true_fp = features_hdf[features_pickle_file.get(mol)]

        true_fps.append(true_fp)
        fp_lens.append(length)
        offsets.append(offset)

    # Parallel access hdf
    offset_lens = list(zip(offsets, fp_lens))

    def get_decoy_fps(input_tuple):
        offset, length = input_tuple
        pubchem_hdf = h5py.File(input_hdf, "r")["fingerprints"]
        return pubchem_hdf[offset : offset + length]

    decoy_fps = utils.simple_parallel(
        offset_lens,
        get_decoy_fps,
        max_cpu=num_workers,
        timeout=60 * 3,
    )

    # For each batch we've sampled, we should process in parallel
    # However, each parallel computations are themselves more efficient
    # when they are vectorized into batches. Thus, with this
    # dataset into memory, we will chunk once more down to
    # num_workers chunks
    fp_lengths = np.array(fp_lens)
    true_fps = np.vstack(true_fps)
    decoy_fps = decoy_fps

    def get_weights(input_tuple, epsilon=1e-12):
        """Batch this calculation"""

        # Inputs
        fp_lengths, true_fps, decoy_fps, offsets = zip(*input_tuple)
        true_fps = np.vstack(true_fps)
        fp_lengths = np.array(fp_lengths)

        # Padding
        fp_shape = true_fps.shape[-1]
        max_fp_len = np.max(fp_lengths)
        pad_amts = max_fp_len - fp_lengths

        padded_decoys = [
            np.vstack([decoys, np.zeros((padding, fp_shape))])
            for decoys, padding in zip(decoy_fps, pad_amts)
        ]
        padded_decoys = np.stack(padded_decoys)

        # Batched tanimoto
        intersect = true_fps[:, None, :] * padded_decoys
        union = true_fps[:, None, :] + padded_decoys - intersect
        tanimoto_sim = intersect.sum(-1) / union.sum(-1)

        # Set all 1's to 0!
        # Do this for the true example
        # is_equiv = tanimoto_sim == 1

        # Define mask for lengths
        valid_decoys = np.arange(max_fp_len)[None, :] < fp_lengths[:, None]

        # Add in epsilon for valid examples
        tanimoto_sim[valid_decoys] += epsilon

        # Sample weights is tani dist
        sample_weights = 1 - tanimoto_sim  # / tanimoto_sim.sum(-1)[:, None]

        # 2D vector of weights
        # Sort from low to high distance
        sample_weights = [
            np.argsort(i[j]) + offset
            for i, j, offset in zip(sample_weights, valid_decoys, offsets)
        ]

        true_inds = [
            np.argwhere(i[j] >= 1).flatten() + offset
            for i, j, offset in zip(tanimoto_sim, valid_decoys, offsets)
        ]

        # Argsorted indices
        return sample_weights, true_inds

    # Batch the current batch into several sub matrices for
    # parallel
    iterator = list(
        utils.batches(
            zip(fp_lengths, true_fps, decoy_fps, offsets), parallel_chunk_size
        )
    )
    # Apply this in parallel
    # 3D vector of vector of vectors
    sample_weights = utils.simple_parallel(
        iterator,
        get_weights,
        max_cpu=num_workers,
        timeout=60 * 10,
    )
    sample_weights, true_inds = zip(*sample_weights)

    res_sample_weights.extend([j for i in sample_weights for j in i])
    res_true_inds.extend([j for i in true_inds for j in i])

sample_weights = res_sample_weights
true_inds = res_true_inds
out_dict = {}
for ind, k in enumerate(inchikey_list):
    out_dict[k] = {
        "rankings": sample_weights[ind].tolist(),
        "true_ind": true_inds[ind].tolist(),
    }
# 4. Dump to file
with open(output_p, "wb") as fp:
    pickle.dump(out_dict, fp)
