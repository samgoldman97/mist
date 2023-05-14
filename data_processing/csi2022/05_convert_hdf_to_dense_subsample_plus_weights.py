""" convert_hdf_to_dense_subsample_plus_weights.py

The full pubchem dataset we export is extremely large; create a subset and
convert to dense

To make it more tractable we had compressed it with gzip and multiple rounds,
but this makes it impractical for a pytorch dataset.

This script takes that output and reconverts it to be larger and uncompressed.

Note. Here, we specifically subsample so that we capture the highest tani
similar isomers

"""
from pathlib import Path
import pickle
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
from functools import partial

from mist import utils

debug = False
subset_size = 256
CHUNKSIZE = 1024
READ_FACTOR = 100
max_cpu = 32

my_dir = Path("data/paired_spectra/csi2022/retrieval_hdf/")
# my_dir = Path("/nfs/ccoleylab001/samlg/MassSpec/csi_decoy_db")

in_dir = Path("data/paired_spectra/csi2022/retrieval_hdf/")


input_hdf = in_dir / "pubchem_with_csi_retrieval_db.hdf5"
input_p = in_dir / "pubchem_with_csi_retrieval_db_index.p"

output_hdf = my_dir / "pubchem_with_csi_retrieval_db_sub.hdf5"
output_p = my_dir / "pubchem_with_csi_retrieval_db_index_sub.p"
output_weights = my_dir / "pubchem_with_csi_retrieval_db_precomputed_weights_sub.p"

# First uncompress for speed
labels_file = "data/paired_spectra/csi2022/labels.tsv"
features_hdf = "fingerprints/precomputed_fp/cache_csi_csi2022.hdf5"
features_p = "fingerprints/precomputed_fp/cache_csi_csi2022_index.p"

# Steps
# 1. Load labels
df = pd.read_csv(labels_file, sep="\t")
inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
inchikey_list = df["inchikey"].values

# 2. Load featurizations and create ar of inchikey, fp pairs
features_pickle_file = pickle.load(open(features_p, "rb"))
features_ar = h5py.File(features_hdf, "r")["features"][:]
inchikey_to_formula = dict(df[["inchikey", "formula"]].values)

search_tuple = [
    (k, features_ar[features_pickle_file[k]]) for k in inchikey_to_formula.keys()
]

# 3. Load in pickle_inds and pubchem inds
pickle_inds = pickle.load(open(features_p, "rb"))
pubchem_inds = pickle.load(open(input_p, "rb"))

# Tuples of inchikey to fingerprint
if debug:
    output_hdf = my_dir / "temp_pubchem_with_csi_retrieval_db_sub.hdf5"
    output_p = my_dir / "temp_pubchem_with_csi_retrieval_db_index_sub.p"
    CHUNKSIZE = 1
    READ_FACTOR = 100
    search_tuple = search_tuple[:100]

# Create a fn that is going to create a list containing indices maps
# I.e., function should take tuple of input fp, input chem formula and return
# which weights ot pull
def parallel_read_indices(
    input_tuple, hdf_file_name, label_to_formula, pickled_indices, k=10
):
    """parallel_read_indices.

    Args:
        input_tuple : Tuple: (inchikey, targ)
        hdf_file_name: hdf_file_name
        label_to_formula: Mapping from name to formula
        pickled_indices: Map from formula to hdf offset and lengths
        k: Num to return
    Return:
        ranking indices, sim tuples (none if not existing)
    """

    name, targ = input_tuple
    formula = label_to_formula.get(name)

    formula_dict = pickled_indices.get(formula)

    if formula_dict is None:
        print(f"Can't find {formula} in hdf")
        return None

    offset = formula_dict["offset"]
    length = formula_dict["length"]
    hdf_file = h5py.File(hdf_file_name, "r")

    sub_fps = hdf_file["fingerprints"][offset : offset + length]

    # Batched tanimoto
    intersect = targ[None, :] * sub_fps
    union = targ[None, :] + sub_fps - intersect

    # Calc sim and also set the true fp to 1e-12
    sim = intersect.sum(-1) / union.sum(-1)
    sim[sim == 1] = 1e-12

    order = np.argsort(sim)[::-1]

    # top k
    top_k_inds = order[:k]
    sim_k = sim[top_k_inds]
    top_k_inds = np.array(top_k_inds) + offset
    return (top_k_inds, sim_k)


# Goal is to construct a list of alllll indices to pull and all their tani
# weights
input_tuple = search_tuple
partial_fn = partial(
    parallel_read_indices,
    hdf_file_name=input_hdf,
    label_to_formula=inchikey_to_formula,
    pickled_indices=pubchem_inds,
    k=subset_size,
)
output_list = utils.chunked_parallel(
    input_tuple,
    partial_fn,
    max_cpu=max_cpu,
    chunks=max_cpu * 2,
    timeout=4000,
    max_retries=3,
)
inchikey_list = np.array(list(zip(*input_tuple))[0])
hdf_inds, hdf_weights = [np.array(i, dtype=object) for i in zip(*output_list)]

# Filter down to things that have entries
# have_entries = [len(i) > 0 for i in hdf_inds]
# inchikey_list = inchikey_list[have_entries]
# hdf_inds = hdf_inds[have_entries]
# hdf_weights = hdf_weights[have_entries]

# Roughly sort these again for portable refilling
reordering = np.argsort([i.min() if len(i) > 0 else 0 for i in hdf_inds])

inchikey_list = inchikey_list[reordering]
hdf_inds = hdf_inds[reordering]
hdf_weights = hdf_weights[reordering]

# Dump the weights
with open(output_weights, "wb") as fp:
    inchikey_to_weights = dict(zip(inchikey_list, hdf_weights))
    pickle.dump(inchikey_to_weights, fp)


# Create new index np array
cur_ind, new_ind_dict, ind_pairs = 0, {}, []
for inchikey_name, _hdf_inds in tqdm(
    zip(
        inchikey_list,
        hdf_inds,
    )
):

    ind_dict = {}
    new_offset, new_length = cur_ind, len(_hdf_inds)
    new_start, new_end = new_offset, new_offset + new_length

    new_ind_dict[inchikey_name] = {"offset": new_offset, "length": new_length}
    if new_length == 0:
        continue

    ind_pairs.extend(list(zip(_hdf_inds, np.arange(new_start, new_end))))
    cur_ind += new_length

# Conduct copy paste
with h5py.File(input_hdf, "r") as h5_input:
    source_file = h5_input["fingerprints"]
    source_shape = source_file.shape

    with h5py.File(output_hdf, "w") as h5f_output:
        fps_dset = h5f_output.create_dataset(
            "fingerprints",
            (cur_ind, source_file.shape[1]),
            chunks=(CHUNKSIZE, source_shape[1]),
            dtype="int8",
        )
        print(f"New database len: {cur_ind}")

        # Copy from old ind to new ind
        for copy_batch in tqdm(list(utils.batches(ind_pairs, CHUNKSIZE * READ_FACTOR))):
            prev_inds, new_inds = zip(*copy_batch)
            new_inds = np.array(new_inds)
            prev_inds = np.array(prev_inds)

            # Get source fps
            # Note: Reading a range is much more efficient, so query the block
            # first
            prev_inds_min, prev_inds_max = prev_inds.min(), prev_inds.max()
            all_from_source = source_file[prev_inds_min : prev_inds_max + 1]
            source_fps = all_from_source[prev_inds - prev_inds_min]
            fps_dset[new_inds] = source_fps

        fps_dset.resize(cur_ind, axis=0)
        print(f"Output fps total: {cur_ind}")
        with open(output_p, "wb") as fp:
            pickle.dump(new_ind_dict, fp)

        print(f"Output fps total: {cur_ind}")
        with open(output_p, "wb") as fp:
            pickle.dump(new_ind_dict, fp)
