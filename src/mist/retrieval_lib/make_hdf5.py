"""make_hdf5.py

Make a retrieval hdf5 dataset. Takes in the pickled form_to_smiles object and
makes hdf5 for a given fingerprint type
"""

from pathlib import Path
from functools import partial
import h5py
import pickle
import numpy as np

import pandas as pd

from tqdm import tqdm

from pathos import multiprocessing as mp
from mist import utils
from mist.data import featurizers

cpus = mp.cpu_count()

CHUNKSIZE = 1024


def make_retrieval_hdf5(dataset_name: str, labels_name: str,
                        form_to_smi: dict, database_name="inthmdb",
                        fp_names: list = ["morgan4096"], debug: bool = False,
                        data_dir = "data/paired_spectra"
                        ):
    """ make_retrieval_hdf5. 

    Args:
        dataset_name (str): dataset
        labels_name (str):  labels
        form_to_smi (dict): Dictionary mapping formula to inchi/smiles tuples
        database_name (str): Name of database for outfile
        fp_names (list): List of strs to be used by featurizer for fingeprint
        debug (bool): debug flag
    """

    data_dir = Path(data_dir)
    data_dir = data_dir / f"{dataset_name}"
    if not data_dir.exists():
        raise ValueError()

    fp_names_str = "-".join(fp_names)
    output_dir = data_dir / "retrieval_hdf"
    output_dir.mkdir(exist_ok=True)
    index_file = output_dir / f"{database_name}_with_{fp_names_str}_retrieval_db_index.p"
    name_file = output_dir / f"{database_name}_with_{fp_names_str}_retrieval_db_names.p"
    hdf_file = output_dir / f"{database_name}_with_{fp_names_str}_retrieval_db.hdf5"

    data_df = pd.read_csv(data_dir / labels_name, sep="\t")
    key = "formula"
    formulae = list(set(data_df[key].values))

    def fingerprint_smi(smi):
        """fingerprint_smi."""
        featurizer = featurizers.FingerprintFeaturizer(fp_names=fp_names)
        return featurizer.featurize_smiles(smi)

    dataset_len = np.sum([len(form_to_smi.get(i, [])) for i in formulae])
    missing_forms = [i for i in formulae if len(form_to_smi.get(i, [])) == 0]
    print(f"Dataset len: {dataset_len}")
    print(f"Num missing forms: {len(missing_forms)}")

    chunksize = min(CHUNKSIZE, dataset_len)
    print("Dumping to h5py")
    h = h5py.File(hdf_file, "w")
    fp_size = featurizers.FingerprintFeaturizer.get_fingerprint_size(fp_names=fp_names)
    fps_dset = h.create_dataset(
        "fingerprints",
        (dataset_len, fp_size),
        chunks=(chunksize, fp_size),
        dtype="int8",
    )

    cur_ind = 0
    new_ind_dict, new_name_dict = {}, {}

    # Batch formulae
    # Get batches of 1000
    for form_batch in tqdm(utils.batches(formulae, 1000)):

        # Fingerprint in big batch
        all_smis = [j[0] for i in form_batch for j in form_to_smi.get(i, [])]
        fingerprinted_smiles = utils.chunked_parallel(all_smis, fingerprint_smi)
        smi_to_fp = dict(zip(all_smis, fingerprinted_smiles))

        for form in tqdm(form_batch):
            # set of tuples of smiles, inchikeys
            tuple_list = list(form_to_smi.get(form, []))
            new_smis = [i[0] for i in tuple_list]
            #new_ikeys = [i[1] for i in tuple_list]
            new_len = len(tuple_list)

            if len(tuple_list) == 0:
                continue

            fingerprinted_smiles = [smi_to_fp[j] for j in new_smis]
            fingerprint_batch = np.vstack(fingerprinted_smiles)
            fps_dset[cur_ind: cur_ind + new_len] = fingerprint_batch

            new_ind_dict[form] = {"offset": cur_ind, "length": new_len}
            ind_to_smi = dict(zip(np.arange(cur_ind, cur_ind + new_len), new_smis))
            new_name_dict.update(ind_to_smi)

            # Adjust out names and inds files
            cur_ind += new_len

    fps_dset.resize(cur_ind, axis=0)
    with open(index_file, "wb") as fp:
        pickle.dump(new_ind_dict, fp)

    with open(name_file, "wb") as fp:
        pickle.dump(new_name_dict, fp)

def make_retrieval_hdf5_file(dataset_name: str, labels_name: str,
                             form_to_smi_file: str, database_name="inthmdb",
                             fp_names: list = ["morgan4096"], debug: bool = False):
    """make_retrieval_hdf5_file.

    Makes hdf5 retrieval accepting form_to_smi_file instead of form_to_smi
    dict.

    Args:
        dataset_name (str): dataset
        labels_name (str):  labels
        form_to_smi_file (str): File mapping formula to inchi/smiles tuples
        database_name (str): Name of database for outfile
        fp_names (list): List of strs to be used by featurizer for fingeprint
        debug (bool): debug flag
    """


    # Load in pickled mapping
    print("Loading in pickled formula map")
    with open(form_to_smi_file, "rb") as f:
        form_to_smi = pickle.load(f)
    print("Done loading in pickled formula map")
    make_retrieval_hdf5(dataset_name=dataset_name, labels_name=labels_name,
                        form_to_smi=form_to_smi, database_name=database_name,
                        fp_names=fp_names, debug=debug)


def make_ranking_file(dataset_name: str, hdf_prefix: str,
                      labels_name: str = "labels.tsv",
                      fp_names: list = ["morgan4096"], debug: bool = False,
                      num_workers=20,):
    """make_ranking_file.

    Makes hdf5 ranking file that maps each name to its true index in the hdf5
    file.

    Args:
        dataset_name (str): dataset
        hdf_prefix (str): Path leading up to prefix for the hdf5
        labels_name (str):  labels
        fp_names (list): List of strs to be used by featurizer for fingeprint
        debug (bool): debug flag
        num_workers (int): Num workers
    """
    hdf_prefix = Path(hdf_prefix)
    hdf_parent, hdf_stub = hdf_prefix.parent, hdf_prefix.stem
    index_file = hdf_parent / f"{hdf_stub}_index.p"
    name_file = hdf_parent / f"{hdf_stub}_names.p"
    hdf_file = hdf_parent / f"{hdf_stub}.hdf5"
    rank_file = hdf_parent / f"{hdf_stub}_ranked.p"
    labels_file = f"data/paired_spectra/{dataset_name}/{labels_name}"

    # Steps
    # 1. Load labels
    df = pd.read_csv(labels_file, sep="\t")
    inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
    inchikey_list = list(set(list(df["inchikey"].values)))
    ikey_to_smi = dict(df[["inchikey", "smiles"]].values)

    # 2. Create featurizer
    def fingerprint_smi(smi):
        """fingeprirnt_smi"""
        featurizer = featurizers.FingerprintFeaturizer(fp_names=fp_names)
        return featurizer.featurize_smiles(smi)

    features = {i: fingerprint_smi(ikey_to_smi[i]) for i in inchikey_list}

    # 3. Load hdf and pickle file
    pubchem_inds = pickle.load(open(index_file, "rb"))
    pubchem_names = pickle.load(open(name_file, "rb"))

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
    out_inchikeys = []
    for index, mol_batch in enumerate(tqdm(iterator, total=num_batches, unit="batch")):
        inds = np.arange(
            index * hdf_chunk_size, index * hdf_chunk_size + len(mol_batch)
        )
        true_fps, decoy_fps, fp_lens, offsets = [], [], [], []

        # Make a list of all things we actually want
        for ind, mol in tqdm(list(zip(inds, mol_batch))):
            true_mol_form = inchikey_to_formula.get(mol)

            ind_dict = pubchem_inds.get(true_mol_form)

            if ind_dict is None:
                continue

            offset, length = ind_dict["offset"], ind_dict["length"]

            # featurize
            true_fp = features[mol]

            true_fps.append(true_fp)
            out_inchikeys.append(mol)
            fp_lens.append(length)
            offsets.append(offset)

        # Parallel access hdf
        offset_lens = list(zip(offsets, fp_lens))

        def get_decoy_fps(input_tuple):
            offset, length = input_tuple
            pubchem_hdf = h5py.File(hdf_file, "r")["fingerprints"]
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
            # intersect = np.einsum("ij, kj -> ik", true_fps, padded_decoys)
            # union = true_fps.sum(-1)[:, None] + padded_decoys.sum(-1)[None, :] - intersect

            ## Calc sim and also set the true fp to 1e-12
            # tanimoto_sim = intersect / union

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
    for ind, k in enumerate(out_inchikeys):
        out_dict[k] = {
            "rankings": sample_weights[ind].tolist(),
            "true_ind": true_inds[ind].tolist(),
        }
    # 4. Dump to file
    with open(rank_file, "wb") as fp:
        pickle.dump(out_dict, fp)


def subsample_with_weights(hdf_prefix: str, labels_file: str,
                           fp_names: list = ["morgan4096"],
                           debug: bool = False, num_workers=20,):
    """subsample_with_weights.

    The full pubchem dataset we export is extremely large; create a subset and
    convert to dense so that it can be accessed efficiently during contrastive
    learning. We previously compressed the fingerprints in the hdf5, but this makes
    it impractical for a pytorch dataset.

    N.B. here, we specifically subsample so that we capture the highest tani
    similar isomers

    Args:
        hdf_prefix (str): Path leading up to prefix for the hdf5
        labels_file (str):  labels
        fp_names (list): List of strs to be used by featurizer for fingeprint
        debug (bool): debug flag
        num_workers (int): Num workers
    """
    subset_size = 256
    CHUNKSIZE = 1024
    READ_FACTOR = 100

    hdf_prefix = Path(hdf_prefix)
    hdf_parent, hdf_stub = hdf_prefix.parent, hdf_prefix.stem

    input_p = hdf_parent / f"{hdf_stub}_index.p"
    input_hdf = hdf_parent / f"{hdf_stub}.hdf5"
    output_hdf = hdf_parent / f"{hdf_stub}_sub.hdf5"
    output_p = hdf_parent / f"{hdf_stub}_sub_index.p"
    output_weights = hdf_parent / f"{hdf_stub}_precomputed_weights_sub.p"

    # Steps
    # 1. Load labels
    df = pd.read_csv(labels_file, sep="\t")

    inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
    inchikey_list = df["inchikey"].values

    # 2. Load featurizations and create ar of inchikey, fp pairs
    inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
    ikey_to_smi = dict(df[["inchikey", "smiles"]].values)

    featurizer = featurizers.FingerprintFeaturizer(fp_names=fp_names)
    search_tuple = [
        (k, featurizer.featurize_smiles(ikey_to_smi[k])) for k in inchikey_to_formula.keys()
    ]

    # 3. Load in pickle_inds and pubchem inds
    pubchem_inds = pickle.load(open(input_p, "rb"))

    # Tuples of inchikey to fingerprint
    if debug:
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
            return (np.array([]), np.array([]))

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


    # Goal is to construct a list of all indices to pull and all their tani
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
        max_cpu=num_workers,
        chunks=num_workers * 2,
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
