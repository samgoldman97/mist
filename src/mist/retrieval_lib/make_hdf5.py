"""make_hdf5.py

Make a retrieval hdf5 dataset. Takes in the pickled form_to_smiles object and
makes hdf5 for a given fingerprint type
"""

import argparse
from pathlib import Path
from functools import partial
import h5py
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm
from mist import utils
from mist.data import featurizers


CHUNKSIZE = 1024


def fingerprint_fn(smi, fp_names, fp_file=None):
    """fingerprint_smi."""
    featurizer = featurizers.FingerprintFeaturizer(
        fp_names=fp_names, fp_file=fp_file
    )
    return featurizer.featurize_smiles(smi)


def make_retrieval_hdf5(
    labels_file: str,
    form_to_smi: dict,
    output_dir: str,
    database_name="inthmdb",
    fp_names: list = ("morgan4096",),
    fp_file: str = None,
    debug: bool = False,
):
    """make_retrieval_hdf5."""

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    fp_names_str = "-".join(fp_names)

    output_file_h5 = output_dir / f"{database_name}_with_{fp_names_str}_retrieval_db.h5"

    data_df = pd.read_csv(labels_file, sep="\t").astype(str)
    all_forms = list(set(data_df["formula"].values))

    # Define fingerprint function
    fingerprint_smi = partial(fingerprint_fn, fp_names=fp_names, 
                              fp_file=fp_file,)

    dataset_len = np.sum([len(form_to_smi.get(i, [])) for i in all_forms])
    missing_forms = [i for i in all_forms if len(form_to_smi.get(i, [])) == 0]
    print(f"Dataset len: {dataset_len}")
    print(f"Num missing forms: {len(missing_forms)}")

    chunksize = min(CHUNKSIZE, dataset_len)
    print("Dumping to h5py")
    with h5py.File(output_file_h5, "w") as h:
        fp_size = featurizers.FingerprintFeaturizer.get_fingerprint_size(
            fp_names=fp_names
        )
        pack_size = np.packbits(np.ones(fp_size).astype(np.uint8)).shape[0]
        print(pack_size, fp_size)

        fps_dset = h.create_dataset(
            "fingerprints",
            (dataset_len, pack_size),
            dtype=np.uint8,
        )
        h.attrs["num_bits"] = fp_size

        cur_ind = 0
        new_ind_dict, new_name_dict = {}, {}
        formulae, formula_offset, formula_lengths, smiles, ikeys = [], [], [], [], []

        # Batch formulae
        # Get batches of 1000
        for form_batch in tqdm(utils.batches(all_forms, chunksize)):

            # Fingerprint in big batch
            all_smis = [j["smi"] for i in form_batch for j in form_to_smi.get(i, [])]
            all_ikeys = [j["ikey"] for i in form_batch for j in form_to_smi.get(i, [])]
            fingerprinted_smiles = utils.chunked_parallel(all_smis, fingerprint_smi)
            smi_to_fp = dict(zip(all_smis, fingerprinted_smiles))

            for form in tqdm(form_batch):

                # set of tuples of smiles, inchikeys
                dict_list = list(form_to_smi.get(form, []))
                new_smis = [i["smi"] for i in dict_list]
                new_ikeys = [i["ikey"] for i in dict_list]
                new_len = len(dict_list)

                if len(dict_list) == 0:
                    continue

                fingerprinted_smiles = [np.packbits(smi_to_fp[j]) for j in new_smis]
                fingerprint_batch = np.vstack(fingerprinted_smiles)
                fps_dset[cur_ind : cur_ind + new_len] = fingerprint_batch

                formulae.append(form)
                formula_offset.append(cur_ind)
                formula_lengths.append(new_len)

                smiles.extend(new_smis)
                ikeys.extend(new_ikeys)

                # Adjust out names and inds files
                cur_ind += new_len

        smile_edit = h.create_dataset("smiles", data=smiles, dtype=h5py.string_dtype())
        ikey_edit = h.create_dataset("ikeys", data=ikeys, dtype=h5py.string_dtype())
        string_dset = h.create_dataset(
            "formulae", data=formulae, dtype=h5py.string_dtype()
        )
        formulae_offset = h.create_dataset(
            "formula_offset", data=formula_offset, dtype=np.int32
        )
        formulae_offset = h.create_dataset(
            "formula_lengths", data=formula_lengths, dtype=np.int32
        )


def make_retrieval_hdf5_file(
    labels_file: str,
    form_to_smi_file: str,
    output_dir: str,
    database_name="inthmdb",
    fp_names: list = ("morgan4096",),
    fp_file: str = None,
    debug: bool = False,
):
    """make_retrieval_hdf5_file.

    Makes hdf5 retrieval accepting form_to_smi_file instead of form_to_smi
    dict.

    Args:
        labels_file (str):  labels
        form_to_smi_file (str): File mapping formula to inchi/smiles tuples
        ouptut_dir (str): Name of output dir 
        database_name (str): Name of database for outfile
        fp_names (tuple): List of strs to be used by featurizer for fingeprint
        fp_file (str): FP file
        debug (bool): debug flag
    """
    # Load in pickled mapping
    print("Loading in pickled formula map")
    with open(form_to_smi_file, "rb") as f:
        form_to_smi = pickle.load(f)
    print("Done loading in pickled formula map")
    make_retrieval_hdf5(
        output_dir=output_dir,
        labels_file=labels_file,
        form_to_smi=form_to_smi,
        database_name=database_name,
        fp_names=fp_names,
        fp_file=fp_file,
        debug=debug,
    )


def ind_rank_fn(ikey, smiles, form, fp_names, hdf_file, subset_size, fp_file):
    """ind_rank_fn."""
    true_fp = fingerprint_fn(smiles, fp_names=fp_names, fp_file=fp_file)

    with h5py.File(hdf_file, "r") as hdf_obj:
        form_ind = np.where(np.array(hdf_obj["formulae"]).astype(str) == form)[0]
        if len(form_ind) == 0:
            print(f"Can't find {form} in hdf")
            return None

        form_ind = form_ind[0]
        length = hdf_obj["formula_lengths"][form_ind]
        offset = hdf_obj["formula_offset"][form_ind]
        sub_fps = hdf_obj["fingerprints"][offset : offset + length]
        sub_ikeys = hdf_obj["ikeys"][offset : offset + length]
        sub_smiles = hdf_obj["smiles"][offset : offset + length]
        num_bits = hdf_obj.attrs["num_bits"]

    # Unpack fp
    sub_fps = utils.unpack_bits(sub_fps, num_bits)

    # Compute distance
    intersect = true_fp * sub_fps
    union = true_fp + sub_fps - intersect
    tani = intersect.sum(-1) / union.sum(-1)
    dist = 1 - tani
    not_self = dist != 0

    dist = dist[not_self]
    sub_smiles = sub_smiles[not_self]
    sub_ikeys = sub_ikeys[not_self]
    sub_fps = sub_fps[not_self]

    if len(dist) == 0:
        print(f"No non-self decoys")
        return None

    order = np.argsort(dist)
    order = order[:subset_size]

    sub_ikeys = sub_ikeys[order]
    sub_smiles = sub_smiles[order]
    sub_dist = dist[order]
    sub_fps = sub_fps[order]

    out_dict = {
        "ikeys": sub_ikeys,
        "smiles": sub_smiles,
        "dist": sub_dist,
        "fps": sub_fps,
        "base_ikey": ikey,
    }
    return out_dict


def export_contrast_h5(
    hdf_file,
    labels_file="labels.tsv",
    fp_names: tuple = ("morgan4096",),
    subset_size=256,
    num_workers=20,
    fp_file: str = None,
):
    """export_contrast_h5.

    Subset based on fingerprint name

    """
    hdf_file = Path(hdf_file)
    output_file = hdf_file.parent / f"{hdf_file.stem}_contrast.h5"

    # Steps
    df = pd.read_csv(labels_file, sep="\t").astype(str)
    inchikey_to_formula = dict(df[["inchikey", "formula"]].values)
    inchikey_list = list(set(list(df["inchikey"].values)))
    ikey_to_smi = dict(df[["inchikey", "smiles"]].values)
    ikey_to_form = dict(df[["inchikey", "formula"]].values)
    max_size = len(ikey_to_smi) * subset_size

    # Define fingerprint function
    compute_objs = [
        {
            "ikey": i,
            "smiles": ikey_to_smi[i],
            "form": ikey_to_form[i],
            "fp_names": fp_names,
            "fp_file": fp_file,
            "hdf_file": hdf_file,
            "subset_size": subset_size,
        }
        for i in inchikey_list
    ]

    apply_fn = lambda x: ind_rank_fn(**x)

    cur_ind = 0
    ikeys, smiles, base_ikeys, ikey_offset, ikey_lengths = [], [], [], [], []
    dists = []

    with h5py.File(output_file, "w") as h:
        chunksize = min(CHUNKSIZE, len(compute_objs))
        fp_size = featurizers.FingerprintFeaturizer.get_fingerprint_size(
            fp_names=fp_names
        )
        pack_size = np.packbits(np.ones(fp_size).astype(np.uint8)).shape[0]

        all_fps = []
        h.attrs["num_bits"] = fp_size
        # fps_dset = h.create_dataset(
        #    "fingerprints",
        #    (max_size, pack_size),
        #    dtype=np.uint8,
        # )
        for compute_batch in tqdm(utils.batches(compute_objs, chunksize)):
            # compute batch
            # Debug
            # int_outputs = [ind_rank_fn(**i) for i in tqdm(compute_batch)]
            int_outputs = utils.chunked_parallel(
                compute_batch, lambda x: ind_rank_fn(**x), max_cpu=num_workers
            )
            for entry in tqdm(int_outputs):

                if entry is None:
                    continue

                new_smis = entry["smiles"]
                new_ikeys = entry["ikeys"]
                new_dist = entry["dist"]
                new_len = len(new_smis)

                fingerprinted_smiles = [np.packbits(j) for j in entry["fps"]]
                fingerprint_batch = np.vstack(fingerprinted_smiles)
                # fps_dset[cur_ind: cur_ind + new_len] = fingerprint_batch
                all_fps.extend(fingerprint_batch)

                base_ikeys.append(entry["base_ikey"])
                ikey_offset.append(cur_ind)
                ikey_lengths.append(new_len)

                smiles.extend(new_smis)
                ikeys.extend(new_ikeys)
                dists.extend(new_dist)

                # Adjust out names and inds files
                cur_ind += new_len

        smile_edit = h.create_dataset("smiles", data=smiles, dtype=h5py.string_dtype())
        ikey_edit = h.create_dataset("ikeys", data=ikeys, dtype=h5py.string_dtype())
        string_dset = h.create_dataset(
            "base_ikeys", data=base_ikeys, dtype=h5py.string_dtype()
        )
        ikey_offset = h.create_dataset("ikey_offset", data=ikey_offset, dtype=np.int32)
        ikey_offset = h.create_dataset(
            "ikey_lengths", data=ikey_lengths, dtype=np.int32
        )
        dists = h.create_dataset("dists", data=dists, dtype=np.float32)

        fingerprints = np.vstack(all_fps)
        fps_dset = h.create_dataset(
            "fingerprints",
            data=fingerprints,
            dtype=np.uint8,
        )

        print(f"Full size {cur_ind} from start size {max_size}")
        # fps_dset.resize(cur_ind+1, axis=0)


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--form-to-smi-file")
    parser.add_argument("--labels-file")
    parser.add_argument("--database-name", default="pubchem")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--fp-file",
        default=None,
        action="store",
        help="Name of fp file with cached outputs"
    )
    parser.add_argument(
        "--fp-names",
        action="store",
        nargs="+",
        help="List of fp names for pred",
        default=["morgan2048"],
        choices=[
            "morgan512",
            "morgan1024",
            "morgan2048",
            "morgan_project",
            "morgan4096",
            "morgan4096_3",
            "maccs",
            "csi",
        ],
    )
    return parser.parse_args()


if __name__=="__main__": 
    args = get_args()
    make_retrieval_hdf5_file(
        labels_file=args.labels_file,
        form_to_smi_file=args.form_to_smi_file,
        output_dir=args.output_dir,
        database_name=args.database_name,
        fp_names=args.fp_names,
        debug=False,
        fp_file=args.fp_file,
    )
