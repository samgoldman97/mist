""" Subset smiles for exporting molecules for prediction and forward augmentation."""

import argparse
import logging
from pathlib import Path
import pandas as pd
import pickle

from mist import utils
from mist.data import featurizers
import numpy as np
from tqdm import tqdm
from functools import partial


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles-file", default="data/unpaired_mols/biomols_filtered.txt"
    )
    parser.add_argument(
        "--labels-file", default="data/paired_spectra/canopus_train/labels.tsv"
    )
    parser.add_argument("--dataset-name", action="store", default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument(
        "--tani-thresh",
        action="store",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--sample-num",
        action="store",
        type=int,
        default=int(1e5),
    )
    parser.add_argument("--load-sims", default=False, action="store_true")
    return parser.parse_args()


def build_labels_df(
    smiles,
    instrument="Unknown (LCMS)",
    ion="[M+H]+",
):
    """build_labels_df"""
    dataset = ["forward_aug" for i in smiles]
    spec = [f"aug_{ind}" for ind, smi in enumerate(smiles)]
    name = [f"" for i in enumerate(smiles)]

    formula = utils.chunked_parallel(smiles, utils.form_from_smi)
    inchikey = utils.chunked_parallel(smiles, utils.inchikey_from_smiles)
    instruments = [instrument for i in smiles]
    ionizations = [ion for i in smiles]

    out_dict = {
        "dataset": dataset,
        "spec": spec,
        "smiles": smiles,
        "name": name,
        "formula": formula,
        "inchikey": inchikey,
        "instrument": instruments,
        "ionization": ionizations,
    }
    df = pd.DataFrame(out_dict)
    return df


def batch_calculation(input_batch: tuple, ref_fps):
    """batch_calculation"""
    test_smis, test_fps = zip(*input_batch)
    fp_x = ref_fps
    fp_y = np.vstack(test_fps)
    einsum_intersect = np.einsum("x i, y i -> xy", fp_x, fp_y)
    einsum_union = fp_x.sum(-1)[:, None] + fp_y.sum(-1)[None, :]
    einsum_union_less_intersect = einsum_union - einsum_intersect
    tani_pairwise = einsum_intersect / einsum_union_less_intersect
    tani_max = tani_pairwise.max(0)
    temp_dict = dict(zip(test_smis, tani_max))
    return temp_dict


def calc_sims(smiles_bank: list, ref_smiles: list, batch_size: int = 500) -> dict:
    """Calculate max sim dictionary"""
    # Create fingerprints of the entire bank of smiles
    featurizer = featurizers.FingerprintFeaturizer(fp_names=["morgan1024"])
    bank_fps = utils.chunked_parallel(smiles_bank, featurizer.featurize_smiles)
    bank_fps = np.vstack(bank_fps)

    # Create fingerprints of entire test bank
    ref_fps = utils.chunked_parallel(ref_smiles, featurizer.featurize_smiles)
    ref_fps = np.vstack(ref_fps)
    bank_fp_smi_pairs = list(zip(smiles_bank, bank_fps))

    partial_fn = partial(
        batch_calculation,
        ref_fps=ref_fps,
    )
    parallel_input = list(utils.batches(bank_fp_smi_pairs, batch_size))

    out_map_list = utils.chunked_parallel(parallel_input, partial_fn, max_cpu=10)
    out_map = {}
    for out_map_temp in out_map_list:
        out_map.update(out_map_temp)
    return out_map


def main():
    args = get_args()
    smi_file = Path(args.smiles_file)
    labels_file = Path(args.labels_file)
    dataset_name = args.dataset_name
    debug = args.debug
    tani_thresh = args.tani_thresh
    sample_num = args.sample_num
    load_sims = args.load_sims

    if dataset_name is None:
        dataset_name = Path(labels_file).parent.name

    # Bank of molecules to compare as ref
    labels_df = pd.read_csv(labels_file, sep="\t")
    ref_smiles = list(pd.unique(labels_df["smiles"]))

    # Test molecules
    with open(smi_file, "r") as fp:
        smiles_bank = [i.strip() for i in fp.readlines()]

    if debug:
        smiles_bank = smiles_bank[:20000]
        ref_smiles = ref_smiles[:5000]

    parent_dir = smi_file.parent
    file_stem = smi_file.stem
    sims_name = parent_dir / f"{file_stem}_sims_{dataset_name}.p"
    out_name = parent_dir / f"{file_stem}_smiles_{dataset_name}.txt"
    out_df = parent_dir / f"{file_stem}_smiles_{dataset_name}_labels.tsv"
    if load_sims and sims_name.exists():
        with open(sims_name, "rb") as fp:
            sim_dict = pickle.load(fp)
    else:
        sim_dict = calc_sims(smiles_bank, ref_smiles)
        with open(sims_name, "wb") as fp:
            pickle.dump(sim_dict, fp)

    filtered_smis = [k for k, v in sim_dict.items() if v < tani_thresh]
    logging.info(
        f"Len of valid smiles below {tani_thresh}: {len(filtered_smis)}/{len(sim_dict)}"
    )

    mask = np.zeros(len(filtered_smis))
    num_keep = min(len(mask), sample_num)
    keep_inds = np.random.choice(len(mask), num_keep, replace=False)
    mask[keep_inds] = 1
    mask = mask.astype(bool)

    new_smi = np.array(filtered_smis)[mask]
    logging.info(f"Filtered down to: {len(new_smi)}")

    out_str = "\n".join(new_smi)
    with open(out_name, "w") as fp:
        fp.write(out_str)

    labels_df = build_labels_df(
        new_smi,
        instrument="Unknown (LCMS)",
        ion="[M+H]+",
    )
    labels_df.to_csv(out_df, sep="\t", index=None)


if __name__ == "__main__":
    main()
