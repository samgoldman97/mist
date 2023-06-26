""" 03_filter_smis.py 

Filter and potentially subsample smiles

"""
import argparse
import numpy as np
import pickle

import mist.utils as utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smiles-pickle", default="data/unpaired_mols/bio_mols/smi_to_classes.p"
    )
    parser.add_argument(
        "--smiles-out", default="data/unpaired_mols/bio_mols/subsample_smi.txt"
    )
    return parser.parse_args()


def main():
    """main."""
    args = get_args()
    mapping = pickle.load(open(args.smiles_pickle, "rb"))
    in_smiles = list(mapping.keys())
    out_name = args.smiles_out

    # Debug
    # in_smiles = in_smiles[:100000]
    min_formal_charges = utils.chunked_parallel(in_smiles, utils.min_formal_from_smi)
    max_formal_charges = utils.chunked_parallel(in_smiles, utils.max_formal_from_smi)

    all_forms = utils.chunked_parallel(in_smiles, utils.form_from_smi)
    num_atoms = utils.chunked_parallel(in_smiles, utils.atoms_from_smi)
    masses = utils.chunked_parallel(in_smiles, utils.mass_from_smi)

    is_not_empty = utils.chunked_parallel(all_forms, lambda x: len(x) > 0)
    single_fragment = utils.chunked_parallel(in_smiles, lambda x: "." not in x)
    only_valid_els = utils.chunked_parallel(all_forms, utils.has_valid_els)
    ge_3_atoms = np.array(num_atoms) > 3
    le_100_atoms = np.array(num_atoms) <= 100
    le_1500_mass = np.array(masses) <= 1500
    form_min_ge = np.array(min_formal_charges) >= -1
    form_max_le = np.array(max_formal_charges) <= 1

    mask = np.ones(len(in_smiles)).astype(bool)
    mask = np.logical_and(mask, np.array(is_not_empty))
    mask = np.logical_and(mask, np.array(single_fragment))
    mask = np.logical_and(mask, np.array(only_valid_els))
    mask = np.logical_and(mask, np.array(ge_3_atoms))
    mask = np.logical_and(mask, le_100_atoms)
    mask = np.logical_and(mask, le_1500_mass)
    mask = np.logical_and(mask, form_min_ge)
    mask = np.logical_and(mask, form_max_le)

    filtered_smis = np.array(in_smiles)[~mask].tolist()
    out_smiles = np.array(in_smiles)[mask].tolist()
    print(filtered_smis)

    print(f"Len of old smiles: {len(in_smiles)}")
    print(f"Len of out smiles: {len(out_smiles)}")
    print(f"Len of filtered smiles: {len(in_smiles) - len(out_smiles)}")

    with open(out_name, "w") as fp:
        fp.write("\n".join(out_smiles))


if __name__ == "__main__":
    main()
