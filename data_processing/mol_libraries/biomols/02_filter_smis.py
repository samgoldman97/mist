""" Filter and potentially subsample smiles. """
import argparse
import numpy as np
from tqdm import tqdm

import mist.utils as utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unfiltered-smis", default="data/unpaired_mols/biomols/biomols_unfiltered.txt"
    )
    parser.add_argument(
        "--filtered-smis", default="data/unpaired_mols/biomols/biomols_filtered.txt"
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )

    return parser.parse_args()


def round_trip(x):
    try:
        mask = (
            utils.Chem.MolFromInchi(utils.Chem.MolToInchi(utils.Chem.MolFromSmiles(x)))
            is not None
        )
    except:
        mask = False

    return mask


def main():
    """main."""
    args = get_args()
    infile = args.unfiltered_smis
    outfile = args.filtered_smis
    debug = args.debug

    with open(infile, "r") as fp:
        in_smiles = [i.strip() for i in tqdm(fp.readlines())]

    if debug:
        in_smiles = in_smiles[:100000]

    min_formal_charges = utils.chunked_parallel(in_smiles, utils.min_formal_from_smi)
    max_formal_charges = utils.chunked_parallel(in_smiles, utils.max_formal_from_smi)

    all_forms = utils.chunked_parallel(in_smiles, utils.form_from_smi)
    num_atoms = utils.chunked_parallel(in_smiles, utils.atoms_from_smi)
    masses = utils.chunked_parallel(in_smiles, utils.mass_from_smi)

    is_not_empty = utils.chunked_parallel(all_forms, lambda x: len(x) > 0)
    single_fragment = utils.chunked_parallel(in_smiles, lambda x: "." not in x)
    not_none = utils.chunked_parallel(in_smiles, round_trip)

    only_valid_els = utils.chunked_parallel(all_forms, utils.has_valid_els)
    ge_3_atoms = np.array(num_atoms) > 3
    le_100_atoms = np.array(num_atoms) <= 100

    # Switch to 1450
    le_1450_mass = np.array(masses) <= 1450
    form_min_ge = np.array(min_formal_charges) >= -1
    form_max_le = np.array(max_formal_charges) <= 1

    mask = np.ones(len(in_smiles)).astype(bool)
    mask = np.logical_and(mask, np.array(is_not_empty))
    mask = np.logical_and(mask, np.array(single_fragment))
    mask = np.logical_and(mask, np.array(only_valid_els))
    mask = np.logical_and(mask, np.array(ge_3_atoms))
    mask = np.logical_and(mask, le_100_atoms)
    mask = np.logical_and(mask, le_1450_mass)
    mask = np.logical_and(mask, form_min_ge)
    mask = np.logical_and(mask, form_max_le)
    mask = np.logical_and(mask, not_none)
    print(np.array(not_none).sum())

    filtered_smis = np.array(in_smiles)[~mask].tolist()
    out_smiles = np.array(in_smiles)[mask].tolist()
    print(filtered_smis)

    print(f"Len of old smiles: {len(in_smiles)}")
    print(f"Len of out smiles: {len(out_smiles)}")
    print(f"Len of filtered smiles: {len(in_smiles) - len(out_smiles)}")

    with open(outfile, "w") as fp:
        fp.write("\n".join(out_smiles))


if __name__ == "__main__":
    main()
