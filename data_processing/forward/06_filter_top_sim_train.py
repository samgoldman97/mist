""" 06_filter_top_sim_train.py

Take the unpaired molecules and filter out by certain lengths

"""

from pathlib import Path
import pandas as pd
import pickle
from rdkit import Chem

import numpy as np

import mist.utils as utils


def inchikey_from_smi(smi):
    """inchikey_from_smi."""
    try:
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        return inchikey
    except:
        return ""


def main():
    """main."""
    # spec_mols = "data/paired_spectra/csi2022/labels.tsv"
    # ikey_values = set(pd.read_csv(spec_mols, sep="\t")['inchikey'].values)

    # dist_file = "data/unpaired_mols/bio_mols/max_sim_csi2022.p"
    unpaired_file_in = "data/unpaired_mols/bio_mols/all_smis_unfiltered.txt"
    unpaired_file_out = Path("data/unpaired_mols/bio_mols/all_smis.txt")
    # filter_thresh = 0.5
    max_set = 300000

    # dist_map = pickle.load(open(dist_file, "rb"))

    all_smi = [i.strip() for i in open(unpaired_file_in, "r").readlines()]

    # all_ikeys = utils.chunked_parallel(all_smi, inchikey_from_smi)

    # dists = np.array([dist_map.get(i, -1) for i in all_smi])
    # mask = dists > filter_thresh

    mask = np.zeros(len(all_smi))
    num_keep = min(len(mask), max_set)
    keep_inds = np.random.choice(len(mask), num_keep, replace=False)
    mask[keep_inds] = 1
    mask = mask.astype(bool)

    # mask_not_spec = np.array([i not in ikey_values for i in all_ikeys])
    # mask = np.logical_and(mask_not_spec, mask)

    print(f"Keeping {mask.sum()} of {len(mask)}")
    new_smi = np.array(all_smi)[mask]
    out_str = "\n".join(new_smi)
    with open(unpaired_file_out, "w") as fp:
        fp.write(out_str)


if __name__ == "__main__":
    main()
