""" 04_create_smi_list_full.py

Create full smiles list including labels files

python3 data_processing/forward/04_create_smi_list_full.py --unpaired-smis data/unpaired_mols/bio_mols/subsample_smi.txt --labels data/paired_spectra/csi2022/labels.tsv --out data/unpaired_mols/bio_mols/all_smis.txt

"""

import argparse
import pandas as pd
import numpy as np
from rdkit import Chem
from mist import utils


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unpaired-smis", default="data/unpaired_mols/bio_mols/subsample_smi.txt"
    )
    parser.add_argument(
        "--labels-file",
        default="data/paired_spectra/csi2022/labels.tsv",
        help="labels to exclude",
    )
    parser.add_argument(
        "--out", default="data/unpaired_mols/bio_mols/all_smis_unfiltered.txt"
    )
    return parser.parse_args()


def inchikey_from_smi(smi):
    """inchikey_from_smi."""
    try:
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        return inchikey
    except:
        return ""


args = get_args()
labels_df = pd.read_csv(args.labels_file, sep="\t")
new_smis = [i.strip() for i in open(args.unpaired_smis, "r").readlines()]

smi_1 = set(labels_df["smiles"].values.tolist())

# Get old inchikeys
print("Get inchikeys from old smis")
old_inchikeys = set(utils.chunked_parallel(list(smi_1), inchikey_from_smi))
print("Get inchikeys from new smis")
new_inchikeys = utils.chunked_parallel(list(new_smis), inchikey_from_smi)


def okay_to_add(new_inchikey):
    return new_inchikey not in old_inchikeys


print("Find where new are not in old")
add_mask = utils.chunked_parallel(new_inchikeys, okay_to_add)

new_smis = np.array(new_smis)[add_mask].tolist()

smi_1 = list(smi_1)
smi_1.extend(new_smis)

all_smi = list(set(smi_1))
open(args.out, "w").write("\n".join(all_smi))
