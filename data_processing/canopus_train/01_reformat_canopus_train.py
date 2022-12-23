""" 01_reformat_canpous_train.py.

Reformat thee "svm_training_data" from sirius paper

"""
from pathlib import Path
import argparse

from rdkit import Chem
import pandas as pd
import numpy as np

from functools import partial
from tqdm import tqdm

from mist import utils


# The following output smiles do not map
# Missing smiles:
# Consider remapping them with pubchem
# C=CC=CCC(=O)N1CCC23c4ccccc4NC4N(C)c5cccc6c5C42CCN(C6C2OC2(C)C)C13
# C=CC=CCC(=O)N1CCC23c4ccccc4NC4Nc5cccc6c5C42CCN(C6C2OC2(C)C)C13
# CC(O)CC(=O)OC(C)CC(=O)OC(C)CC(=O)OC(C)CC(=O)OC(C)CC(=O)OC(C)CC(=O)OC(C)CC(=O)OC(C)CC(=O)OC(C)CC(=O)O
# CC(C)CC1NC(=O)C(NC(=O)c2ncccc2O)C(C)OC(=O)C(c2ccccc2)N(C)C(=O)CNC(=O)C(C(C)C(C)C)NC(=O)CN(C)C(=O)C2CC(O)CN2C1=O
# CCCC(=O)NC(C(=O)NC(C(=O)NC(CC(C)C)C(O)CC(=O)NC(C)C(=O)NC(CC(C)C)C(O)CC(=O)O)C(C)C)C(C)C
# CCC(C)C1NC(=O)C2CCCN2C(=O)C(CCC(C)C)OC(=O)CCNC(=O)C(C)NC(=O)C(C(C)C)NC1=O
# CC12CC3([S+](=O)([O-])O)OC(O1)C1(COC(=O)c4ccccc4)C3CC21OC1OC(CO)C(O)C(O)C1O
# CC1=C2C(=O)C=C(CO)C2C2OC(=O)C(C[S+](=O)([O-])O)C2C(OC(=O)Cc2ccc(O)cc2)C1
# NC(C[SH](=O)=O)C(=O)O


parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()
DEBUG = args.debug

dataset_name = "canopus_train"
if DEBUG:
    dataset_name = "canopus_train_debug"

# Source locations
SVM_DIRECTORY = Path("data/raw/svm_training_data")
LABEL_SRC = SVM_DIRECTORY / "compound_ids.tsv"
SPECTRA_SRC = SVM_DIRECTORY / "public_data"

# Target locations
TARGET_DIRECTORY = Path(f"data/paired_spectra/{dataset_name}")
SPECTRA_TRG = TARGET_DIRECTORY / "spec_files"
LABEL_TRG = TARGET_DIRECTORY / "labels.tsv"


def get_ionization(file_):
    """get_ionization.

    Args:
        file_:
    """
    file_name = Path(file_).stem
    ionization = utils.parse_spectra(file_)[0].get("ionization", "")

    # Replace
    ionization = ionization.replace(" ", "")
    return file_name, ionization


def main():
    """main."""
    # Make target directories
    TARGET_DIRECTORY.mkdir(exist_ok=True)
    SPECTRA_TRG.mkdir(exist_ok=True)

    source_spectra = list(SPECTRA_SRC.glob("*.ms"))

    ionization_map = dict(
        utils.chunked_parallel(source_spectra, get_ionization, chunks=1000)
    )

    ## Create output labels
    # Copy true compound labels
    df = pd.read_csv(LABEL_SRC, sep="\t")

    if DEBUG:
        df = df[:100]

    smile_vals = df["standardized SMILES"]
    orig_formulae = [utils.uncharged_formula(i, mol_type="smiles") for i in smile_vals]

    # Standardize smiles
    smiles_standardizer = utils.SmilesStandardizer()
    smiles = [smiles_standardizer.standardize_smiles(x) for x in tqdm(smile_vals)]
    mols = [Chem.MolFromSmiles(i) for i in smiles]
    inchikeys = [Chem.MolToInchiKey(i) for i in mols]

    # new smiles
    output_formulae = [utils.uncharged_formula(i, mol_type="mol") for i in mols]
    unequal_forms = np.array([i != j for i, j in zip(output_formulae, orig_formulae)])

    print("\n".join(smile_vals[unequal_forms]))

    not_eq = np.sum(unequal_forms)
    print(f"{not_eq} formulae are different out of {len(orig_formulae)}")

    not_eq = np.sum([i != j for i, j in zip(smiles, smile_vals)])
    print(f"{not_eq} smiles are different out of {len(smiles)}")

    output_df = {
        "smiles": smiles,
        "formula": output_formulae,
        "name": "",
        "ionization": [ionization_map.get(j) for j in df["name"].values],
        "spec": df["name"].values,
        "inchikey": inchikeys,
    }

    df_out = pd.DataFrame(output_df)
    df_out["dataset"] = dataset_name
    df_out = df_out[
        ["dataset", "spec", "name", "ionization", "formula", "smiles", "inchikey"]
    ]
    df_out.to_csv(LABEL_TRG, sep="\t", index=False)

    valid_specs = set(df_out["spec"].values)
    # valid_formulae = set(df_out["formula"].values)
    copy_specs(valid_specs)


def copy_specs(valid_spec):
    """copy_specs.

    Args:
        valid_spec:
    """

    # Copy all spectra files
    source_spectra = [i for i in SPECTRA_SRC.glob("*") if i.stem in valid_spec]

    # Do bulk copy
    def read_write_spec(x, targ):
        """read_write_spec."""
        meta, spec = utils.parse_spectra(x)

        meta_keys = list(meta.keys())
        meta_keep = [
            "compound",
            "formula",
            "parentmass",
            "ionization",
            "InChi",
            "InChIKey",
        ]
        meta_comment = set(meta_keys).difference(set(meta_keep))

        out_meta = "\n".join([f">{i} {meta.get(i, None)}" for i in meta_keep])
        out_comment = "\n".join([f"#{i} {meta.get(i, None)}" for i in meta_comment])

        peak_list = []
        for k, v in spec:
            peak_entry = []
            peak_entry.append(f">{k}")
            peak_entry.extend([f"{row[0]} {row[1]}" for row in v])
            peak_list.append("\n".join(peak_entry))

        out_peaks = "\n\n".join(peak_list)

        total_str = f"{out_meta}\n{out_comment}\n\n{out_peaks}"
        outpath = Path(targ) / Path(x).name

        with open(outpath, "w") as fp:
            fp.write(total_str)

    single_func = partial(read_write_spec, targ=SPECTRA_TRG)
    utils.chunked_parallel(source_spectra, single_func, chunks=100)


if __name__ == "__main__":
    main()
