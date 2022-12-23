""" magma_utils.py

Additional utility file to assist with fingerprinting.

"""

import os
from ast import literal_eval

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def get_magma_fingerprint(smile):
    """ get_magma_fingerprint. """
    mol = Chem.MolFromSmiles(smile, sanitize=False)
    Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
        ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
    )
    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    fingerprint = np.zeros((0,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint


def get_magma_fingerprint_bits(smile):
    """ get magma fingerprint bits """
    fingerprint = get_magma_fingerprint(smile)
    hot_indices = list(np.where(np.array(list(fingerprint)) == 1)[0])
    return hot_indices


def read_magma_file(magma_frag_file):
    """Read in magma file"""
    if (
        magma_frag_file is not None
        and os.path.exists(magma_frag_file)
        and os.path.getsize(magma_frag_file) > 0
    ):

        # correct for inconsistency by me in file parsing (sad)
        sep = "\t"
        spectra_df = pd.read_csv(magma_frag_file, index_col=0, sep=sep)
        if (
            "smiles" not in spectra_df.columns
            or "chemical_formula" not in spectra_df.columns
        ):
            pass
        else:
            spectra_df = _convert_str_to_list(spectra_df, "smiles")
            spectra_df = _convert_str_to_list(spectra_df, "chemical_formula")
            if "mass_to_charge" not in spectra_df.columns:
                spectra_df["mass_to_charge"] = spectra_df["mz"]

            return spectra_df
    spectra_df = pd.DataFrame(
        columns=[
            "mass_to_charge",
            "intensity",
            "chemical_formula",
            "smiles",
            "molecule_peak",
        ]
    )
    return spectra_df


def _convert_str_to_list(df, column):
    """_convert_str_to_list"""
    df.loc[:, column] = df.loc[:, column].apply(
        lambda x: literal_eval(x) if x != "NAN" and not pd.isna(x) else []
    )
    return df
