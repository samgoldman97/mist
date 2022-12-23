""" form_subsets.py

Map a compound list of smiles strings to a set of formulae.

"""

from typing import List, Tuple
import pickle
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

import mist.utils as utils


def read_smi_txt(smi_file, debug=False):
    """Load in smiles txt file with one smiles per line"""
    smi_list = []
    with open(smi_file, "r") as fp:
        for index, line in enumerate(fp):
            line = line.strip()
            if line:
                smi = line.split("\t")[0].strip()
                smi_list.append(smi)
            if debug and index > 10000:
                return smi_list
    return smi_list


def calc_formula_to_moltuples(smi_list: List[str]) -> dict:
    """Map smiles to their formula + inchikey"""
    output_list = utils.chunked_parallel(smi_list, single_form_from_smi)
    formulae, mol_tuples = zip(*output_list)

    outdict = defaultdict(lambda: set())
    for mol_tuple, formula in tqdm(zip(mol_tuples, formulae)):
        outdict[formula].add(mol_tuple)
    return dict(outdict)


def single_form_from_smi(smi: str) -> Tuple[str, Tuple[str, str]]:
    """Compute single formula + inchi key from a smiles string"""
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            form = utils.uncharged_formula(mol)

            # first remove stereochemistry
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            inchi_key = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

            return form, (smi, inchi_key)
        else:
            return "", ("", "")
    except:
        return "", ("", "")


def build_form_map(smi_file, dump_file=None, debug=False):
    """ build_form_map. """
    smi_list = read_smi_txt(smi_file, debug=debug)
    form_to_mols = calc_formula_to_moltuples(smi_list)

    if dump_file is not None:
        with open(dump_file, "wb") as f:
            pickle.dump(form_to_mols, f)

    return form_to_mols
