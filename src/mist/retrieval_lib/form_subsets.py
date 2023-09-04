""" form_subsets.py

Map a compound list of smiles strings to a set of formulae.

"""

import argparse
from pathlib import Path
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
                smi = line.split("\t")[-1].strip()
                smi_list.append(smi)
            if debug and index > 10000:
                return smi_list
    return smi_list


def calc_formula_to_moltuples(smi_list: List[str]) -> dict:
    """Map smiles to their formula + inchikey"""
    output_list = utils.chunked_parallel(smi_list, single_form_from_smi)
    outdict = defaultdict(lambda: [])
    form_to_ikeys = defaultdict(lambda: set())
    for entry in tqdm(output_list):
        form = entry.pop("form")
        if entry["ikey"] not in form_to_ikeys[form]:
            outdict[form].append(entry)
            form_to_ikeys[form].add(entry["ikey"])
    return dict(outdict)


def single_form_from_smi(smi: str) -> dict:
    """Get dict output"""
    null_dict = dict(form="", smi="", ikey="")
    try:
        mol = Chem.MolFromSmiles(smi)

        if mol is not None:
            form = utils.uncharged_formula(mol)

            # first remove stereochemistry
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            ikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))

            return dict(form=form, smi=smi, ikey=ikey)
        else:
            return null_dict
    except:
        return null_dict


def build_form_map(smi_file, dump_file=None, debug=False):
    """build_form_map."""
    smi_list = read_smi_txt(smi_file, debug=debug)
    form_to_mols = calc_formula_to_moltuples(smi_list)

    if dump_file is not None:
        with open(dump_file, "wb") as f:
            pickle.dump(form_to_mols, f)

    return form_to_mols

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-smiles", 
                        default="data/unpaired_mols/pubchem/cid_smiles.txt")
    parser.add_argument("--out-map",
                        default="formula_inchikey.p")
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    Path(args.out_map).parent.mkdir(exist_ok=True, parents=True)
    built_map = build_form_map(
        smi_file=args.input_smiles, dump_file=args.out_map, debug=args.debug
    )
