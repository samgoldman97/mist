""" 03_make_formula_subsets.py

Process pubchem smiles subsets

"""
import mist.retrieval_lib.form_subsets as form_subsets

PUBCHEM_FILE = "data/raw/pubchem/cid_smiles.txt"
PUBCHEM_FORMULA = "data/raw/pubchem/pubchem_formuale_inchikey.p"


if __name__ == "__main__":
    built_map = form_subsets.build_form_map(smi_file=PUBCHEM_FILE,
                                            dump_file=PUBCHEM_FORMULA,
                                            debug=False)
