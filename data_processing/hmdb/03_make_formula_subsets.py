""" 03_make_formula_subsets.py

Process hmdb smiles subsets

"""
import mist.retrieval_lib.form_subsets as form_subsets

hmdb_smi_file = "data/raw/hmdb/smiles.txt"
hmdb_form_file = "data/raw/hmdb/hmdb_formulae_inchikey.p"


if __name__ == "__main__":
    built_map = form_subsets.build_form_map(smi_file=hmdb_smi_file,
                                            dump_file=hmdb_form_file,
                                            debug=False)
