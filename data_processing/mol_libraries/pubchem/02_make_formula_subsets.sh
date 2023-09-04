# Process pubchem smiles subsets
pubchem_file="data/unpaired_mols/pubchem/cid_smiles.txt"
pubchem_formula="data/unpaired_mols/pubchem/pubchem_formula_inchikey.p"
python3 src/mist/retrieval_lib/form_subsets.py \
    --input-smiles  $pubchem_file \
    --out-map $pubchem_formula
