# Process pubchem smiles subsets
hmdb_file="data/unpaired_mols/hmdb/smiles.txt"
hmdb_formula="data/unpaired_mols/hmdb/hmdb_formula_inchikey.p"
python3 src/mist/retrieval_lib/form_subsets.py \
    --input-smiles  $hmdb_file \
    --out-map $hmdb_formula
