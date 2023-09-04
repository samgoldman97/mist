# Download biomols from zenodo
wget  https://zenodo.org/record/8151490/files/biomols.zip
unzip biomols.zip

mv biomols data/unpaired_mols/

# Delete extraneous files
rm -f data/unpaired_mols/biomols/biomols_filter.txt
rm -f data/unpaired_mols/biomols/biomols_filter_formulae.txt
rm -f data/unpaired_mols/biomols/biomols_with_decoys.txt
rm -f data/unpaired_mols/biomols/biomols_with_decoys_split.tsv
rm -f biomols.zip
mv data/unpaired_mols/biomols/biomols.txt data/unpaired_mols/biomols/biomols_unfiltered.txt

