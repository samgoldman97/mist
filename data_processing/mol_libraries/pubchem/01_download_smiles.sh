# Download CID-SMILES from source on pubchem
#wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz
#
## Unzip
#gunzip CID-SMILES.gz
#
#mkdir -p data/unpaired_mols/pubchem
#mv CID-SMILES data/unpaired_mols/pubchem/cid_smiles.txt

# Download as originally compared and prepared
wget https://zenodo.org/record/8084088/files/cid_smiles.txt
mkdir -p data/unpaired_mols/pubchem
mv cid_smles.txt data/unpaired_mols/pubchem/cid_smiles.txt
