# Download CID-SMILES
wget https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz

# Unzip
gunzip CID-SMILES.gz

mkdir data/raw/pubchem
mv CID-SMILES data/raw/pubchem/cid_smiles.txt
