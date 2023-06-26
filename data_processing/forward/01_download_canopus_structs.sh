wget https://bio.informatik.uni-jena.de/wp/wp-content/uploads/2020/04/structures.csv.gz
gunzip structures.csv.gz
mkdir data/unpaired_mols/bio_mols
mv structures.csv data/unpaired_mols/bio_mols/bio_inchis.csv
