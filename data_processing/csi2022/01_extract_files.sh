# Extract data from files
# Assume data has been downloaded and put in data/raw
unzip data/raw/csievaldata.zip
mv csievaldata/ data/raw/
cd data/raw/csievaldata
cd crossval
tar -xf spectra.tgz
tar -xf trees.tgz
cd ../independent
tar -xf spectra.tgz
tar -xf trees.tgz
cd ../../../../
