# Download sdf file
wget https://hmdb.ca/system/downloads/current/structures.zip

# Unzip
unzip structures.zip
rm structures.zip
mkdir data/raw/hmdb
mv structures.sdf data/raw/hmdb/
