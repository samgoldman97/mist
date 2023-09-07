# Original data link
#SVM_URL="https://bio.informatik.uni-jena.de/wp/wp-content/uploads/2020/08/svm_training_data.zip"

export_link="https://www.dropbox.com/scl/fi/ddwa659ywqrsiolv9pxje/canopus_train_export_v2.tar?rlkey=1qgithmd6sztivwuqr4c505h5"
export_link="https://zenodo.org/record/8316682/files/canopus_train_export_v2.tar"

cd data/paired_spectra/
wget -O canopus_train_export.tar $export_link

tar -xvf canopus_train_export.tar
mv canopus_train_export canopus_train
rm -f canopus_train_export.tar
cd ../../
