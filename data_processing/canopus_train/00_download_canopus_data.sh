SVM_URL="https://bio.informatik.uni-jena.de/wp/wp-content/uploads/2020/08/svm_training_data.zip"

wget $SVM_URL
unzip svm_training_data.zip
rm svm_training_data.zip
mv svm_training_data data/raw/
