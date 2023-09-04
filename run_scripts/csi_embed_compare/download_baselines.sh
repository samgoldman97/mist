# Update the environment
conda activate ms-gen
# Only necessary if running baselines
pip install spec2vec 
pip install ms2deepscore 
pip install matchms 

baseline_folder="results/csi_dist_compare/saved_models/"
mkdir -p $baseline_folder

wget https://zenodo.org/record/4699356/files/MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5?download=1
mv MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5?download=1 MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5
mv MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 $baseline_folder

#Download pretrained spec2vec
wget https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model?download=1
mv spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model?download=1 spec2vec.model
mv spec2vec.model $baseline_folder

wget https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.trainables.syn1neg.npy?download=1
mv spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.trainables.syn1neg.npy?download=1 spec2vec.model.trainables.syn1neg.npy
mv spec2vec.model.trainables.syn1neg.npy $baseline_folder

wget https://zenodo.org/record/4173596/files/spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.wv.vectors.npy?download=1
mv spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model.wv.vectors.npy?download=1 spec2vec.model.wv.vectors.npy
mv spec2vec.model.wv.vectors.npy $baseline_folder
