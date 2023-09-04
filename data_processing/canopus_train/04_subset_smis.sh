""" Build iceberg augmentation dataset  """
dataset_name=canopus_train
sample_num=200000

# Subsets
python src/mist/retrieval_lib/subset_smis.py \
        --smiles-file data/unpaired_mols/biomols/biomols_filtered.txt \
        --labels-file data/paired_spectra/${dataset_name}/labels.tsv \
        --dataset-name $dataset_name \
        --tani-thresh 0.8 \
        --sample-num $sample_num  #\
        #--load-sims

mkdir -p data/paired_spectra/${dataset_name}/aug_iceberg_${dataset_name}/
