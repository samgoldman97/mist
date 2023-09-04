""" After exporting mgf for augmentation, assign subformulae """
dataset_name="canopus_train"
split_no_ext="canopus_hplus_100_0"
labels_file="data/paired_spectra/${dataset_name}/aug_iceberg_${dataset_name}/biomols_filtered_smiles_${dataset_name}_labels.tsv"
mgf_file="data/paired_spectra/${dataset_name}/aug_iceberg_${dataset_name}/$split_no_ext/full_out.mgf"
output_dir="data/paired_spectra/${dataset_name}/aug_iceberg_${dataset_name}/$split_no_ext/subforms/"

# Subsets
python src/mist/subformulae/assign_subformulae.py \
    --spec-files $mgf_file \
    --labels-file $labels_file \
    --mass-diff-thresh 20 \
    --output-dir $output_dir \
    --max-formulae 50 \
    --num-workers 32 
