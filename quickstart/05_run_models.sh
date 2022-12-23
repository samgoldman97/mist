# 05_run_models.py

# Predict fingerprints
labels="labels_with_putative_form.tsv"
fp_ckpt="pretrained_models/mist_fp_canopus_pretrain.ckpt"
contrast_ckpt="pretrained_models/mist_contrastive_canopus_pretrain.ckpt"

hdf_prefix="data/paired_spectra/quickstart/retrieval_hdf/quickstart_with_morgan4096_retrieval_db"
output_dir="quickstart/model_predictions"

mkdir $output_dir

# Predict fingerprint
# Note: Please ignore the inchi construction errors; in the prospective setting
# no molecules are made.
python3 run_scripts/pred_fp.py --num-workers 0 --labels-name $labels --dataset-name quickstart --model $fp_ckpt --save $output_dir

python3 run_scripts/retrieval_contrastive.py --num-workers 0 --dataset-name quickstart  --hdf-prefix $hdf_prefix --save $output_dir --labels-name $labels --model $contrast_ckpt --out-name "contrastive_rankings.p" --dist cosine
#

ranking_file="${output_dir}/contrastive_rankings.p"
smi_outs=$output_dir/"smiles_outputs.tsv"
names_file=${hdf_prefix}_names.p
python3 analysis/retrieval/create_smi_output.py --ranking $ranking_file --save-name $smi_outs --k 5 --names-file $names_file


# Contrastive embed
python3 run_scripts/embed_contrastive.py --num-workers 0 --model  \
$contrast_ckpt --dataset-name quickstart --save-dir $output_dir --labels-name \
$labels --out-name "contrastive_embed"
