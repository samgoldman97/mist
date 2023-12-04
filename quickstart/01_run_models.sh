# Predict fingerprints
labels="quickstart/quickstart_labels.tsv"
smiles_lib="quickstart/lookup_smiles.txt"
fp_ckpt="pretrained_models/mist_fp_canopus_pretrain.ckpt"
contrast_ckpt="pretrained_models/mist_contrastive_canopus_pretrain.ckpt"
spec_mgf="quickstart/quickstart.mgf"

res_dir="quickstart/model_predictions"
res_subform=$res_dir/subforms/
mkdir -p $res_subform

# Create subform labels
python3 src/mist/subformulae/assign_subformulae.py \
    --spec-files $spec_mgf \
    --labels-file $labels \
    --mass-diff-thresh 20 \
    --output-dir $res_subform\
    --max-formulae 50 \
    --num-workers 32  \
    --feature-id FEATURE_ID 

# Predict fingerprints
python3 src/mist/pred_fp.py \
        --num-workers 0 \
        --labels-file $labels \
        --subform-folder $res_subform \
        --dataset-name quickstart \
        --model $fp_ckpt \
        --save-dir $res_dir/fp_preds

# Contrastive embedding
python3 src/mist/embed_contrast.py \
        --num-workers 0 \
        --labels-file $labels \
        --subform-folder $res_subform \
        --dataset-name quickstart \
        --model $contrast_ckpt \
        --save-dir $res_dir/contrastive_embed \
        --out-name  "contrastive_embeds.p"

# Map smiles 
outmap=$res_dir/lib/quickstart_formula_inchikey.p
python3 src/mist/retrieval_lib/form_subsets.py \
    --input-smiles  $smiles_lib \
    --out-map $outmap

# Build hdf 
python3 src/mist/retrieval_lib/make_hdf5.py \
    --form-to-smi-file $outmap \
    --labels-file $labels \
    --output-dir $res_dir/lib/ \
    --database-name quickstart \
    --fp-names morgan4096

retrieval_db=$res_dir/lib/quickstart_with_morgan4096_retrieval_db.h5
retrieval_dir=$res_dir/retrieval
fp_preds=$res_dir/fp_preds/fp_preds_quickstart.p


# FP retrieval on hdf
python3 src/mist/retrieval_fp.py \
    --dist-name cosine \
    --num-workers 0 \
    --labels-file $labels \
    --fp-pred-file $fp_preds \
    --save-dir $retrieval_dir \
    --hdf-file $retrieval_db \
    --top-k 200 \
    --output-tsv

# Contrastive retrieval
python3 src/mist/retrieval_contrast.py \
    --dataset-name quickstart \
    --hdf-file $retrieval_db \
    --labels-file $labels \
    --num-workers 10 \
    --dist-name cosine \
    --top-k 200 \
    --output-tsv \
    --subform-folder $res_subform \
    --save-dir $retrieval_dir \
    --model-ckpt $contrast_ckpt
