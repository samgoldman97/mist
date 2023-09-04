dataset_name="csi2022"
output_dir="data/paired_spectra/${dataset_name}/subformulae/default_subformulae"
python src/mist/subformulae/assign_subformulae.py \
    --spec-files data/paired_spectra/${dataset_name}/spec_files/ \
    --labels-file data/paired_spectra/${dataset_name}/labels.tsv \
    --mass-diff-thresh 20 \
    --output-dir $output_dir \
    --max-formulae 50 \
    --num-workers 32  #\
