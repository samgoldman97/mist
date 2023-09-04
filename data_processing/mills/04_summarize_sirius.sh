#conda activate ms-gen
echo "Summarizing mills"
python3 data_processing/sirius_processing/03_create_sirius_summary.py --labels-file  data/paired_spectra/mills/labels.tsv --sirius-folder data/paired_spectra/mills/sirius_outputs
