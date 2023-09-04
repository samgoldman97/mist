""" Run magma on all training data"""
magma_file=src/mist/magma/run_magma.py

echo "Magma on canopus_train"
python3 $magma_file \
--spectra-dir data/paired_spectra/canopus_train/spec_files  \
--output-dir data/paired_spectra/canopus_train/magma_outputs  \
--spec-labels data/paired_spectra/canopus_train/labels.tsv  \
--max-peaks 50 \
--num-workers 60
#--debug
