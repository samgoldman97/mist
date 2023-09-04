magma_file=src/mist/magma/run_magma.py

echo "Magma on canopus_train"
python3 $magma_file \
--spectra-dir data/paired_spectra/csi2022/spec_files  \
--output-dir data/paired_spectra/csi2022/magma_outputs  \
--spec-labels data/paired_spectra/csi2022/labels.tsv  \
--max-peaks 50 \
--num-workers 60 #\
#--debug
