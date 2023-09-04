CUDA_VISIBLE_DEVICES=1,2,3 python3 src/mist/hyperopt_xformer.py \
--cache-featurizers \
--fp-names morgan4096 \
--labels-file data/paired_spectra/canopus_train/labels.tsv \
--spec-folder data/paired_spectra/canopus_train/spec_files/ \
--seed 1 \
--gpus 1 \
--split-file data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv \
--batch-size 128 \
--max-epochs 600 \
--patience 20 \
--loss-fn cosine \
--cpus-per-trial 6 \
--gpus-per-trial 1 \
--num-h-samples 100 \
--max-concurrent 10 \
--num-workers 6 \
--persistent-workers \
--save-dir results/hyperopt_xformer

#--embed-instrument \
