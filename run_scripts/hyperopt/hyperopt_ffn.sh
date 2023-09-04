CUDA_VISIBLE_DEVICES=1,2,3 python3 src/mist/hyperopt_ffn_binned.py \
--cache-featurizers \
--labels-file data/paired_spectra/canopus_train/labels.tsv \
--spec-folder data/paired_spectra/canopus_train/spec_files/ \
--fp-names morgan4096 \
--seed 1 \
--gpus 1 \
--split-file data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv \
--max-epochs 600 \
--patience 20 \
--batch-size 64 \
--loss-fn cosine \
--cpus-per-trial 6 \
--gpus-per-trial 1 \
--num-h-samples 100 \
--max-concurrent 10 \
--num-workers 6 \
--persistent-workers \
--save-dir results/hyperopt_ffn_binned


#--embed-instrument \
