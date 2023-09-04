CUDA_VISIBLE_DEVICES=1,2,3 python3 src/mist/hyperopt_mist.py \
--cache-featurizers \
--fp-names morgan4096 \
--seed 1 \
--gpus 1 \
--split-file data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv \
--labels-file data/paired_spectra/canopus_train/labels.tsv \
--spec-folder data/paired_spectra/canopus_train/spec_files/ \
--subform-folder data/paired_spectra/canopus_train/subformulae/default_subformulae/ \
--magma-folder data/paired_spectra/canopus_train/magma_outputs/magma_tsv/ \
--batch-size 128 \
--max-epochs 600 \
--pairwise-featurization \
--set-pooling cls \
--cls-type ms1 \
--patience 20 \
--magma-aux-loss  \
--magma-modulo 512 \
--iterative-preds growing \
--loss-fn cosine \
--cpus-per-trial 8 \
--gpus-per-trial 1 \
--num-h-samples 100 \
--max-concurrent 10 \
--num-workers 8 \
--augment-prob 0.5 \
--inten-prob 0.12 \
--remove-prob 0.5 \
--remove-weights exp \
--no-diffs  \
--save-dir results/hyperopt_mist/  


# add to also include  forward augmentation
#--forward-labels []\ 
#--forward-aug-folder filename \
#--frac-orig 0.7 \
