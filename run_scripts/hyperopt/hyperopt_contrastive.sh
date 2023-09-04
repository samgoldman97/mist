CUDA_VISIBLE_DEVICES=1,2,3 python3 src/mist/hyperopt_contrastive.py \
--cache-featurizers \
--hdf-file data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_contrast.h5 \
--split-file data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv \
--labels-file data/paired_spectra/canopus_train/labels.tsv \
--spec-folder data/paired_spectra/canopus_train/spec_files/ \
--subform-folder data/paired_spectra/canopus_train/subformulae/default_subformulae/ \
--magma-folder data/paired_spectra/canopus_train/magma_outputs/magma_tsv/ \
--seed 1 \
--gpus 1 \
--split-file data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv \
--batch-size 128 \
--max-epochs 600 \
--patience 10 \
--cpus-per-trial 8 \
--gpus-per-trial 0.5 \
--num-h-samples 100 \
--max-concurrent 10 \
--num-workers 8 \
--augment-prob 0.5 \
--inten-prob 0.12 \
--remove-prob 0.1 \
--remove-weights exp \
--save-dir results/hyperopt_contrastive/ \
--dist-name cosine \
--compound-lib intpubchem \
--contrastive-loss nce \
--contrastive-decoy-pool mean \
--contrastive-latent h0 \
--contrastive-scale 10 \
--contrastive-bias 0.1 \
--num-decoys 64 \
--max-db-decoys 256 \
--decoy-norm-exp 4 \
--negative-strategy hardisomer_tani_pickled \
--ckpt-file  results/canopus_fp_mist/split_0/canopus_hplus_100_0/best.ckpt


# add to also include  forward augmentation
#--forward-labels []\ 
#--forward-aug-folder filename \
#--frac-orig 0.7 \
