# 🌫️ MIST: Metabolite Inference with Spectrum Transformers
[![DOI](https://zenodo.org/badge/564051299.svg)](https://zenodo.org/badge/latestdoi/564051299)  

This repository provides implementations and code examples for [Metabolite Inference with Spectrum Transformers (MIST)](https://www.nature.com/articles/s42256-023-00708-3). MIST models can be used to predict molecular fingerprints from tandem mass spectrometry data and, when trained in a contrastive learning framework, enable embedding and structure annotation by database lookup. Rather than directly embed binned spectra, MIST applies a transformer architecture to directly encode and learn to represent collections of chemical formula.  MIST has also since been extended to predict precursor chemical formulae as [MIST-CF](https://github.com/samgoldman97/mist-cf).

_Samuel Goldman, Jeremy Wohlwend, Martin Strazar, Guy Haroush, Ramnik J. Xavier, Connor W. Coley_


__Update__: This branch provides an updated version of the MIST method for increased usability and developability. See the [change log](#changelog) for specific details.

![Model graphic](MIST_graphic.png)


## Table of Contents

1. [Install & setup](#setup)      
2. [Quick start](#quickstart)   
3. [Data](#data)   
4. [Training models](#training)     
5. [Experiments](#paper)    
6. [Change log](#changelog)   
7. [Citations](#citations)     


## Install & setup <a name="setup"></a>

After git cloning the repository, the environment and package can be installed.  Please note that the environment downloaded attempts to utilize cuda11.1. Please comment this line out in environment.yml if you do not plan to use gpu support prior to the commands below. We strongly recommend replacing conda with [mamba](https://mamba.readthedocs.io/en/latest/installation.html) for fast install (e.g., `mamba env create -f environment.yml`).

```
conda env create -f environment.yml
conda activate ms-gen
pip install -r requirements.txt
python setup.py develop
```

This environment was tested on Ubuntu 20.04.1 with CUDA Version 11.4 . It takes roughly 10 minutes to install using Mamba. 


## Quick start <a name="quickstart"></a>

After creating a python environment, pretrained models can be used to: 

1. Predict fingerprints from spectra  (`quickstart/model_predictions/fp_preds/`)  
2. Annotate spectra by ranking candidates in a reference smiles list (`quickstart/model_predictions/retrieval/`)  
3. Embed spectra into a dense continuous space  (`quickstart/model_predictions/contrastive_embed/`)   

To showcase these capabilities, we include an MGF file, `quickstart/quickstart.mgf` (a sample from the Mills et al. data), along with a set of sample smiles `quickstart/lookup_smiles.txt`. 

```

conda activate ms-gen
. quickstart/00_download_models.sh
. quickstart/01_run_models.sh

```

Output predictions can be found in `quickstart/model_predictions` and are included by default with the repository. We provide an additional notebook `notebooks/mist_demo.ipynb` that shows these calls programmatically, rather than in the command line.

## Data <a name="data"></a>

Training models requires the use of paired mass spectra data and unpaired libraries of molecules as annotation candidates. 


### Downloading and preparing paired datasets

We utilize two datasets to train models: 

1. **csi2022**: H+ Spectra from GNPS, NIST, MONA, and others kindly provided by Kai Duhrkop from the SIRIUS and CSI:FingerID team. This dataset is used to complete most benchmarking done.  
2. **canopus\_train**: Public data extracted from GNPS and prepared by the 2021 CANOPUS methods paper. This has since been renamed "NPLIB1" in our subsequent papers.  

Each paired spectra dataset will have the following standardized folders and components, living under a single dataset folder:  
 
1. **labels.tsv**: A file containing the columns ["dataset", "spec", "name", "ionization", "formula", "smiles", "inchikey", "instrument"], where "smiles" coreresponds to an *achiral* version of the smiles string.      
2. **spec\_files**: A directory containing each .ms file in the dataset    
3. **subformulae**: Outputs of a subformula labeling program run on the corresponding .ms directory     
4. **magma_outputs**: Outputs of a MAGMa program run on the corresponding spec files directory     
5. **splits**: Splits contains all splits. These are in the form of a table with 2 columns including split name and category (train, val, test, or exclude)   
6. **retrieval\_hdf**: Folder to hold hdf files used for retrieval and contrastive model training. Note we construct these with relevant isomers for the dataset.      
7. [optional] **prev_results**: Folder to hold any previous results on the dataset if benchmarked by another author    
8. [optional] **data augmentation**: Another part of model training is the use of simulated spectra from a forward model.  After training these different forward models, we store relevant predictions inside spearate folders here.  

We are not able to redistribute the CSI2022 dataset. The `canopus_train` dataset (including split changes) can be downloaded and prepared for minimal model execution:

```

. data_processing/canopus_train/00_download_canopus_data.sh

``` 


We intentionally do not include the retrieval HDF file in the data download, as the retrieval file is larger (>5GB). This can be re-made by following the instructions below to process PubChem (or one of the other unpaired libraries), then running `python data_processing/canopus_train/03_retrieval_hdf.py`. The full data processing pipeline used to prepare relevant files can be found in `data_processing/canopus_train/` (i.e., subformulae assignment, magma execution, retrieval and contrastive dataframe construction, subsetting of smiles to be used in augmentation, and assigning subformuale to the augmented mgf provided).


### Unpaired molecules

We consider processing three example datasets to be used as unpaired molecules: _biomols_, a dataset of biologicaly-relevant molecules prepared by Duhrkop et al. for the CANOPUS manuscript, _hmdb_, the Human Metabolome Database, and _pubchem_, the most complete dataset of molecules. Instructions for downloading and processing each of these can be found in `data_processing/mol_libraries/`. 

MIST uses these databases of molecules (without spectra) in two ways: 

1. _Data augmentation_: To train our models, we utilize an auxiliary forward molecule-to-spectrum model to add training examples to the dataset. The primary requirements are that these augmented spectra are provided as a labels file and an mgf file. We provide an example of this in the `data/paired_spectra/canopus_train/aug_iceberg_canopus_train/`. See the [ms-pred github repository](https://github.com/samgoldman97/ms-pred) for details on training a model and exporting an mgf. See `data_processing/canopus_train/04_subset_smis.sh` for how we subsetted the biomolecules dataset to create labels for the ms-pred prediction and `data_processing/canopus_train/05_buid_aug_mgf.sh` for how we process the resulting mgf into subformulae assignments after export.  
2. _Retrieval libraries_: A second use for these libraries is to build retrieval databases or as contrastive decoys. See `data_processing/canopus_train/03_retrieval_hdf.py` for call signatures to construct both of these, after creating a mapping of chem formula to smiles (e.g., `data_processing/mol_libraries/pubchem/02_make_formula_subsets.sh`).  


## Training models <a name="training"></a>

After downloading the canopus\_train dataset, the following two commands demonstrate how to train models that can be used (as illustrated in the quickstart). **The config files specify the exact parameters used in experiments as reported in the paper.**


**MIST Fingerprint model**: 

```

CUDA_VISIBLE_DEVICES=0 python src/mist/train_mist.py \
    --cache-featurizers \
    --labels-file 'data/paired_spectra/canopus_train/labels.tsv' \
    --subform-folder 'data/paired_spectra/canopus_train/subformulae/subformulae_default/' \
    --spec-folder 'data/paired_spectra/canopus_train/spec_files/' \
    --magma-folder 'data/paired_spectra/canopus_train/magma_outputs/magma_tsv/' \
    --fp-names morgan4096 \
    --num-workers 16 \
    --seed 1 \
    --gpus 1 \
    --augment-data \
    --batch-size 128 \
    --iterative-preds 'growing' \
    --iterative-loss-weight 0.4 \
    --learning-rate 0.00077 \
    --weight-decay 1e-07 \
    --lr-decay-frac 0.9 \
    --hidden-size 256 \
    --pairwise-featurization \
    --peak-attn-layers 2 \
    --refine-layers 4 \
    --spectra-dropout 0.1 \
    --magma-aux-loss \
    --magma-loss-lambda 8 \
    --magma-modulo 512 \
    --split-file 'data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv' \
    --forward-labels 'data/paired_spectra/canopus_train/aug_iceberg_canopus_train/biomols_filtered_smiles_canopus_train_labels.tsv' \
    --forward-aug-folder 'data/paired_spectra/canopus_train/aug_iceberg_canopus_train/canopus_hplus_100_0/subforms/' \
    --frac-orig 0.6 \
    --form-embedder 'pos-cos' \
    --no-diffs \
    --save-dir results/canopus_fp_mist/split_0

``` 

**Contrastive model**: 

```

CUDA_VISIBLE_DEVICES=0 python src/mist/train_contrastive.py \
    --seed 1 \
    --labels-file 'data/paired_spectra/canopus_train/labels.tsv' \
    --subform-folder 'data/paired_spectra/canopus_train/subformulae/subformulae_default/' \
    --spec-folder 'data/paired_spectra/canopus_train/spec_files/' \
    --magma-folder 'data/paired_spectra/canopus_train/magma_outputs/' \
    --hdf-file 'data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_contrast.h5' \
    --augment-data \
    --contrastive-weight 0.6 \
    --contrastive-scale 16 \
    --num-decoys 64 \
    --max-db-decoys 256 \
    --decoy-norm-exp 4 \
    --negative-strategy 'hardisomer_tani_pickled' \
    --dist-name 'cosine' \
    --learning-rate 0.00057 \
    --weight-decay 1e-07 \
    --scheduler \
    --lr-decay-frac 0.7138 \
    --patience 10 \
    --gpus 1 \
    --batch-size 32 \
    --num-workers 8 \
    --cache-featurizers \
    --ckpt-file 'results/canopus_fp_mist/split_0/canopus_hplus_100_0/best.ckpt' \
    --split-file 'data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv' \
    --forward-labels 'data/paired_spectra/canopus_train/aug_iceberg_canopus_train/biomols_filtered_smiles_canopus_train_labels.tsv' \
    --forward-aug-folder 'data/paired_spectra/canopus_train/aug_iceberg_canopus_train/canopus_hplus_100_0/subforms/' \
    --frac-orig 0.2 \
    --save-dir results/canopus_contrastive_mist/split_0


``` 


## Experiments <a name="paper"></a>   

We detail our pipeline for executing updated experiments below. Because the comparisons on the CSI dataset require proprietary data, some will not be runnable. The execution and scripts are included here to help illustrate the logic. Results are precomputed and shown in the analysis notebooks (`notebooks/`).


### Dataset analysis

We provide summary statistics and chemical classifications of the CANOPUS (NPLIB1) dataset and combined dataset in `notebooks/dataset_analysis.ipynb`. The chemical classes are assigned using NPClassifier, which is run via the GNPS endpoint. This is accessed in `run_scripts/dataset_analysis/chem_classify.py`.

### Hyperparameter optimization

Hyperparameters were previously optimized using Ray Tune and Optuna as described in the released paper. We use a variation of these parameters by default, but provide additional scripts demonstrating the workflow for how to tune parameters. See `run_scripts/hyperopt/`.

### CSI fingerprint comparison

We compare four models using the partially proprietary CSI2022 dataset that includes NIST. These models are a feed forward network (FFN), Sinusoidal Transformer, MIST, and CSI:FingerID (as provided by the authors). Configurations for these models can be found and edited in `configs/csi_compare`. The models themselves can be trained by running the following scripts:

1. `. run_scripts/csi_fp_compare/train_ffn.sh`   
2. `. run_scripts/csi_fp_compare/train_xformer.sh`   
3. `. run_scripts/csi_fp_compare/train_mist.sh`   

After training models, predictions can be made with `python run_scripts/csi_fp_compare/eval_models.py`. Results are analyzed generating partial paper figures in `notebooks/fp_csi_compare.ipynb`

To compare against this in future iterations, we recommend comparing against the following splits: 

1. `data/paired_spectra/canopus_train/splits/canopus_hplus_100_0.tsv`    
2. `data/paired_spectra/canopus_train/splits/canopus_hplus_100_1.tsv`  
3. `data/paired_spectra/canopus_train/splits/canopus_hplus_100_2.tsv`  


### CSI retrieval analysis
After training fingerprint models, a single contrastive model can be trained on top of the MIST fingerprint model. 

```
. run_scripts/csi_retrieval_compare/train_contrastive.sh
```

With both trained fingerprint models and contrastive models, retrieval can be executed and evaluated:

```

python run_scripts/csi_retrieval_compare/mist_fp_retrieval.py
python run_scripts/csi_retrieval_compare/mist_contrastive_retrieval.py

# Conduct equivalent retrieval with SIRIUS/CSI:FingerID exported fingerprints
python run_scripts/csi_retrieval_compare/csi_fp_retrieval.py
python run_scripts/csi_retrieval_compare/analyze_results.py

```

Results of retrieval are subsequently be analyzed in `notebooks/retrieval_csi_compare.ipynb`


### CSI distance analysis

We also provide code for conducting a latent space retrieval analysis on the contrastive mist model.

```

# Download MS2DeepScore and spec2vec
. run_scripts/csi_embed_compare/download_baselines.sh

# Run baselines. Note that the cosine similarity calculation takes quite long
python run_scripts/csi_embed_compare/dist_baselines.py

# Run mist
python run_scripts/csi_embed_compare/dist_mist.py 

```

These can be inspected in `notebooks/embed_csi_compare.ipynb`


### Public data fingerprint comparison

We include code to compare the FFN and MIST models on the public dataset. Config files can be found in `configs/canopus_compare/`. These can be run with the following scripts: 

1. `. run_scripts/canopus_compare/train_fp_ffn.sh`
2. `. run_scripts/canopus_compare/train_fp_mist.sh`


After training these models, they can be evaluated using:

```
python run_scripts/canopus_compare/eval_fp_models.py
```

Results of predictions are subsequently be analyzed in `notebooks/fp_canopus_compare.ipynb`.


### Public data retrieval analysis

We use two types of models for public retrieval: MIST FP and MIST contrastive models. The contrastive model requires an HDF5 set of decoys that can be made using `data_processing/canopus_train/03_retrieval_hdf.py`. This creates both the retrieval database and also the contrastive training database. With this in hand, a contrastive model can be trained on top of the fingerprint model

```

. run_scripts/canopus_compare/train_contrastive_mist.sh

```

With both trained fingerprint models and contrastive models, retrieval can be executed and evaluated:

```
python run_scripts/canopus_compare/retrieval_fp.py
python run_scripts/canopus_compare/retrieval_contrastive_mist.py
python run_scripts/canopus_compare/eval_retrieval.py
```

Results are subsequently analyzed and inspected in `notebooks/retrieval_canopus_compare.ipynb`


## Change log <a name="changelog"></a>

We detail changes since the published MIST manuscript.

**Model changes**    
1. **Internal subform labeling**: We reduce the reliance on SIRIUS by implementing our own subformula labeling module, greatly simplifying the MIST workflow.
2. **Formula embeddings**: We previously utilized floating point value formula embeddings. We have added support for sinusoidal embeddings of formula count integers, as utilized in our subsequent model MIST-CF.   
3. **More element options**: We have added additional potential elements to the model. 
4. **Adam optimizer**: We have switched from RAdam to Adam to reduce external dependencies.
5. **External forward augmentation**: A key component of MIST is that we trained on augmented, predicted spectra. This module was previously included _within_ the MIST code repository. We have since developed an external repository, [MS-pred](https://github.com/samgoldman97/ms-pred). As a result, we now only include scripts to process a labels file and MGF file into subformula assignments (i.e., for a additional supervision to MIST) with the expectation that forward models can be trained and used for prediction in a separate repository. In the trained models and examples, external augmentation is conducted using the ICEBERG model (trained on same train/test splits). It is still imperative to exclude the test set molecules from the augmentation set.
6. **Model simplifications**: We simplify the MIST architecture as in MIST-CF so that all inputs to the transformer are concatenated once, passed through the transformer, pooled, and run through an MLP. This is different from the original architecture that included multiple layers of MLPs prior to the transformer. 
7. **Model covariates**: We have added optionality to encode instrument types, multiple adduct types, and also the differences between each subformula peak into the model. It would be straightforward to train a new model on a dataset that includes adducts outside of H+.   
8. **Minor parameter changes**: Minor parameters have been changed throughout (e.g., the number of augmented spectra from 300k to 200k). Pelase refer to config files as the gold standard for how experiments were conducted. 
9. **MAGMa labeling**: The implementation for MAGMa has been simplified and now fingerprinting is conducted with an internal re-implementation of the circular fingerprinting algorithm `src/mist/magma/frag_fp.py`. 


**Analysis changes**    
1. **Worst case retrieval**: Previously, we optimistically reported retrieval. That is, if 2 compounds tied in distance, we reported the lower of the two. Herein, we have switched analysis to the worst case (i.e,. the higher/worse ranking of the tied compounds). This conservatively underestimates performance in line with analysis we have done in the forward modeling direction (mol-to-spec).   
2. **Jupyter notebooks**: Analysis is now conducted in Jupyter notebooks rather than scripts.   


**Organizational changes**    
1. **Better file structure**: The file structure has been re-oriented for simplicity.   
2. **No ensembles**: In our published work, we report accuracy in terms of ensembles in main text figures. While these do increase performance, they create added code complexity and limit future development. We have thus removed them to make the code easier and more tractable for others to work with.      
3. **Easier quickstart**: The quickstart demo has been reduced to 2 scripts (model download and execution).  
4. **Split formats**: The splits have all been reformatted as TSV files instead of .txt and .csv extensions.  
5. **Prospective removal**: We have removed the prospective analysis from this code. It is still accessible in the original paper branch.   
6. **HDF Simplifications**: HDF Files have been simplified to reduce the number of necessary additional files for model training. Fingerprints have been stored as packed bit vectors


## Citations <a name="citations"></a>  

We ask users to cite [MIST](https://www.nature.com/articles/s42256-023-00708-3) directly by referencing the following paper:

Goldman, S., Wohlwend, J., Stražar, M. et al. Annotating metabolite mass spectra with domain-inspired chemical formula transformers. Nat Mach Intell (2023). https://doi.org/10.1038/s42256-023-00708-3   

MIST also builds on a number of other projects, ideas, and software including SIRIUS, MAGMa substructure labeling, the canopus\_train data, the Mills et al. IBD data, NPClassifier to classify compounds, PubChem as a retrieval library, and HMDB as a retrieval library. Please consider citing the following full list of papers when relevant:  
 
1. Kai Dührkop, Markus Fleischauer, Marcus Ludwig, Alexander A. Aksenov, Alexey V. Melnik, Marvin Meusel, Pieter C. Dorrestein, Juho Rousu, and Sebastian Böcker, SIRIUS 4: Turning tandem mass spectra into metabolite structure information. Nature Methods 16, 299–302, 2019.   
2. Ridder, Lars, Justin JJ van der Hooft, and Stefan Verhoeven. "Automatic compound annotation from mass spectrometry data using MAGMa." Mass Spectrometry 3.Special_Issue_2 (2014): S0033-S0033.    
3. Wang, Mingxun, et al. "Sharing and community curation of mass spectrometry data with Global Natural Products Social Molecular Networking." Nature biotechnology 34.8 (2016): 828-837.    
4. Dührkop, Kai, et al. "Systematic classification of unknown metabolites using high-resolution fragmentation mass spectra." Nature Biotechnology 39.4 (2021): 462-471.   
5. Mills, Robert H., et al. "Multi-omics analyses of the ulcerative colitis gut microbiome link Bacteroides vulgatus proteases with disease severity." Nature Microbiology 7.2 (2022): 262-276.   
6. Kim, Hyun Woo, et al. "NPClassifier: a deep neural network-based structural classification tool for natural products." Journal of natural products 84.11 (2021): 2795-2807.   
7. Kim, Sunghwan, et al. "PubChem 2019 update: improved access to chemical data." Nucleic acids research 47.D1 (2019): D1102-D1109.    
8. Wishart, David S., et al. "HMDB 5.0: the human metabolome database for 2022." Nucleic Acids Research 50.D1 (2022): D622-D631.
