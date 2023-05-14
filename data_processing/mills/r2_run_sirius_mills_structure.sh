#!/bin/bash
#SBATCH -n 30           # 1 core
#SBATCH -t 0-10:00:00   # 10 hours
#SBATCH -J sirius # sensible name for the job
#SBATCH --output=logs/sirius_%j.log   # Standard output and error log
#SBATCH -p sched_mit_ccoley
#SBATCH --mem-per-cpu=20G # 10 gb
#SBATCH -w node1237

##SBATCH -w node1238
##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Import module
#source /etc/profile 
#source /home/samlg/.bashrc

SIRIUS=sirius5/sirius/bin/sirius
CORES=32

# Assumes we know the correct chemical formula
echo "Processing mills" 

#sirius_processing/sirius/bin/sirius --cores 30 --output data/paired_spectra/broad/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/broad/spec_files/ formula --ppm-max-ms2 30

#sirius --cores 4 --output data/paired_spectra/broad/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/broad/spec_files/ formula --ppm-max-ms2 30
mkdir data/paired_spectra/mills
mkdir data/paired_spectra/mills/sirius_outputs
mkdir data/paired_spectra/mills/sirius_outputs_structure

SUMMARY_DIR=data/paired_spectra/mills/sirius_outputs_structure_summary
mkdir $SUMMARY_DIR
password=$SIRIUS_PW
INPUT_FILE=data/raw/mills/Mills_mzxml/mgf_export_sirius.mgf
INPUT_FILE=data/raw/mills/Mills_mzxml/mgf_export_sirius_filtered_500.mgf

# Ignore formula
#--naming-convention %filename \
$SIRIUS login -u samlg@mit.edu -p
$SIRIUS  \
    --cores $CORES \
    --output  data/paired_spectra/mills/sirius_outputs_structure/  \
    --input $INPUT_FILE \
    formula  \
    --ppm-max-ms2 10  \
    fingerprint \
    structure \
    --db hmdb \
    write-summaries \
    --output $SUMMARY_DIR \


    #--db hmdb \
