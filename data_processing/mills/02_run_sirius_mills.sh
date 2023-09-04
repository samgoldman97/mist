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

SIRIUS=sirius/bin/sirius
CORES=32

# Assumes we know the correct chemical formula
echo "Processing mills" 

#sirius_processing/sirius/bin/sirius --cores 30 --output data/paired_spectra/broad/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/broad/spec_files/ formula --ppm-max-ms2 30

#sirius --cores 4 --output data/paired_spectra/broad/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/broad/spec_files/ formula --ppm-max-ms2 30
mkdir data/paired_spectra/mills
mkdir data/paired_spectra/mills/sirius_outputs

# Ignore formula
$SIRIUS --cores $CORES --output data/paired_spectra/mills/sirius_outputs_2/  --input data/raw/mills/Mills_mzxml/mgf_export_sirius.mgf --ignore-formula formula --ppm-max-ms2 10  #--naming-convention %filename
