#!/bin/bash
#SBATCH -N 1 # 1 node
#SBATCH -n 30           # 1 core
#SBATCH -t 2-00:00:00   # 2 days
#SBATCH -J sirius # sensible name for the job
#SBATCH --output=logs/sirius_%j.log   # Standard output and error log
#SBATCH -p sched_mit_ccoley
#SBATCH --mem-per-cpu=12G # 10 gb
#SBATCH -w node1237

##SBATCH -w node1238
##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Import module
source /home/samlg/.bashrc
SIRIUS=sirius/bin/sirius

conda activate ms-gen

# Assumes we know the correct chemical formula
# Use mass diff 30
#echo "Processing csi2022"
#$SIRIUS --cores 30 --output  data/paired_spectra/canopus_train/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/canopus_train/spec_files/ formula --ppm-max-ms2 30

echo "Summarizing canopus train"
python3 src/mist/sirius/summarize_sirius.py --labels-file  data/paired_spectra/canopus_train/labels.tsv --sirius-folder  data/paired_spectra/canopus_train/sirius_outputs
