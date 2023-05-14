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
source /etc/profile 
source /home/samlg/.bashrc


summarize_script=src/mist/sirius/summarize_sirius.py

# Assumes we know the correct chemical formula
# Use mass diff 30

# Already run
#echo "Processing casmi"
#sirius_processing/sirius/bin/sirius --cores 30 --output  data/paired_spectra/casmi/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/casmi/spec_files/ formula --ppm-max-ms2 30
#echo "Processing csi2022_debug"
#sirius_processing/sirius/bin/sirius --cores 30 --output  data/paired_spectra/csi2022_debug/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/csi2022_debug/spec_files/ formula --ppm-max-ms2 30
echo "Processing csi2022"
sirius_processing/sirius/bin/sirius --cores 30 --output  data/paired_spectra/csi2022/sirius_outputs/ --naming-convention %filename --input data/paired_spectra/csi2022/spec_files/ formula --ppm-max-ms2 30

conda activate ms-gen
echo "Summarizing casmi"
python3 $summarize_script  --labels-file  data/paired_spectra/casmi/labels.tsv --sirius-folder  data/paired_spectra/casmi/sirius_outputs

echo "Summarizing csi2022_debug"
python3 $summarize_script  --labels-file  data/paired_spectra/csi2022_debug/labels.tsv --sirius-folder  data/paired_spectra/csi2022_debug/sirius_outputs
echo "Summarizing csi2022"
python3 $summarize_script  --labels-file  data/paired_spectra/csi2022/labels.tsv --sirius-folder  data/paired_spectra/csi2022/sirius_outputs
