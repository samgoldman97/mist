#!/bin/bash
#SBATCH -n 30           # 1 core
#SBATCH -t 2-00:00:00   # 2 days
#SBATCH -J sirius # sensible name for the job
#SBATCH --output=logs/sirius_%j.log   # Standard output and error log
#SBATCH -p sched_mit_ccoley
#SBATCH --mem-per-cpu=7G # 7 GB
#SBATCH -w node1238

##SBATCH -w node1238
##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Import module
source /etc/profile 
source /home/samlg/.bashrc

conda activate ms-gen

magma_file=src/mist/magma/run_magma.py

echo "Magma on casmi"
python3 $magma_file --spectra-dir data/paired_spectra/casmi/sirius_outputs  --output-dir data/paired_spectra/casmi/magma_outputs  --spec-labels data/paired_spectra/casmi/labels.tsv

echo "Magma on csi2022_debug"
python3 $magma_file  --spectra-dir data/paired_spectra/csi2022_debug/sirius_outputs  --output-dir data/paired_spectra/csi2022_debug/magma_outputs  --spec-labels data/paired_spectra/csi2022_debug/labels.tsv

echo "Magma on csi2022"
python3 $magma_file --spectra-dir data/paired_spectra/csi2022/sirius_outputs  --output-dir data/paired_spectra/csi2022/magma_outputs  --spec-labels data/paired_spectra/csi2022/labels.tsv
