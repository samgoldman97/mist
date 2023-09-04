#!/bin/bash
#SBATCH -n 1           # 1 core
#SBATCH -t 0-05:00:00   # 5 hours
#SBATCH -J ms # sensible name for the job
#SBATCH --output=logs/ms_run_%j.log   # Standard output and error log
#SBATCH -p sched_mit_ccoley
#SBATCH --mem-per-cpu=20000 # 10 gb
##SBATCH --mem=20000 # 20 gb

##SBATCH -w node1236
##SBATCH --gres=gpu:1 #1 gpu
##SBATCH --mem=20000  # 20 gb 
##SBATCH -p {Partition Name} # Partition with GPUs

# Use this to run generic scripts:
# sbatch --export=CMD="python my_python_script --my-arg" src/scripts/slurm_scripts/generic_slurm.sh

# Import module --> replace with individual username 
#source /etc/profile 
#source /home/samlg/.bashrc

# Activate conda
# source {path}/miniconda3/etc/profile.d/conda.sh

# Activate right python version
# conda activate {conda_env}
conda activate ms-gen

echo "Cuda visible:"
echo $CUDA_VISIBLE_DEVICES

# Evaluate the passed in command... in this case, it should be python
eval $CMD
