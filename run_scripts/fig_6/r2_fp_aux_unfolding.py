"""Compare the auxilary unfolding model"""
import subprocess
from collections import defaultdict
from pathlib import Path
import shutil

dirs = [
    "results/2023_05_03_top_layers"
]


# Step 1: Make fp predictions for entire directory
num_workers = 16
dataset_name = "canopus_train"
base_script = f"python3 run_scripts/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --output-targs --subset-datasets test_only"
for dir_ in dirs:
    for ckpt in Path(dir_).rglob("*.ckpt"):
        cmd = f"{base_script} --model-ckpt {str(ckpt)} --save-dir {ckpt.parent.parent / 'preds'}"
        #subprocess.call(cmd, shell=True)


cmd = f"python3 analysis/fp_ablations/summarize_model_ablation.py --dir {dirs[0]}"
subprocess.call(cmd, shell=True)
