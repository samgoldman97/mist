"""FP Predictions batched for repo"""
import subprocess
from collections import defaultdict
from pathlib import Path
import shutil

dirs = [
    "results/2022_11_01_canopus_data_ablations",
    "results/2022_11_01_canopus_model_ablations",
]

dir_names = ["ffn", "data", "model"]

# Step 1: Make fp predictions for entire directory
num_workers = 16
dataset_name = "canopus_train"
base_script = f"python3 run_scripts/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --output-targs --subset-datasets test_only"
for dir_ in dirs:
    for ckpt in Path(dir_).rglob("*.ckpt"):
        cmd = f"{base_script} --model-ckpt {str(ckpt)} --save-dir {ckpt.parent.parent / 'preds'}"
        subprocess.call(cmd, shell=True)

# Make summaries
#cmd = f"python3 analysis/fp_ablations/summarize_data_ablation.py --dir {dirs[0]}"
#subprocess.call(cmd, shell=True)

cmd = f"python3 analysis/fp_ablations/summarize_model_ablation.py --dir {dirs[1]}"
subprocess.call(cmd, shell=True)

# Plot data
#for metric in ["cos_sim", "ll"]:
#    data_ablation_plot = f"python3 analysis/fp_ablations/data_ablation_plot.py --ablation-file {Path(dirs[0]) / 'fp_pred_summary.tsv'} --metric {metric}"
#    subprocess.call(data_ablation_plot, shell=True)
    
for metric in ["cos_sim", "ll"]:
    data_ablation_plot = f"python3 analysis/fp_ablations/model_ablation_plot.py --ablation-file {Path(dirs[1]) / 'fp_pred_summary.tsv'} --metric {metric}"
    subprocess.call(data_ablation_plot, shell=True)
