""" eval fp models """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

res_dirs = [
    "results/canopus_fp_ffn",
    "results/canopus_fp_mist/",
]

num_workers = 16
dataset_name = "canopus_train"
labels = "data/paired_spectra/canopus_train/labels.tsv"
subform = "data/paired_spectra/canopus_train/subformulae/default_subformulae/"
spec = "data/paired_spectra/canopus_train/spec_files/"
devices = ",".join(["1"])
base_script = f"python3 src/mist/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --output-targs --subset-datasets test_only --labels-file {labels} --spec-folder {spec} --subform-folder {subform}" 

for dir_ in res_dirs:
    for ckpt in Path(dir_).rglob("*.ckpt"):
        fold_name = ckpt.parent.name
        savedir = ckpt.parent / "preds"
        cmd = f"{base_script} --model-ckpt {str(ckpt)} --save-dir {savedir}"
        subprocess.call(cmd, shell=True)
