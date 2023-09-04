""" Make predictions for all models that were trained """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

res_dirs = [
    "results/csi_fp_mist/",
    "results/csi_fp_ffn/",
    "results/csi_fp_xformer/",
]

num_workers = 16
dataset_name = "csi2022"
labels = "data/paired_spectra/csi2022/labels.tsv"
subform = f"data/paired_spectra/{dataset_name}/subformulae/subformulae_default/"
spec = f"data/paired_spectra/{dataset_name}/spec_files/"
devices = ",".join(["1"])
base_script = f"python3 src/mist/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --output-targs --subset-datasets test_only --labels-file {labels} --spec-folder {spec} --subform-folder {subform}" 

for dir_ in res_dirs:
    for ckpt in Path(dir_).rglob("*.ckpt"):
        fold_name = ckpt.parent.name
        savedir = ckpt.parent / "preds"
        cmd = f"{base_script} --model-ckpt {str(ckpt)} --save-dir {savedir}"
        subprocess.call(cmd, shell=True)
