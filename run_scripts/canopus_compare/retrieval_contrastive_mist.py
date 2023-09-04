""" Conduct contrastive retrieval for the mist model"""
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

model_inputs = [
    {
        "res_dir": "results/canopus_contrastive_mist/",
        "save_dir": "results/canopus_retrieval_compare/mist_contrastive_retrieval/",
    },
]

num_workers = 16
dataset_name = "canopus_train"

labels_file = f"data/paired_spectra/{dataset_name}/labels.tsv"
subform_folder = f"data/paired_spectra/{dataset_name}/subformulae/subformulae_default/"
devices = ",".join(["1"])
cuda_vis_str = f"CUDA_VISIBLE_DEVICES={devices}"

contrastive_retrieval_base = f"""
{cuda_vis_str} python3 src/mist/retrieval_contrast.py \\
    --dataset-name {dataset_name}  \\
    --hdf-file data/paired_spectra/{dataset_name}/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db.h5 \\
    --labels-file {labels_file} \\
    --subset-datasets test_only \\
    --subform-folder {subform_folder} \\
    --num-workers {num_workers} \\
    --dist-name cosine \\
    --gpu \\
    --top-k 256"""


for test_dict in model_inputs:

    res_dir = Path(test_dict["res_dir"])
    save_dir = Path(test_dict["save_dir"])
    ckpts = list(res_dir.rglob("best.ckpt"))
    joined_ckpts = "\n".join([str(i) for i in ckpts])
    print(f"Checkpoints runing: {joined_ckpts}")
    for ckpt in ckpts:

        # Predict fingerprints
        fold_name = ckpt.parent.name
        save_dir_temp = save_dir / f"{fold_name}"
        save_dir_temp.mkdir(exist_ok=True, parents=True)

        cmd = f"""{contrastive_retrieval_base} \\
        --model-ckpt {str(ckpt)} \\
        --save-dir {save_dir_temp} """
        print(cmd)
        subprocess.call(cmd, shell=True)
