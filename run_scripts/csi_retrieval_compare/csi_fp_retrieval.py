""" Conduct fp-based retrieval for the mist model"""
import pickle
from pathlib import Path
import subprocess

model_inputs = [
    {
        "res_dir": "data/paired_spectra/csi2022/prev_results_csi/",
        "save_dir": "results/csi_retrieval_compare/sirius_fp_retrieval",
    },
]

num_workers = 16
dataset_name = "csi2022"

labels_file = f"data/paired_spectra/{dataset_name}/labels.tsv"
subform = f"data/paired_spectra/{dataset_name}/subformulae/subformulae_default/"
spec = f"data/paired_spectra/{dataset_name}/spec_files/"

devices = ",".join(["1"])
cuda_vis_str = f"CUDA_VISIBLE_DEVICES={devices}"

fp_retrieval_base = f"{cuda_vis_str} python src/mist/retrieval_fp.py --dist-name cosine --num-workers {num_workers} --labels-file {labels_file} --top-k 200"

for test_dict in model_inputs:
    res_dir = Path(test_dict["res_dir"])
    save_dir = Path(test_dict["save_dir"])
    fp_files = list(res_dir.rglob("*.p"))
    for pred_file in fp_files:
        # Load pickle file
        with open(pred_file, "rb") as f:
            pred_dict = pickle.load(f)
        split = pred_dict["split_name"]

        save_dir_temp = save_dir / split
        save_dir_temp.mkdir(exist_ok=True, parents=True)

        # Run retrieval
        cmd = (
            f"{fp_retrieval_base} --fp-pred-file {pred_file} --save-dir {save_dir_temp}"
        )
        print(cmd)
        subprocess.call(cmd, shell=True)
