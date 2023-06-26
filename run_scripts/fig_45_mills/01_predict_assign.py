""" predict_assign.py

Use during training to make predictions for forward model

"""

import yaml
import subprocess
from pathlib import Path

# Step 1: Make forward predictions

res_folder = Path("results/2022_10_07_csi_pro_forward/")

for subdir in Path(res_folder).glob("*"):
    if not subdir.is_dir():
        continue
    ckpt_file = list(subdir.rglob("*.ckpt"))
    assert(len(ckpt_file) == 1)
    ckpt_file = ckpt_file[0]

    arg_file = list(subdir.rglob("args.yaml"))
    assert(len(arg_file) == 1)
    arg_file = arg_file[0]
    args = yaml.safe_load(open(arg_file, "r"))
    split_name = args['split_name'].split("_")[0]
    fold_name = split_name

    #print(split_name, fold_name)
    #print(ckpt_file)

    save_name = res_folder / f"fold_{fold_name}_csi_preds.p"
    cmd_1 = f"python3 run_scripts/predict_forward_ffn.py --checkpoint {ckpt_file} --batch-size 64 --num-workers 16 --dataset-name data/unpaired_mols/bio_mols/all_smis.txt --save-tuples --save-name {save_name} --gpu" 
    print(cmd_1)
    subprocess.call(cmd_1, shell=True)

    tree_out = Path(f"data/paired_spectra/csi2022/csi_spec_preds_fold_{fold_name}/")

    cmd_2 = f"python3 data_processing/forward/08_assign_formulae.py  --preds {save_name} --out {tree_out} --num-bins 15000 --upper-limit 1500"

    print(cmd_2)
    subprocess.call(cmd_2, shell=True)

    cmd_3 = f"python3 data_processing/forward/09_export_trees.py --new-spec-dir {tree_out / 'spectra'}"
    print(cmd_3)
    subprocess.call(cmd_3, shell=True)

    cmd_4 = f"python3 data_processing/forward/10_add_forward_decoys.py --output-folder {tree_out} --labels {tree_out / 'labels.tsv'} --fp-names morgan4096"
    print(cmd_4)
    subprocess.call(cmd_4, shell=True)
