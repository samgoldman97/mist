""" predict_assign.py """

import yaml
import subprocess
from pathlib import Path

# Step 1: Make forward predictions

valid_folds = ["0"]
res_folder = Path('results/2022_10_21_canopus_forward/')

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
    split_name = args['split_name']


    fold_num = split_name[-5]
    if str(fold_num) not in valid_folds:
        print(f"skipping fold {fold_num}")
        continue

    #print(split_name, fold_num)
    #print(ckpt_file)

    save_name = res_folder / f"fold_{fold_num}_morgan_preds.p"
    cmd_1 = f"python run_scripts/predict_forward_ffn.py --checkpoint {ckpt_file} --batch-size 64 --num-workers 16 --dataset-name data/unpaired_mols/bio_mols/all_smis.txt --save-tuples --save-name {save_name} --gpu"
    print(cmd_1)
    #subprocess.call(cmd_1, shell=True)

    tree_out = Path(f"data/paired_spectra/canopus_train/morgan4096_spec_preds_fold_{fold_num}/")
    split_file = str(tree_out.parent / "splits" / split_name)
    orig_labels = str(tree_out.parent / "labels.tsv")

    cmd_2 = f"python3 data_processing/forward/08_assign_formulae.py  --preds {save_name} --out {tree_out} --num-bins 15000 --upper-limit 1500 --orig-labels {orig_labels} --split-file {split_file}"

    print(cmd_2)
    #subprocess.call(cmd_2, shell=True)

    cmd_3 = f"python3 data_processing/forward/09_export_trees.py --new-spec-dir {tree_out / 'spectra'}"
    print(cmd_3)
    subprocess.call(cmd_3, shell=True)

    cmd_4 = f"python3 data_processing/forward/10_add_forward_decoys.py --output-folder {tree_out} --labels {tree_out / 'labels.tsv'} --fp-names morgan4096"
    print(cmd_4)
    subprocess.call(cmd_4, shell=True)
