""" 03_contrast_retrieval_data_ablation.py 
"""

import subprocess
from pathlib import Path
import yaml


fp_folder = Path("results/2022_11_01_canopus_data_ablations")
base_dir = Path("results/2022_11_01_canopus_contrastive_data_ablations")

out_ranking_folder = base_dir / "ranking_store"
out_ranking_folder_fp = out_ranking_folder / "fps"
out_ranking_folder_contrast = out_ranking_folder / "contrast"
out_ranking_folder_joint = out_ranking_folder / "joint"

# Make folders
out_ranking_folder.mkdir(exist_ok=True)
out_ranking_folder_fp.mkdir(exist_ok=True)
out_ranking_folder_contrast.mkdir(exist_ok=True)
out_ranking_folder_joint.mkdir(exist_ok=True)

num_workers = 16

dataset_name = "canopus_train"
base_data_dir = Path(f"data/paired_spectra/{dataset_name}/")
labels_file = base_data_dir / "labels.tsv"

# Loop over all contrastive learning models
base_contrastive = f"python run_scripts/retrieval_contrastive.py --dataset-name {dataset_name} --subset-datasets test_only --num-workers {num_workers} --gpu --dist-name cosine --hdf-prefix {base_data_dir / 'retrieval_hdf/intpubchem_with_morgan4096_retrieval_db'}"
base_fp = f"python3 run_scripts/retrieval_fp.py --labels-file {labels_file} --dist-name cosine --num-workers {num_workers} --hdf-prefix {base_data_dir / 'retrieval_hdf/intpubchem_with_morgan4096_retrieval_db'}"
for ckpt in base_dir.rglob("*.ckpt"):

    args_file = ckpt.parent.parent / "args.yaml"
    yaml_args = yaml.safe_load(open(args_file, "r"))

    split_file = yaml_args['split_file']
    split_frac = int(Path(split_file).stem.split("_")[-2])
    
    # Run contrastive model
    save_dir_contrast = out_ranking_folder_contrast / f"{split_frac}"
    contrast_cmd = f"{base_contrastive} --model-ckpt {str(ckpt)} --save-dir {save_dir_contrast}"
    print(contrast_cmd)
    #subprocess.run(contrast_cmd, shell=True)

    # Run fingerprint model
    ckpt_file = Path(yaml_args['ckpt_file'])
    fp_file = list(ckpt_file.parent.parent.rglob("preds/*.p"))[0]
    save_dir_fp = out_ranking_folder_fp / f"{split_frac}"
    fp_cmd = f"{base_fp} --fp-pred-file {fp_file} --save-dir {save_dir_fp}"
    print(fp_cmd)
    #subprocess.run(fp_cmd, shell=True)

    # Get rankings and merge
    r1 = list(save_dir_fp.rglob("*.p"))[-1]
    r2 = list(save_dir_contrast.rglob("*.p"))[-1]

    ctr = 0
    save_name = out_ranking_folder_joint / f"{split_frac}" / f"{ctr}.p"
    while save_name.exists():
        save_name = out_ranking_folder_joint / f"{split_frac}" / f"{ctr}.p"
        ctr += 1

    python_avg = f"python3 analysis/retrieval/avg_model_dists.py --first-ranking {r1} --second-ranking {r2} --lam 0.3 --save-name {save_name}"
    print(python_avg)
    #subprocess.run(python_avg, shell=True)


# Extract all
base_extract = f"python analysis/retrieval/extract_rankings.py  --labels {labels_file} --true-ranking data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_ranked.p"
for out_file in out_ranking_folder_joint.rglob("*.p"):
    new_cmd = f"{base_extract} --ranking {str(out_file)}"
    print(new_cmd)
    #subprocess.call(new_cmd, shell=True)


# Merge ind found files
output = out_ranking_folder_joint /  'ind_found_collective.p'
cmd = f"python3 analysis/retrieval_ablations/merge_ind_found.py --dir {out_ranking_folder_joint} --out {output.name}"
#subprocess.call(cmd, shell=True)

save_dir = base_dir / "plots"
save_dir.mkdir(exist_ok=True)
save_name = save_dir / "data_ablation.pdf"
cmd = f"python3 analysis/retrieval_ablations/data_ablation_plot.py --ablation-file {output} --save-name {save_name}"
subprocess.call(cmd, shell=True)


