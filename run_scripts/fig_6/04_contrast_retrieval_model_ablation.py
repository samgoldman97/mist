""" contrast_retrieval_ablation.py 

Compile rankings for:
1. FP Fingerprint
2  MIST Fingeprrint
3. Contrastive FFN
5. Contrastive MIST - Pretrain
4. Contrastive MIST
6. Contrastive MIST + FP

"""
import pickle
import subprocess
from pathlib import Path
import yaml

base_dir = Path("results/2022_11_01_canopus_contrastive_model_ablations/")
fp_dir = Path("results/2022_11_01_canopus_model_ablations/")
out_ranking_folder = base_dir / "ranking_store"
out_ranking_folder.mkdir(exist_ok=True)

num_workers = 16

dataset_name = "canopus_train"
base_data_dir = Path(f"data/paired_spectra/{dataset_name}/")
labels_file = base_data_dir / "labels.tsv"

# 1. FFN fp Fingerprint

for j in fp_dir.glob("ffn_*/preds/*.p"):
    ffn_fp_cmd = f"python3 run_scripts/retrieval_fp.py --labels-file {labels_file} --fp-pred-file {j} --save-dir {out_ranking_folder / 'retrieval-ffn-fp'} --dist-name cosine --num-workers {num_workers} --hdf-prefix {base_data_dir / 'retrieval_hdf/intpubchem_with_morgan4096_retrieval_db'}" 
    print(ffn_fp_cmd)
    #subprocess.run(ffn_fp_cmd, shell=True)


# 2. MIST Fingerprint —> working first
for mist_fp in fp_dir.glob("full_model_*/preds/*.p"):
    mist_fp_cmd = f"python3 run_scripts/retrieval_fp.py --labels-file {labels_file} --fp-pred-file {mist_fp} --save-dir {out_ranking_folder / 'retrieval-mist-fp'} --dist-name cosine --num-workers {num_workers} --hdf-prefix {base_data_dir / 'retrieval_hdf/intpubchem_with_morgan4096_retrieval_db'}"
    print(mist_fp_cmd)
    #subprocess.run(mist_fp_cmd, shell=True)

# Run all contrastive commands (2-5)
base_contrastive = f"python run_scripts/retrieval_contrastive.py --dataset-name {dataset_name} --subset-datasets test_only --num-workers {num_workers} --gpu --dist-name cosine --hdf-prefix {base_data_dir / 'retrieval_hdf/intpubchem_with_morgan4096_retrieval_db'}"
for ckpt in base_dir.rglob("*.ckpt"):
    folder_name = ckpt.parent.parent.name
    args = yaml.safe_load(open(ckpt.parent.parent / "args.yaml", "r"))
    if args['no_pretrain_load']:
        folder_name = "mist-no-pretrain"
    elif 'ffn' in args['ckpt_file']:
        folder_name = "ffn"
    else:
        folder_name = "mist-contrastive"

    save_dir = out_ranking_folder / folder_name
    cmd = f"{base_contrastive} --model-ckpt {str(ckpt)} --save-dir {save_dir}"
    print(cmd)
    #subprocess.call(cmd, shell=True)


# Now run mist contrastive + fingerprint

save_dir_contrast = out_ranking_folder / "mist-contrastive"
save_dir_fp = out_ranking_folder / "retrieval-mist-fp"
fp_dict, contrast_dict = {}, {}
for pfile in save_dir_contrast.glob("*.p"):
    if "ind_found" in str(pfile): continue
    p_out = pickle.load(open(pfile, "rb"))
    contrast_dict[p_out['split_file']] = pfile

for pfile in save_dir_fp.glob("*.p"):
    if "ind_found" in str(pfile): continue
    p_out = pickle.load(open(pfile, "rb"))
    fp_dict[p_out['args']['split_file']] = pfile

for split_file, mist_fp_ranking in fp_dict.items():
    split_num = Path(split_file).stem.rsplit("_", 1)[-1]
    paired_out = out_ranking_folder / f"mist-contrast-fp/{split_num}.p"
    mist_contrast_ranking = contrast_dict[split_file]
    python_avg = f"python3 analysis/retrieval/avg_model_dists.py --first-ranking {mist_contrast_ranking} --second-ranking {mist_fp_ranking} --lam 0.7 --save-name {paired_out}"
    print(python_avg)
    #subprocess.call(python_avg, shell=True)

base_extract = f"python analysis/retrieval/extract_rankings.py  --labels {labels_file} --true-ranking data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_ranked.p"

for out_file in out_ranking_folder.rglob("*.p"):
    new_cmd = f"{base_extract} --ranking {str(out_file)}"
    #subprocess.call(new_cmd, shell=True)
    print(new_cmd)

# Merge ind found files
output = out_ranking_folder /  'ind_found_collective.p'
cmd = f"python3 analysis/retrieval_ablations/merge_ind_found.py --dir {out_ranking_folder} --out {output.name}"
#subprocess.call(cmd, shell=True)

save_dir = base_dir / "plots"
save_dir.mkdir(exist_ok=True)
save_name = save_dir / "model_ablation.pdf"
cmd = f"python3 analysis/retrieval_ablations/model_ablation_plot.py --ablation-file {output} --save-name {save_name}"
subprocess.call(cmd, shell=True)
