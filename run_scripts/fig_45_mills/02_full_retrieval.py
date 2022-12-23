""" 02_full_retrieval

Compute full retrieval pipeline for CSI2022 retrospective dataset for
alidation

"""

import subprocess
from pathlib import Path
from datetime import datetime

fp_model_dir = "results/2022_10_28_mist_csi_prospective/"
contrastive_model_dir = "results/2022_10_30_contrastive_csi_prospective/"

base_output_folder = Path("results/2022_12_01_prospective_analysis/")
base_output_folder.mkdir(exist_ok=True)

dataset_names = ["mills"]
retrieval_dbs = ["inthmdb", "intpubchem"]


# Prefix for labels
labels_name = "labels_with_putative_form.tsv"
num_workers = 16

for dataset_name in dataset_names:
    dataset_base = base_output_folder / f"{dataset_name}"
    dataset_base.mkdir(exist_ok=True)
    data_dir = Path(f"data/paired_spectra/{dataset_name}")
    labels_file = data_dir / labels_name
    fp_folder = dataset_base / "fp_preds"
    fp_folder.mkdir(exist_ok=True)

    # Step 1: predict fp's
    base_script = f"python3 run_scripts/pred_fp.py --num-workers {num_workers} --gpu  --labels-name {labels_name} --dataset-name {dataset_name}"
    for ctr, ckpt_file in enumerate(Path(fp_model_dir).rglob("*.ckpt")):
        out_folder  = fp_folder / f"model_{ctr}"
        out_folder.mkdir(exist_ok=True)
        cmd = f"{base_script} --model {ckpt_file} --save {out_folder}"
        print(cmd)
        #subprocess.run(cmd, shell=True)

    # Step 2: Average fingerprints
    str_preds = " ".join([str(i) for i in fp_folder.glob("*/*.p")])
    merged_fp_save_name = fp_folder / "merged_fp_preds.p"
    cmd = f"python analysis/fp_preds/average_model_fp_preds.py --fp-files {str_preds} --save-name {merged_fp_save_name}"
    #subprocess.run(cmd, shell=True)


    for retrieval_db in retrieval_dbs:
        hdf_prefix= data_dir / f'retrieval_hdf/{retrieval_db}_with_morgan4096_retrieval_db'

        save_dir = dataset_base / retrieval_db
        fp_retrieval_savedir = save_dir / "fp_retrieval"
        save_dir.mkdir(exist_ok=True)
        fp_retrieval_savedir.mkdir(exist_ok=True)

        # Step 3: Run fp retrieval 
        fp_cmd = f"python3 run_scripts/retrieval_fp.py --labels-file {labels_file} --dist-name cosine --num-workers {num_workers} --hdf-prefix  {hdf_prefix} --fp-pred-file {merged_fp_save_name} --save-dir {fp_retrieval_savedir}"
        fp_out = list(Path(fp_retrieval_savedir).glob("*.p"))[0]
        #fp_out = merged_fp_save_name

        # Replace with merged_fp_preds --> merged_fp_save_name
        #subprocess.run(fp_cmd, shell=True)

        # Step 4: Run contrastive retrieval
        contrast_retrieval_savedir = save_dir / "contrast_retrieval"
        contrast_retrieval_savedir.mkdir(exist_ok=True)
        for ctr, ckpt_file in enumerate(Path(contrastive_model_dir).rglob("*.ckpt")):
            contrast_subdir = contrast_retrieval_savedir / f"{ctr}"
            contrast_subdir.mkdir(exist_ok=True)

            contrast_cmd = f"python run_scripts/retrieval_contrastive.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --dist-name cosine --hdf-prefix {hdf_prefix} --model {ckpt_file} --save-dir {contrast_subdir} --labels-name {labels_name}"
            print(contrast_cmd)
            #subprocess.run(contrast_cmd, shell=True)

        # Step 5: Merge contrastive files
        contrast_ret_files = contrast_retrieval_savedir.glob("*/*.p")
        contrast_out_file = contrast_retrieval_savedir / "ret_file_concat.p"
        in_file_str = " ".join([str(i) for i in contrast_ret_files])
        cmd = f"python analysis/retrieval/ensemble_model_dists.py --in-files {in_file_str} --out-file {contrast_out_file}"
        print(cmd)
        subprocess.call(cmd, shell=True)

        # Step 6: Average contrastive files
        save_name = save_dir / "final_retrieval.p"
        python_avg = f"python3 analysis/retrieval/avg_model_dists.py --first-ranking {fp_out} --second-ranking {contrast_out_file} --lam 0.3 --save-name {save_name}"
        print(python_avg)
        subprocess.run(python_avg, shell=True)

        # Step 7: Extract smiles
        smi_outs = save_dir / "smiles_outputs.tsv"
        names_file = str(hdf_prefix) + "_names.p"
        extract_script = f"python analysis/retrieval/create_smi_output.py --ranking {save_name} --save-name {smi_outs} --k 5 --names-file {names_file}"
        print(extract_script)
        subprocess.run(extract_script, shell=True)

        # Step 8: Classify these
        classif_script = f"python analysis/datasets/pred_classify.py --pred-file {smi_outs} --save-name {smi_outs.parent / 'chem_classes.p'}"
        print(classif_script)
        subprocess.run(classif_script, shell=True)
