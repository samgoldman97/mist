"""FP Predictions batched for repo.

Make FP predictions with the binned FFN model and MIST models and process these
into figures.

"""
import subprocess
from collections import defaultdict
from pathlib import Path
import shutil

dirs = [
    "results/2022_10_06_ffn_binned_csi_best",
    "results/2022_10_25_mist_csi_best_ensemble"
]

num_workers = 16
dataset_name = "csi2022"
base_script = f"python3 run_scripts/pred_fp.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --output-targs --subset-datasets test_only"

# Merge strat determines if we want to make single model or ensemble model
# predictions
merge_strat = "ensemble"
merge_strat = "single"

outdirs = []
for dir_ in dirs:
    fold_to_fp = defaultdict(lambda: [])
    for ckpt in Path(dir_).rglob("*.ckpt"):
        fold_name = ckpt.parent.name
        savedir = ckpt.parent / "preds"
        cmd = f"{base_script} --model-ckpt {str(ckpt)} --save-dir {savedir}"

        #subprocess.call(cmd, shell=True)
        output_fp = list(savedir.glob("*.p"))[0]
        fold_to_fp[fold_name].append(output_fp)

    #
    outdir = Path(dir_) / f"out_preds_{merge_strat}"
    outdir.mkdir(exist_ok=True)
    if merge_strat == "ensemble":
        for k, v in fold_to_fp.items():
            str_preds = " ".join([str(i) for i in v])

            # Run merge predictions
            save_name = outdir / f"{k}_preds.p"
            cmd = f"python analysis/fp_preds/average_model_fp_preds.py --fp-files {str_preds} --save-name {save_name}"

            # average predictions
            #subprocess.call(cmd, shell=True)

    elif merge_strat == "single":
        for k, v in fold_to_fp.items():
            ## Run merge predictions
            save_name = outdir / f"{k}_preds.p"
            shutil.copy2(v[0], save_name)
            # average predictions
    else:
        raise NotImplementedError()
    outdirs.append(outdir)

for dir_ in outdirs:
    dir_ = Path(dir_)
    pred_files = list(dir_.glob("*.p"))
    pred_files_str = " ".join([str(i) for i in pred_files])
    out_file = dir_ / "preds/merged_fp_preds.p"
    out_file.parent.mkdir(exist_ok=True)
    merge_str = f"python analysis/fp_preds/cat_fp_preds.py --in-files {pred_files_str} --out {out_file}"

    print(merge_str)
    #subprocess.call(merge_str, shell=True)


# Conduct plotting
ffn_fp_file = f"results/2022_10_06_ffn_binned_csi_best/out_preds_{merge_strat}/preds/merged_fp_preds.p"
mist_fp_file = f"results/2022_10_25_mist_csi_best_ensemble/out_preds_{merge_strat}/preds/merged_fp_preds.p"
csi_fp_file = "data/paired_spectra/csi2022/prev_results/spectra_encoding_csi2022_Fold_012.p"
res_folder = Path(mist_fp_file).parent / "plots"
res_folder.mkdir(exist_ok=True)


# Scatter plots
cmds = [f"python3 analysis/fp_preds/fp_scatter.py --fp-pred-file {mist_fp_file} --csi-baseline {csi_fp_file} --metric Cosine --pool-method spectra --png",
        f"python3 analysis/fp_preds/fp_scatter.py --fp-pred-file {mist_fp_file} --csi-baseline {csi_fp_file} --metric LL --pool-method spectra --png",
        f"python3 analysis/fp_preds/fp_scatter.py --fp-pred-file {mist_fp_file} --csi-baseline {csi_fp_file} --metric LL --pool-method bit --png", 

        # Boxplot
        f"python3 analysis/fp_preds/fp_boxplot.py --fp-pred-files {csi_fp_file} {mist_fp_file} {ffn_fp_file} --model-names CSI:FingerID MIST FFN --save-name {res_folder / 'boxplot.pdf'}",

        # Barplot
        f"python analysis/fp_preds/cls_barplot.py --pred-file {mist_fp_file} --labels-file data/paired_spectra/csi2022/labels.tsv --baseline {csi_fp_file} --save-name {res_folder / 'barplot.png'} --png"
]

for cmd in cmds:
    print(cmd)
    subprocess.call(cmd, shell=True)
