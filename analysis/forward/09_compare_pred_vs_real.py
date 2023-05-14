import numpy as np
import pickle
from pathlib import Path
import pandas as pd

from mist import utils

dataset = "csi2022"
dataset = "gnps2015_debug"

real_path = Path(f"data/paired_spectra/{dataset}/")
real_tbl = real_path / "sirius_outputs/summary_statistics/summary_df.tsv"
real_names = pd.read_csv(real_path / "labels.tsv", sep="\t")
train_names = pd.read_csv(real_path / "splits/csi_split.txt")
train_entries = train_names[train_names["Fold_0"] == "train"]["name"].values
test_entries = train_names[train_names["Fold_0"] == "test"]["name"].values
real_name_to_smi = dict(real_names[["spec", "smiles"]].values)

real_name_to_file_map = pd.read_csv(real_tbl, sep="\t")
real_name_to_tbl = dict(real_name_to_file_map[["spec_name", "spec_file"]].values)
real_name_to_tree = dict(real_name_to_file_map[["spec_name", "tree_file"]].values)


pred_path = Path("data/unpaired_spectra/canopus_smiles/forward_preds_overfit_train/")
pred_names = pd.read_csv(pred_path / "labels.tsv", sep="\t")
pred_smi_to_name = dict(pred_names[["smiles", "spec"]].values)
pred_name_to_tbl = {
    i: pred_path / f"spectra/{i}.tsv" for i in pred_names["spec"].values
}

out_preds = f"results/2022_08_03_forward_ffn_best/{dataset}_overfit_train.p"
preds = pickle.load(open(out_preds, "rb"))
names_to_preds = dict(zip(preds["names"], preds["preds"]))


for real_name in train_entries:
    smi = real_name_to_smi.get(real_name)
    pred_name = pred_smi_to_name.get(smi)
    if pred_name is None:
        continue
    pred_temp = pred_name_to_tbl[pred_name]
    print(pred_temp)
    df_pred = pd.read_csv(pred_name_to_tbl[pred_name], sep="\t")
    df_real = pd.read_csv(real_name_to_tbl[real_name], sep="\t")
    df_real["wt"] = [utils.formula_mass(i) for i in df_real["formula"]]
    actual_pred = names_to_preds[smi]
    overlapped_formulas = set(df_pred["chemicalFormula"].values).intersection(
        df_real["formula"].values
    )
    print(len(overlapped_formulas) / len(df_real))
