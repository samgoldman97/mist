"""stats_from_smi_output.py

Given a retrieval result that contains smiles outputs for each example in a
tsv, compute stats about top 1 accuracy


Cmd:
```
python3 analysis/retrieval/stats_from_smi_output.py --labels
data/paired_spectra/broad/labels.tsv  --out-tsv
Results/2022_08_27_mist_no_aug_morgan/2022_08_27-1256_011159_589fafcd07beca2529bd6deab4354946/retrieval/retrieval_fp_intpubchem_with_morgan4096_retrieval_db_broad_cosine_0_k_smi.tsv
```
"""
import argparse
import numpy as np
import pandas as pd
import json

from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument("--labels", help="Map names to inchikeys")
parser.add_argument("--out-tsv", help="Pred ranking file")
args = parser.parse_args()

labels_df = pd.read_csv(args.labels, sep="\t")

name_to_formula = dict(labels_df[["spec", "formula"]].values)
name_to_ikey = dict(labels_df[["spec", "inchikey"]].values)
name_to_adduct = dict(labels_df[["spec", "ionization"]].values)


pred_df = pd.read_csv(args.out_tsv, sep="\t")

# Filter to 0
pred_df = pred_df[pred_df["rank"] == 0]
inchikeys = [Chem.MolToInchiKey(Chem.MolFromSmiles(i)) for i in pred_df["smi"]]
pred_df["inchikey"] = inchikeys

# Get columsn where "Formula correct"
# Get columns where "smiles correct"
# Group by adduct
ions, formula_correct, smi_correct = [], [], []
for name, ikey, form in pred_df[["name", "inchikey", "form"]].values:
    true_form, true_ikey, true_adduct = (
        name_to_formula[name],
        name_to_ikey[name],
        name_to_adduct[name],
    )
    smi_correct.append(ikey == true_ikey)
    ions.append(true_adduct)
    formula_correct.append(form == true_form)
    # print(form, true_form)

pred_df["true_form"] = formula_correct
pred_df["true_smi"] = smi_correct
pred_df["true_adduct"] = ions

frac_true_form = pred_df["true_form"].mean()
frac_true_smi = pred_df["true_smi"].mean()

print(f"Frac true form: {frac_true_form}")
print(f"Frac true smi: {frac_true_smi}")

print("Group by true form % correct")
print(pred_df.groupby("true_form")[["true_smi"]].mean())

print("Group by adduct % correct")
print(pred_df.groupby(["true_form", "true_adduct"])[["true_smi"]].mean())

print("Group by true form and adduct % correct")
print(pred_df.groupby(["true_adduct", "true_form", "true_smi"]).count())

# TODO: Group by various adduct types
