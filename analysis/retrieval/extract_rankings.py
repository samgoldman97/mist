"""extract_rankings.py

Given a retrieval result that contains distances, extract the ind for correctly
retrieving each example.

Set None for any example not found

Cmd:
```
python3 analysis/retrieval/extract_rankings.py --true-ranking
data/paired_spectra/csi2022/retrieval_hdf/pubchem_with_csi_retrieval_db_ranked.p
--labels data/paired_spectra/csi2022/labels.tsv --ranking
results/2022_08_22_mist_best_aug_fast_lr/2022_08_22-2021_050461_b98650637903469ce90e91e258e9e363/preds/retrieval/retrieval_fp/retrieval_fp_pubchem_with_csi_retrieval_db_csi2022_cosine_0.p
```
"""
import pickle
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--true-ranking", help="Name of file containing true ranking")
parser.add_argument("--labels", help="Map names to inchikeys")
parser.add_argument("--ranking", help="Pred ranking file")
parser.add_argument(
    "--save-name", help="Name of save output", action="store", default=None
)
args = parser.parse_args()

true_ranking = Path(args.true_ranking)
pred_ranking = Path(args.ranking)
labels = Path(args.labels)
save_name = args.save_name
if save_name is None:
    save_name = pred_ranking.parent / f"{pred_ranking.stem}_ind_found.p"
save_name = Path(save_name)
save_name.parent.mkdir(exist_ok=True)

if not true_ranking.exists():
    print(f"File {true_ranking} does not exist")

if not pred_ranking.exists():
    print(f"File {pred_ranking} does not exist")

if not labels.exists():
    print(f"File {labels} does not exist")


# Load in the true ranking file
with open(true_ranking, "rb") as fp:
    subset_ranking = pickle.load(fp)

with open(pred_ranking, "rb") as fp:
    pred_ranking_outs = pickle.load(fp)

df = pd.read_csv(labels, sep="\t")
spec_to_inchikey = dict(df[["spec", "inchikey"]].values)

# top_ks defines the accuracy values to subset at
top_ks = [1, 5, 10, 20, 50, 100, 500, 10000]

# 1. Get true ind  (map from name to true ranking)
name_to_true_ind = {}
for k, v in spec_to_inchikey.items():
    true_ind = subset_ranking.get(v, {}).get("true_ind", [])
    name_to_true_ind[k] = true_ind

name_to_ranking = dict(zip(pred_ranking_outs["names"], pred_ranking_outs["ranking"]))
name_to_dists = dict(zip(pred_ranking_outs["names"], pred_ranking_outs["dists"]))

names, new_rankings = [], []
for name, ranking in name_to_ranking.items():
    actual_targ_fps = name_to_true_ind.get(name)

    subsets = name_to_ranking.get(name)
    dists = name_to_dists.get(name)

    order = np.argsort(dists)
    dists_sorted = dists[order]
    inds_sorted = subsets[order]

    is_true = np.zeros_like(inds_sorted)
    for real in actual_targ_fps:
        is_true = np.logical_or(is_true, inds_sorted == real)

    ind_found = np.argwhere(is_true).flatten()

    # add 1
    new_rank = np.min(ind_found) + 1 if len(ind_found) > 0 else None
    new_rankings.append(new_rank)
    names.append(name)

# Create stats about
new_rankings = np.array(new_rankings)

temp_rankings = np.copy(new_rankings)
temp_rankings[temp_rankings == None] = 100000

subset_acc = {}
for k in top_ks:
    subset_acc[k] = np.mean(temp_rankings <= k)

out_str = json.dumps(subset_acc, indent=2)
print(out_str)

# Define output file
del pred_ranking_outs["dists"]
del pred_ranking_outs["ranking"]

pred_ranking_outs["ind_found"] = new_rankings
pred_ranking_outs["names"] = np.array(names)

# Define ranking
with open(save_name, "wb") as fp:
    pickle.dump(pred_ranking_outs, fp)
