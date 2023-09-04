"""ranking_summary.py

Given the results from extract_rankings.py, compute the ranking table

Cmd:
```
python3 analysis/retrieval/ranking_summary.py --ranking-file
```
"""
import pickle
import argparse
import numpy as np
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--ranking-file", help="Pred ranking file")
parser.add_argument(
    "--subset-datasets",
    action="store",
    default="none",
    choices=["none", "test_only"],
    help="Settings for how to subset the dataset",
)
parser.add_argument(
    "--split-file",
    action="store",
    default=None,
    help="Include split file to use only test file",
)
args = parser.parse_args()
ranking_file = args.ranking_file
subset_strat = args.subset_datasets
split_file = args.split_file

with open(ranking_file, "rb") as fp:
    ranking_inputs = pickle.load(fp)


# top_ks defines the accuracy values to subset at
top_ks = [1, 5, 10, 20, 50, 100, 200, 500]

inds_found, names = ranking_inputs["ind_found"], ranking_inputs["names"]
inds_found = inds_found.astype(float)
inds_found[np.isnan(inds_found)] = 100000

if subset_strat == "none":
    pass
elif subset_strat == "test_only":
    split_df = pd.read_csv(split_file, sep=",")
    split = sorted(list(set(split_df.keys()).difference("name")))[0]
    print(f"Split name: {split}")
    valid_names = set(split_df["name"][split_df[split] == "test"].values)
    valid_mask = [name in valid_names for ind, name in enumerate(names)]
    inds_found = inds_found[valid_mask]
    names = names[valid_mask]


subset_acc = {}
for k in top_ks:
    subset_acc[k] = np.mean(inds_found <= k)

out_str = json.dumps(subset_acc, indent=2)
print(out_str)
