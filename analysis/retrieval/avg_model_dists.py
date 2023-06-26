"""avg_model_dists.py

This script is concerned with taking ranking dists for an HDF from 2 files and
average their dists with specified weights
"""

import copy
import pickle
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--lam", help="avg weight for first fp", default=0.7, type=float)
parser.add_argument("--first-ranking", help="Name of first models' rankings")
parser.add_argument("--second-ranking", help="Name of second model's rankings")
parser.add_argument(
    "--save-name", help="Name of save output", action="store", default=None
)
args = parser.parse_args()

first_ranking = Path(args.first_ranking)
second_ranking = Path(args.second_ranking)
lam = args.lam
save_name = args.save_name
lam_str = str(lam).replace(".", "_")
if save_name is None:
    save_name = second_ranking.parent / f"{second_ranking.stem}_merged_dist_{lam_str}.p"

save_name = Path(save_name)
save_name.parent.mkdir(exist_ok=True)
if not first_ranking.exists():
    print(f"File {first_ranking} does not exist")

if not second_ranking.exists():
    print(f"File {second_ranking} does not exist")

with open(first_ranking, "rb") as fp:
    first_ranking_outs = pickle.load(fp)

with open(second_ranking, "rb") as fp:
    second_ranking_outs = pickle.load(fp)

# 1. Get true ind  (map from name to true ranking)
name_to_ranking_1 = dict(
    zip(first_ranking_outs["names"], first_ranking_outs["ranking"])
)
name_to_dists_1 = dict(zip(first_ranking_outs["names"], first_ranking_outs["dists"]))

# For each entry in the loaded rankings
name_to_ranking_2 = dict(
    zip(second_ranking_outs["names"], second_ranking_outs["ranking"])
)
name_to_dists_2 = dict(zip(second_ranking_outs["names"], second_ranking_outs["dists"]))

# For each example get when it was returned
new_names, new_dists, new_rankings = [], [], []
for name, ranking in name_to_ranking_2.items():
    subsets_1 = name_to_ranking_1.get(name)
    subsets_2 = name_to_ranking_2.get(name)

    dist_1 = name_to_dists_1.get(name)
    dist_2 = name_to_dists_2.get(name)

    # Put these both in numerical order
    orig_order_1 = np.argsort(subsets_1)
    orig_order_2 = np.argsort(subsets_2)

    dists_a = dist_1[orig_order_1]
    dists_b = dist_2[orig_order_2]

    # Get the minimum from both orders
    offset_a, offset_b = 0, 0
    if len(subsets_1) > 0:
        offset_a = np.min(subsets_1)
        offset_b = np.min(subsets_2)
    assert offset_a == offset_b

    new_dist = dists_a * (lam) + dists_b * (1 - lam)
    new_argsort = np.argsort(new_dist)
    re_ranked = new_argsort + offset_a
    new_dist = new_dist[new_argsort]

    # Update ranking 2
    new_dists.append(new_dist)
    new_rankings.append(re_ranked)
    new_names.append(name)

second_ranking_outs["names"] = new_names
second_ranking_outs["ranking"] = new_rankings
second_ranking_outs["dists"] = new_dists

# Dump to output
with open(save_name, "wb") as fp:
    pickle.dump(second_ranking_outs, fp)
