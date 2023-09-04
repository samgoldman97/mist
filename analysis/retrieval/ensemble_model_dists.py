""" average_model_fp_preds.py

Script to average predicted distances from an ensemble of models

"""

import re
from collections import defaultdict
import pickle
from pathlib import Path
import argparse
import numpy as np
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-files",
        help="FP Files to average",
        action="store",
        nargs="+",
    )
    parser.add_argument(
        "--out-file",
        help="Name of outfile",
        action="store",
    )
    return parser.parse_args()


def main():
    args = get_args()

    objs = []
    for in_file in args.in_files:
        pickle_path = Path(in_file)
        assert pickle_path.exists()
        with open(pickle_path, "rb") as fp:
            objs.append(pickle.load(fp))

    out_file = Path(args.out_file)
    out_file.parent.mkdir(exist_ok=True)

    # Merge v_list by computing dicts for each
    pred_obj_base = objs[0]
    name_to_dist = defaultdict(lambda: [])
    name_to_ranking = defaultdict(lambda: [])
    for pred_obj in tqdm(objs):
        for name, dist, ranking in zip(
            pred_obj["names"], pred_obj["dists"], pred_obj["ranking"]
        ):
            sorted_rank = np.argsort(ranking)
            name_to_dist[name].append(dist[sorted_rank])
            name_to_ranking[name] = ranking[sorted_rank]

    # Average all name_to_pred and name_to_targ
    name_list, dist_list, ranking_list = [], [], []
    for k, v in name_to_dist.items():
        name_list.append(k)
        dist_list.append(np.vstack(name_to_dist[k]).mean(0))
        ranking_list.append(name_to_ranking[k])

    cur_mod_name = pred_obj_base["model"]
    pred_obj_base["model"] = f"{cur_mod_name}Ensemble"
    pred_obj_base["names"] = name_list
    pred_obj_base["ranking"] = ranking_list
    pred_obj_base["dists"] = dist_list

    print(f"Dumping to {out_file}")
    with open(out_file, "wb") as fp:
        pickle.dump(pred_obj_base, fp)


if __name__ == "__main__":
    main()
