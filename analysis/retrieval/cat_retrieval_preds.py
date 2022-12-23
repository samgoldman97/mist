""" cat_retrieval_preds.py

Merge multiple folds of predictions

"""
import pickle
from pathlib import Path
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-files", help="Input files to merge", action="store", nargs="+", default=[]
    )
    parser.add_argument(
        "--out-file",
        help="Name of outfile",
        action="store",
    )
    return parser.parse_args()


def main():
    args = get_args()
    input_files = args.in_files
    out_file = Path(args.out_file)

    out_dir = out_file.parent
    out_dir.mkdir(exist_ok=True)
    pickle_paths = input_files

    merge_objs = []
    for i in pickle_paths:
        with open(i, "rb") as fp:
            objs = pickle.load(fp)
            merge_objs.append(objs)

    # names, dists, rankings
    out_obj = merge_objs[0]
    out_names, out_dists, out_ranking = [], [], []
    for pred_obj in merge_objs:
        out_names.extend(pred_obj["names"])
        out_dists.extend(pred_obj["dists"])
        out_ranking.extend(pred_obj["ranking"])

    out_names = np.array(out_names, dtype=object)
    out_dists = np.array(out_dists, dtype=object)
    out_ranking = np.array(out_ranking, dtype=object)

    new_order = np.argsort(out_names)
    out_names = out_names[new_order]
    out_dists = out_dists[new_order]
    out_ranking = out_ranking[new_order]

    out_obj["names"] = out_names
    out_obj["dists"] = out_dists
    out_obj["ranking"] = out_ranking

    print(f"Dumping to {out_file}")
    with open(out_file, "wb") as fp:
        pickle.dump(out_obj, fp)


if __name__ == "__main__":
    main()
