""" cat_fp_preds.py

Merge multiple folds of predictions

"""

import re
from collections import defaultdict
import pickle
from pathlib import Path
import argparse
import numpy as np

PRED_REGEX = "spectra_encoding_(.*).p"


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

    out_obj = merge_objs[0]
    out_names, out_preds, out_targs = [], [], []
    for pred_obj in merge_objs:
        out_names.append(pred_obj["names"])
        out_preds.append(pred_obj["preds"])
        out_targs.append(pred_obj["targs"])

    out_names = np.concatenate(out_names)
    out_preds = np.vstack(out_preds)
    out_targs = np.vstack(out_targs)

    new_order = np.argsort(out_names)
    out_names = out_names[new_order]
    out_preds = out_preds[new_order]
    out_targs = out_targs[new_order]

    out_obj["names"] = out_names
    out_obj["preds"] = out_preds
    out_obj["targs"] = out_targs

    print(f"Dumping to {out_file}")
    with open(out_file, "wb") as fp:
        pickle.dump(out_obj, fp)


if __name__ == "__main__":
    main()
