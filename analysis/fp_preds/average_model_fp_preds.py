""" average_model_fp_preds.py

Script to average predictions to use in ensembling.

"""

import re
from collections import defaultdict
import pickle
from pathlib import Path
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp-files",
        help="FP Files to average",
        action="store",
        nargs="+",
    )
    parser.add_argument(
        "--save-name",
        help="Name of outfile",
        action="store",
        default="results/2022_06_06_averaged_preds/",
    )
    return parser.parse_args()


def main():
    args = get_args()
    pickle_objs = [pickle.load(open(i, "rb")) for i in args.fp_files]
    name_to_pred, name_to_targ = defaultdict(lambda: []), defaultdict(lambda: [])
    for pred_obj in pickle_objs:
        for name, pred, targ in zip(
            pred_obj["names"],
            pred_obj["preds"],
            pred_obj["targs"],
        ):
            name_to_pred[name].append(pred)
            name_to_targ[name].append(targ)

    # Average all name_to_pred and name_to_targ
    name_list, pred_list, targ_list = [], [], []
    for k, v in name_to_pred.items():
        pred = v
        targ = name_to_targ[k]
        name_list.append(k)
        pred_list.append(np.vstack(pred).mean(0))
        if targ[0] is not None:
            targ_list.append(np.vstack(targ).mean(0))

    pred_obj_base = pickle_objs[0]
    cur_mod_name = pred_obj_base["args"]["model"]
    pred_obj_base["args"]["model"] = f"{cur_mod_name}Ensemble"
    pred_obj_base["names"] = name_list
    pred_obj_base["preds"] = np.stack(pred_list)

    if len(targ_list) > 0:
        pred_obj_base["targs"] = np.stack(targ_list)
    print(f"Dumping to {args.save_name}")
    with open(args.save_name, "wb") as fp:
        pickle.dump(pred_obj_base, fp)


if __name__ == "__main__":
    main()
