""" subset_ind_found.py

Subset one dataset to only have the names found from a second dataset

Use this to subset csi finger id predictions to appropriate indices

"""

import pickle
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--modify-ind-found", help="Name of first models' rankings")
parser.add_argument("--ref-ind-found", help="Name of second model's rankings")
parser.add_argument(
    "--save-name", help="Name of save output", action="store", default=None
)
args = parser.parse_args()

targ_rank = Path(args.ref_ind_found)
modify_rank = Path(args.modify_ind_found)


targ_obj = pickle.load(open(targ_rank, "rb"))
modify_obj = pickle.load(open(modify_rank, "rb"))

targ_names = set(targ_obj["names"])


modify_obj_mask = [i in targ_names for i in modify_obj["names"]]

modify_obj["names"] = np.array(modify_obj["names"])[modify_obj_mask]
modify_obj["ind_found"] = np.array(modify_obj["ind_found"])[modify_obj_mask]

with open(args.save_name, "wb") as fp:
    pickle.dump(modify_obj, fp)
