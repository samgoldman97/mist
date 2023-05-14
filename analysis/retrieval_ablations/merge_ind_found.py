"""merge_ind_found.py

Merge ind found in a single folder


"""
import argparse
import pickle
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir", help="Name of save output", action="store", default=None)
parser.add_argument(
    "--out-name",
    help="Name of save output",
    action="store",
    default="ind_found_collective.p",
)
args = parser.parse_args()

in_dir = Path(args.dir)
out_name = in_dir / args.out_name


out_list = []
for f in in_dir.rglob("*_ind_found.p"):
    out_dict = pickle.load(open(f, "rb"))
    out_dict["file"] = str(f)

    out_list.append(out_dict)

pickle.dump(out_list, open(out_name, "wb"))
