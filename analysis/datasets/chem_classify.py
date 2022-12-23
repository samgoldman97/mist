"""chem_classify.py

Call classyfire rest api for queries in batches

"""

import json
import argparse
from pathlib import Path
import pickle
import time
import requests
from tqdm import tqdm

import pandas as pd
import numpy as np

from mist import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/paired_spectra/csi2022/labels.tsv")
    parser.add_argument(
        "--save-name", default="data/unpaired_mols/bio_mols/new_smi_to_classes.p"
    )
    return parser.parse_args()


def main():
    args = get_args()
    save_name = Path(args.save_name)
    if not save_name.parent.exists():
        raise ValueError()

    # Load progress cache already
    full_out = {}
    if save_name.exists():
        with open(save_name, "rb") as fp:
            full_out = pickle.load(fp)
            assert isinstance(full_out, dict)

    # Get labels file
    label_file = args.labels
    labels = pd.read_csv(label_file, sep="\t")
    rows = labels[["spec", "smiles"]].values

    row_mask = [i not in full_out for i in rows[:, 0]]
    rows = rows[row_mask]

    # Spec, smi
    all_rows = [tuple(i) for i in rows]

    # Go in batches of 10
    all_batches = list(utils.batches(all_rows, 10))
    save_num = 500
    for input_ex in tqdm(all_batches):

        all_datas = utils.simple_parallel(
            input_ex, utils.npclassifer_query, override_cpu=10
        )
        temp_out = {}
        for i in all_datas:
            temp_out.update(i)

        # Add to running output
        full_out.update(temp_out)
        if len(full_out) % save_num == 0:
            # Export
            with open(save_name, "wb") as fp:
                print(f"Len of full out: {len(full_out)}")
                pickle.dump(full_out, fp)

    with open(save_name, "wb") as fp:
        print(f"Len of full out: {len(full_out)}")
        pickle.dump(full_out, fp)


if __name__ == "__main__":
    main()
