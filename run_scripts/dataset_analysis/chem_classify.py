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


output_folder = "results/dataset_analyses/"
labels_files = [
    "data/paired_spectra/csi2022/labels.tsv",
    "data/paired_spectra/canopus_train/labels.tsv",
]
save_name = "smi_to_classes.p"

for file in labels_files:
    file = Path(file)
    dataset_name = file.parent.name
    save_name = Path(output_folder) / dataset_name / save_name
    save_name.parent.mkdir(exist_ok=True, parents=True)

    # Load progress cache already
    full_out = {}
    if save_name.exists():
        with open(save_name, "rb") as fp:
            full_out = pickle.load(fp)
            assert isinstance(full_out, dict)

    # Get labels file
    labels = pd.read_csv(file, sep="\t")
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
