""" Update canopus splits """
import copy
import pandas as pd
import numpy as np
from pathlib import Path

val_frac = 0.05
input_files = list(Path("data/paired_spectra/canopus_train/splits").glob("*.csv"))

for input_file in input_files:
    df_input = pd.read_csv(input_file)
    name_cols = set(df_input.keys())
    name_cols = name_cols.difference(["name"])
    assert len(name_cols) == 1
    fold_name = list(name_cols)[0]

    cur_labels = df_input[fold_name].values
    train_inds = cur_labels == "train"
    test_inds = cur_labels == "test"

    dataset_size = len(cur_labels)
    val_size = int(val_frac * dataset_size)
    # total_inds = np.sum(train_inds) + np.sum(test_inds)

    np.random.seed(42)
    opts = np.argwhere(train_inds).flatten()
    val_inds = np.random.choice(opts, val_size, replace=False)

    new_df = copy.deepcopy(df_input[["name", fold_name]])

    # Seed
    new_df.loc[val_inds, fold_name] = "val"
    new_df.to_csv(input_file, index=False)
