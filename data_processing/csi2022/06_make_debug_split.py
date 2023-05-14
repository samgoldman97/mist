""" make_debug_split.py """

import copy
import pandas as pd
import numpy as np

input_file = "data/paired_spectra/csi2022/splits/csi_split_0.txt"
output_file = "data/paired_spectra/csi2022/splits/csi_subset_split.txt"

df_input = pd.read_csv(input_file)

new_df = copy.deepcopy(df_input[["name", "Fold_0"]])
exclude_inds = int(len(new_df) * 0.3)

# Seed
np.random.seed(42)
exclude_inds = np.random.choice(len(new_df), exclude_inds, replace=False)
new_df.loc[exclude_inds, "Fold_0"] = "exclude"
new_df.to_csv(output_file, index=False)
