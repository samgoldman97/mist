""" 04_make_splits.py
Make canopus splits

"""


import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt


canopus_data_dir = Path("data/paired_spectra/canopus_train")
canopus_label_file = canopus_data_dir / Path("labels.tsv")
np.random.seed(42)


def random_split(inchi_keys):
    """ """
    pass


def main():

    df = pd.read_csv(canopus_label_file, sep="\t")
    ions = df["ionization"]
    valid_ions = ions == "[M+H]+"
    sub_df = df[valid_ions]
    for j in range(3):
        molecule_set = set(sub_df["inchikey"].values)

        train_frac, test_frac = 0.9, 0.1
        num_train = int(train_frac * len(molecule_set))
        num_test = len(molecule_set) - num_train

        # Divide by inchi keys
        full_inchikey_list = list(molecule_set)
        np.random.shuffle(full_inchikey_list)
        train = set(full_inchikey_list[:num_train])
        test = set(full_inchikey_list[num_train:])
        print(f"Num train total compounds: {len(train)}")
        print(f"Num test total compounds: {len(test)}")

        # Export to
        train_sub_fractions = [100, 80, 60, 40, 20]
        for subsample in train_sub_fractions:
            train_temp_num = int(len(train) * subsample / 100)
            train_temp = set(
                np.random.choice(list(train), train_temp_num, replace=False)
            )

            output_name = Path(f"data/paired_spectra/canopus_train/splits")
            if not output_name.is_dir():
                output_name.mkdir(exist_ok=True)

            spec_to_entries = defaultdict(lambda: {})
            for _, row in sub_df.iterrows():
                spec_id = row["spec"]
                inchikey = row["inchikey"]

                if inchikey in train_temp:
                    fold = "train"
                elif inchikey in test:
                    fold = "test"
                else:
                    fold = "exclude"

                fold_name = f"Fold_{subsample}_{j}"
                spec_to_entries[spec_id][fold_name] = fold
                spec_to_entries[spec_id]["name"] = spec_id

            export_df = pd.DataFrame(list(spec_to_entries.values()))

            # Name first
            export_df = export_df.sort_index(axis=1, ascending=False)
            export_df.to_csv(
                output_name / Path(f"canopus_hplus_{subsample}_{j}.csv"), index=False
            )


if __name__ == "__main__":
    main()
