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


data_dir = Path("data/paired_spectra/csi2022")
label_file = data_dir / Path("labels.tsv")


def random_split(inchi_keys):
    """ """
    pass


def main():
    np.random.seed(42)

    df = pd.read_csv(label_file, sep="\t")
    ions = df["ionization"]
    valid_ions = ions == "[M+H]+"
    sub_df = df[valid_ions]

    molecule_set = set(sub_df["inchikey"].values)

    train_frac, test_frac = 0.95, 0.05
    num_train = int(train_frac * len(molecule_set))
    num_test = len(molecule_set) - num_train

    # Divide by inchi keys
    full_inchikey_list = list(molecule_set)
    np.random.shuffle(full_inchikey_list)
    train = set(full_inchikey_list[:num_train])
    test = set(full_inchikey_list[num_train:])
    print(f"Num train total compounds: {len(train)}")
    print(f"Num test total compounds: {len(test)}")

    output_name = data_dir / "splits"
    if not output_name.is_dir():
        output_name.mkdir(exist_ok=True)

    fold_name = "prospective"
    spec_to_entries = defaultdict(lambda: {})
    train_temp = train
    for _, row in sub_df.iterrows():
        spec_id = row["spec"]
        inchikey = row["inchikey"]

        if inchikey in train_temp:
            fold = "train"
        elif inchikey in test:
            fold = "test"
        else:
            fold = "exclude"

        spec_to_entries[spec_id][fold_name] = fold
        spec_to_entries[spec_id]["name"] = spec_id

    export_df = pd.DataFrame(list(spec_to_entries.values()))

    # Name first
    export_df = export_df.sort_index(axis=1, ascending=False)
    export_df.to_csv(output_name / Path(f"prospective_split.csv"), index=False)


if __name__ == "__main__":
    main()
