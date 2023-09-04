""" stats_spec_dataset.py 

Examples:
python analysis/datasets/stats_spec_dataset.py --dataset data/paired_spectra/broad/labels.tsv --ion "[M+H]+"
python analysis/datasets/stats_spec_dataset.py --dataset data/paired_spectra/canopus_train/labels.tsv --ion "[M+H]+"
python analysis/datasets/stats_spec_dataset.py --dataset data/paired_spectra/csi2022/labels.tsv
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-labels",
        default="data/paired_spectra/canopus_train/labels.tsv",
        help="Dataset 1 to analyze",
    )
    parser.add_argument("--ion-filter", default=None, help="Ion type to filter")
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    labels_file = args.dataset_labels
    dataset = Path(labels_file).parent.name
    ion = args.ion_filter
    df = pd.read_csv(labels_file, sep="\t")
    if ion is not None:
        df = df[df["ionization"] == ion]
    all_smis = set(df["smiles"].values)
    all_specs = set(df["spec"].values)
    print(f"Dataset {dataset} filtered to ion {ion}")
    print(f"Num unique smiles: {len(all_smis)}")
    print(f"Num unique specs: {len(all_specs)}")
