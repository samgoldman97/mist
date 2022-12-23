""" compare_datasets.py 

Compare the overlap betwen two datasets in terms of inchikeys

"""
import os
import argparse
import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-1", default="gnps2015", help="Dataset 1 to compare")
    parser.add_argument("--dataset-2", default="broad", help="Dataset 2 to compare")
    return parser.parse_args()


def compare_dataset(dataset_1, dataset_2):
    """compare_dataset."""

    get_tsv = lambda x: os.path.join(f"data/paired_spectra/{x}/labels.tsv")

    df_1 = pd.read_csv(get_tsv(dataset_1), sep="\t")
    df_2 = pd.read_csv(get_tsv(dataset_2), sep="\t")

    standard_smi_1 = df_1["inchikey"].values
    standard_smi_2 = df_2["inchikey"].values

    standard_smi_1_set = set(standard_smi_1)
    standard_smi_2_set = set(standard_smi_2)

    one_not_two = [i not in standard_smi_2_set for i in standard_smi_1]
    two_not_one = [i not in standard_smi_1_set for i in standard_smi_2]

    intersect = standard_smi_1_set.intersection(standard_smi_2_set)
    print(f"Unique comps in dataset {dataset_1}: {len(standard_smi_1_set)}")
    print(f"Unique comps in dataset {dataset_2}: {len(standard_smi_2_set)}")

    print(f"Ovrerlap comps: {len(intersect)}")

    print(f"Unique set 1 spectra: {np.sum(one_not_two)}")
    print(f"Unique set 2 spectra: {np.sum(two_not_one)}")


if __name__ == "__main__":
    args = get_args()
    compare_dataset(args.dataset_1, args.dataset_2)
