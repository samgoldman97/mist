"""data_utils.py

Hold different common relative paths

"""
from pathlib import Path

def paired_get_labels(dataset_name, labels_name="labels.tsv", base_folder="data/paired_spectra"):
    base_folder = Path(base_folder)
    return base_folder / f"{dataset_name}/{labels_name}"


def paired_get_spec_folder(dataset_name, base_folder="data/paired_spectra"):
    base_folder = Path(base_folder)
    return base_folder / f"{dataset_name}/spec_files"

def paired_get_magma_folder(dataset_name, base_folder="data/paired_spectra"):
    base_folder = Path(base_folder)
    return base_folder / f"{dataset_name}/magma_outputs"


def paired_get_sirius_folder(dataset_name, base_folder="data/paired_spectra"):
    base_folder = Path(base_folder)
    return base_folder / f"{dataset_name}/sirius_outputs"


def paired_get_sirius_summary(dataset_name, base_folder="data/paired_spectra"):
    base_folder = Path(base_folder)
    return base_folder / f"{dataset_name}/sirius_outputs/summary_statistics/summary_df.tsv"