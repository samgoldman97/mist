"""data_utils.py

Hold different common relative paths

"""


def retrieval_get_folder(lib_name):
    return f"data/retrieval_libraries/{lib_name}/"


def paired_get_labels(dataset_name, labels_name="labels.tsv"):
    return f"data/paired_spectra/{dataset_name}/{labels_name}"


def paired_get_spec_folder(dataset_name):
    return f"data/paired_spectra/{dataset_name}/spec_files"


def paired_get_magma_folder(dataset_name):
    return f"data/paired_spectra/{dataset_name}/magma_outputs/"


def paired_get_sirius_folder(dataset_name):
    return f"data/paired_spectra/{dataset_name}/sirius_outputs/"


def paired_get_sirius_summary(dataset_name):
    return (
        f"data/paired_spectra/{dataset_name}/sirius_outputs/"
        f"summary_statistics/summary_df.tsv"
    )
