""" 03_check_db.py

Check if smiles are in the indicated dataset

"""

import subprocess
import pandas as pd
import pickle
from pathlib import Path
from datetime import datetime


from mist import utils

base_output_folder = Path("results/2022_11_03_prospective_analysis/")
base_output_folder = Path("results/2023_05_10_prospective_reanalysis_forms//")
base_output_folder.mkdir(exist_ok=True)

labels_name = "labels_putative_h_plus.tsv" # Prefix for labels
res_dir = Path("results/2022_10_11_prospective_broad")
num_workers = 16
res_dir.mkdir(exist_ok=True)

dataset_names = ["broad"]
retrieval_dbs = ["inthmdb", "intpubchem"]

for dataset_name in dataset_names:
    dataset_base = base_output_folder / f"{dataset_name}"
    dataset_base.mkdir(exist_ok=True)
    data_dir = Path(f"data/paired_spectra/{dataset_name}")
    labels_file = data_dir / "labels_putative_h_plus.tsv"
    ikey_spec  = pd.read_csv(labels_file, sep="\t")[['inchikey', 'spec']]
    for retrieval_db in retrieval_dbs:
        save_dir = dataset_base / retrieval_db
        hdf_names = data_dir / f'retrieval_hdf/{retrieval_db}_with_morgan4096_retrieval_db_names.p'


        name_set = pickle.load(open(hdf_names, "rb"))
        smiles = list(name_set.values())
        ikeys = utils.chunked_parallel(list(smiles),
                                       utils.inchikey_from_smiles)
        ikey_set = set(ikeys)
        out_entries = []
        for ikey, spec in ikey_spec.values:
            out_entries.append({"name": spec, "in_db": ikey in ikey_set})

        out_df = pd.DataFrame(out_entries)
        out_df.to_csv(save_dir / "spec_found.tsv", sep="\t", index=None)
