""" 06_retrieval_hdf.py

Build retrieval hdf files necessary

"""

import mist.retrieval_lib.make_hdf5 as make_hdf5

if __name__=="__main__":

    make_hdf5.make_retrieval_hdf5_file(
        dataset_name="canopus_train", labels_name="labels.tsv",
        form_to_smi_file="data/raw/pubchem/pubchem_formulae_inchikey.p",
        database_name="intpubchem",
        fp_names=["morgan4096"],
        debug=False,
    )

    make_hdf5.make_ranking_file(
        dataset_name="canopus_train",
        hdf_prefix="data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db",
        labels_name="labels.tsv",
        num_workers=20
    )

    make_hdf5.subsample_with_weights(
        hdf_prefix="data/paired_spectra/canopus_train/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db",
        labels_file="data/paired_spectra/canopus_train/labels.tsv",
        fp_names=["morgan4096"],
        debug=False,
        num_workers=20
    )
