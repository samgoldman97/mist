""" 06_retrieval_hdf.py

Build retrieval hdf files necessary

"""

import mist.retrieval_lib.make_hdf5 as make_hdf5
import mist.retrieval_lib.form_subsets as form_subsets

PUBCHEM_FILE = "data/raw/pubchem/cid_smiles.txt"
PUBCHEM_FORMULA = "data/raw/pubchem/pubchem_formuale_inchikey.p"
num_workers = 20 

if __name__=="__main__":

    # First make mapping from formula to all smiles
    built_map = form_subsets.build_form_map(smi_file=PUBCHEM_FILE,
                                            dump_file=PUBCHEM_FORMULA,
                                            debug=False)

    # Construct lookup library
    make_hdf5.make_retrieval_hdf5_file(
        dataset_name="canopus_train_public", labels_name="labels.tsv",
        form_to_smi_file="data/raw/pubchem/pubchem_formulae_inchikey.p",
        database_name="intpubchem",
        fp_names=["morgan4096"],
        debug=False,
    )

    # Make a retrieval ranking file
    make_hdf5.make_ranking_file(
        dataset_name="canopus_train_public",
        hdf_prefix="data/paired_spectra/canopus_train_public/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db",
        labels_name="labels.tsv",
        num_workers=num_workers
    )

    # Subsample for contrastive learning
    make_hdf5.subsample_with_weights(
        hdf_prefix="data/paired_spectra/canopus_train_public/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db",
        labels_file="data/paired_spectra/canopus_train_public/labels.tsv",
        fp_names=["morgan4096"],
        debug=False,
        num_workers=num_workers
    )
