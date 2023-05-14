import mist.retrieval_lib.make_hdf5 as make_hdf5

if __name__=="__main__":

    make_hdf5.make_retrieval_hdf5_file(
        dataset_name="mills", labels_name="labels_with_putative_form.tsv",
        form_to_smi_file="data/raw/hmdb/hmdb_formulae_inchikey.p",
        database_name="inthmdb",
        fp_names=["morgan4096"],
        debug=False,)

    make_hdf5.make_retrieval_hdf5_file(
        dataset_name="mills", labels_name="labels_with_putative_form.tsv",
        form_to_smi_file="data/raw/pubchem/pubchem_formulae_inchikey.p",
        database_name="intpubchem",
        fp_names=["morgan4096"],
        debug=False,)
