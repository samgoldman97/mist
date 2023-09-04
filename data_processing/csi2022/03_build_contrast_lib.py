""" Build retrieval hdf files for evaluation and contrastive learning"""

import mist.retrieval_lib.make_hdf5 as make_hdf5

if __name__ == "__main__":
    full_db = (
        "data/paired_spectra/csi2022/retrieval_hdf/pubchem_with_csi_retrieval_db.h5"
    )
    labels_file="data/paired_spectra/csi2022/labels.tsv"
    fp_file = "fingerprints/precomputed_fp/cache_csi_csi2022.h5"
    make_hdf5.export_contrast_h5(
        hdf_file=full_db,
        labels_file=labels_file,
        fp_names=("csi",),
        subset_size=128,
        num_workers=20,
        fp_file=fp_file
    )

    # Uncomment and run to build retrieval for csi dataset with morgan fp
    labels_file = "data/paired_spectra/csi2022/labels.tsv"
    make_hdf5.make_retrieval_hdf5_file(
        labels_file=labels_file,
        form_to_smi_file="data/unpaired_mols/pubchem/pubchem_formula_inchikey.p",
        output_dir="data/paired_spectra/csi2022/retrieval_hdf/",
        database_name="intpubchem",
        fp_names=("morgan4096",),
        debug=False,
    )

    full_db = "data/paired_spectra/csi2022/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db.h5"
    make_hdf5.export_contrast_h5(
        hdf_file=full_db,
        labels_file=labels_file,
        fp_names=("morgan4096",),
        subset_size=128,
        num_workers=20,
    )

