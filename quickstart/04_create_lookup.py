""" 04_create_lookup.py

Converts sirius outputs into summary file and
adds putative chem formula from sirius to the quickstart labels file


"""
import mist.retrieval_lib.make_hdf5 as make_hdf5
import mist.utils as utils


if __name__=="__main__":
    smi_list = "quickstart/lookup_smiles.txt"
    all_smi = [i.strip() for i in open(smi_list, "r").readlines()]
    forms = [utils.form_from_smi(i) for i in all_smi]
    ikeys = [utils.inchikey_from_smiles(i) for i in all_smi]
    form_to_smi = {i: [] for i in set(forms)}
    for form, smi, ikey in zip(forms, all_smi, ikeys):
        form_to_smi[form].append((smi, ikey))

    make_hdf5.make_retrieval_hdf5(
        dataset_name="quickstart", labels_name="labels_with_putative_form.tsv",
        form_to_smi=form_to_smi,
        database_name="quickstart",
        fp_names=["morgan4096"],
        debug=False,)
