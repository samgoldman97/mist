""" build_lookup_quickstart.py

python data_processing/quickstart/build_lookup.py > quickstart/lookup_smiles.txt

"""
import pickle
import pandas as pd

form_file = "data/paired_spectra/quickstart/labels_with_putative_form.tsv"
form_to_smi = "data/raw/hmdb/hmdb_formulae_inchikey.p"

df = pd.read_csv(form_file, sep="\t")
forms = pd.unique(df['formula'])
lookup_smi = []

with open(form_to_smi, "rb") as fp:
    form_map = pickle.load(fp)

smi_list = []
for form in forms:
    temp = form_map.get(form, [])
    smi_list.extend([i[0] for i in temp])

smi_list = list(set(smi_list))
joint_smi_list = "\n".join(smi_list)
print(joint_smi_list)
