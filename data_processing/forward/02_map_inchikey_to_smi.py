""" 02_map_inchikey_to_smi.py

Encode inchikeys to smiles

"""
import pickle
from rdkit import Chem
from mist import utils

PUBCHEM_FILE = "data/raw/pubchem/cid_smiles.txt"
PUBCHEM_INCHI_TO_SMI = "data/raw/pubchem/inchi_to_smi.p"
BIO_INCHI_FILE = "data/unpaired_mols/bio_mols/bio_inchis.csv"
smi_to_classes_out = "data/unpaired_mols/bio_mols/smi_to_classes.p"


def get_pubchem_smi_list(pubchem_file, debug=False):
    """get_pubchem_smi_list."""
    smi_list = []
    with open(pubchem_file, "r") as fp:
        for index, line in enumerate(fp):
            line = line.strip()
            if line:
                smi = line.split("\t")[1].strip()
                smi_list.append(smi)
            if debug and index > 10000:
                return smi_list
    return smi_list


def inchikey_from_smi(smi):
    """inchikey_from_smi."""
    try:
        inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
        return inchikey

    except:
        return None


if __name__ == "__main__":

    debug = False
    smi_list = get_pubchem_smi_list(PUBCHEM_FILE, debug=debug)
    inchikey_list = utils.chunked_parallel(smi_list, inchikey_from_smi, max_cpu=32)
    inchikey_to_smi = dict(zip(inchikey_list, smi_list))
    with open(PUBCHEM_INCHI_TO_SMI, "wb") as fp:
        pickle.dump(inchikey_to_smi, fp)
    with open(PUBCHEM_INCHI_TO_SMI, "rb") as fp:
        inchikey_to_smi = pickle.load(fp)

    # Rewrite the canopus file
    inchi_to_classes = dict()
    for j in open(BIO_INCHI_FILE, "r").readlines():
        line = j.strip().split("\t")
        inchi, classes = line[0], line[1:]
        inchi_to_classes[inchi] = classes

    smi_to_classes = dict()
    for inchi, cs in inchi_to_classes.items():
        if inchi in inchikey_to_smi:
            smi = inchikey_to_smi[inchi]
            smi_to_classes[smi] = cs

    print(f"Number of smiles resolved: {len(smi_to_classes)}")
    with open(smi_to_classes_out, "wb") as fp:
        pickle.dump(smi_to_classes, fp)
