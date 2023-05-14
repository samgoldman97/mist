"""fingerprint.py """

import numpy as np
from pathlib import Path
import pickle
import h5py

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem import AllChem, DataStructs

from mist import utils


class Fingerprinter:
    def __init__(
        self, fp_type: str = "morgan4096_3", dataset_name: str = None, **kwargs
    ):
        self.fp_type = fp_type

        if self.fp_type == "morgan4096_2":
            self.nbits = 4096
        elif self.fp_type == "morgan4096_3":
            self.nbits = 4096
        elif self.fp_type == "morgan2048_2":
            self.nbits = 2048
        elif self.fp_type == "morgan2048_3":
            self.nbits = 2048
        elif self.fp_type == "morgan_form":
            self.nbits_morg = 4096
            self.nbits = self.nbits_morg + utils.ELEMENT_DIM_MASS
        elif self.fp_type == "csi":
            self.nbits = 5496
            # , "forwardcsi2022", "forward2022"]
            dataset_names = ["csi2022"]
            self.fp_index_obj = {}
            cur_ind = 0
            self.fp_dataset = None
            # To avoid building multiple caches, just add more items to df...
            for dataset_name in dataset_names:
                base_folder = Path("fingerprints/precomputed_fp/")
                index_file = base_folder / f"cache_csi_{dataset_name}_index.p"
                fp_file = base_folder / f"cache_csi_{dataset_name}.hdf5"
                fp_index_obj = pickle.load(open(index_file, "rb"))
                fp_dataset = h5py.File(fp_file, "r")["features"][:]

                new_fps = fp_dataset.shape[0]
                for i, j in fp_index_obj.items():
                    j = j + cur_ind
                    self.fp_index_obj[i] = j
                if self.fp_dataset is None:
                    self.fp_dataset = fp_dataset
                else:
                    self.fp_dataset = np.vstack([self.fp_dataset, fp_dataset])
                cur_ind += new_fps
        else:
            pass

    def get_nbits(self):
        return self.nbits

    def get_fp_inchikey(self, inchikey):
        if self.fp_type == "csi":
            index = self.fp_index_obj.get(inchikey)
            if index is None:
                return None
            return self.fp_dataset[index]
        else:
            raise NotImplementedError()

    def get_fp(self, mol: Chem.Mol) -> np.ndarray:

        if mol is None:
            return None

        if self.fp_type == "morgan4096_2":
            fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            return fingerprint

        elif self.fp_type == "morgan4096_3":
            fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            return fingerprint

        elif self.fp_type == "morgan2048_2":
            fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            return fingerprint

        elif self.fp_type == "morgan2048_3":
            fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            return fingerprint

        elif self.fp_type == "morgan_form":
            fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=self.nbits_morg)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            form_str = utils.CalcMolFormula(mol)
            form_vec = utils.formula_to_dense_mass_norm(form_str)
            fingerprint = np.concatenate([fingerprint, form_vec])
            return fingerprint
        elif self.fp_type == "csi":
            inchikey = Chem.MolToInchiKey(mol)
            index = self.fp_index_obj.get(inchikey)
            if index is None:
                return None

            return self.fp_dataset[index]
        else:
            raise NotImplementedError()

    def get_fp_weight(self, mol: Chem.Mol) -> (int, np.ndarray):

        if mol is None:
            return None, None
        weight = MolWt(mol)

        if self.fp_type == "morgan4096_2":
            fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)

            return fingerprint, weight
        elif self.fp_type == "morgan4096_3":
            fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)

            return fingerprint, weight
        elif self.fp_type == "morgan2048_2":
            fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)

            return fingerprint, weight
        elif self.fp_type == "morgan2048_3":
            fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=self.nbits)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)

            return fingerprint, weight
        elif self.fp_type == "morgan_form":
            fp = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=self.nbits_morg)
            fingerprint = np.zeros((0,), dtype=np.int32)
            DataStructs.ConvertToNumpyArray(fp, fingerprint)
            form_str = utils.CalcMolFormula(mol)
            form_vec = utils.formula_to_dense_mass_norm(form_str)
            fingerprint = np.concatenate([fingerprint, form_vec])
            return fingerprint, weight
        elif self.fp_type == "csi":
            inchikey = Chem.MolToInchiKey(mol)
            index = self.fp_index_obj.get(inchikey)
            if index is None:
                return None, None

            return self.fp_dataset[index], weight
        else:
            raise NotImplementedError()

    def get_fp_smi(self, smi: str) -> np.ndarray:
        return self.get_fp(Chem.MolFromSmiles(smi), nbits=self.nbits)

    def get_fp_smi_wt(self, smi: str) -> np.ndarray:
        return self.get_fp_weight(Chem.MolFromSmiles(smi))
