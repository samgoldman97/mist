"""
featurizers.py

Hold featurizers & collate fns for various spectra and molecules in a single
file
"""
from pathlib import Path
import logging
import pickle
from abc import ABC, abstractmethod
from typing import List, Dict, Callable

import h5py

import json
import pandas as pd
import numpy as np
import torch

from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint

from mist import utils
from mist.magma import magma_utils
from mist.data import data, data_utils


def get_mol_featurizer(mol_features, **kwargs):
    """ConsoleLogger."""

    return {"none": NoneFeaturizer, "fingerprint": FingerprintFeaturizer,}[
        mol_features
    ](**kwargs)


def get_spec_featurizer(spec_features, **kwargs):
    return {
        "none": NoneFeaturizer,
        "binned": BinnedFeaturizer,
        "mz_xformer": MZFeaturizer,
        "peakformula": PeakFormula,
        "peakformula_test": PeakFormulaTest,
    }[spec_features](**kwargs)


def get_paired_featurizer(spec_features, mol_features, **kwargs):
    """get_paired_featurizer.

    Args:
        spec_features (str): Spec featurizer
        mol_features (str): Mol featurizer

    """

    mol_featurizer = get_mol_featurizer(mol_features, **kwargs)
    spec_featurizer = get_spec_featurizer(spec_features, **kwargs)
    paired_featurizer = PairedFeaturizer(spec_featurizer, mol_featurizer, **kwargs)
    return paired_featurizer


class PairedFeaturizer(object):
    """PairedFeaturizer"""

    def __init__(self, spec_featurizer, mol_featurizer, **kwarg):
        """__init__."""
        self.spec_featurizer = spec_featurizer
        self.mol_featurizer = mol_featurizer

    def featurize_mol(self, mol: data.Mol, **kwargs) -> Dict:
        return self.mol_featurizer.featurize(mol, **kwargs)

    def featurize_spec(self, mol: data.Mol, **kwargs) -> Dict:
        return self.spec_featurizer.featurize(mol, **kwargs)

    def get_mol_collate(self) -> Callable:
        return self.mol_featurizer.collate_fn

    def get_spec_collate(self) -> Callable:
        return self.spec_featurizer.collate_fn

    def set_spec_featurizer(self, spec_featurizer):
        self.spec_featurizer = spec_featurizer

    def set_mol_featurizer(self, mol_featurizer):
        self.mol_featurizer = mol_featurizer


class Featurizer(ABC):
    """Featurizer"""

    def __init__(
        self, dataset_name: str = None, cache_featurizers: bool = False, **kwargs
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.cache_featurizers = cache_featurizers
        self.cache = {}

    @abstractmethod
    def _encode(self, obj: object) -> str:
        """Encode object into a string representation"""
        raise NotImplementedError()

    def _featurize(self, obj: object) -> Dict:
        """Internal featurize class that does not utilize the cache"""
        raise {}

    def featurize(self, obj: object, train_mode=False, **kwargs) -> Dict:
        """Featurizer a single object"""
        encoded_obj = self._encode(obj)

        if self.cache_featurizers:
            if encoded_obj in self.cache:
                featurized = self.cache[encoded_obj]
            else:
                featurized = self._featurize(obj)
                self.cache[encoded_obj] = featurized
        else:
            featurized = self._featurize(obj)

        return featurized


class NoneFeaturizer(Featurizer):
    """NoneFeaturizer"""

    def _encode(self, obj) -> str:
        return ""

    @staticmethod
    def collate_fn(objs) -> Dict:
        return {}

    def featurize(self, *args, **kwargs) -> Dict:
        """Override featurize with empty dict return"""
        return {}


class MolFeaturizer(Featurizer):
    """MolFeaturizer"""

    def _encode(self, mol: data.Mol) -> str:
        """Encode mol into smiles repr"""
        smi = mol.get_smiles()
        return smi


class SpecFeaturizer(Featurizer):
    """SpecFeaturizer"""

    def _encode(self, spec: data.Spectra) -> str:
        """Encode spectra into name"""
        return spec.get_spec_name()


class FingerprintFeaturizer(MolFeaturizer):
    """MolFeaturizer"""

    def __init__(self, fp_names: List[str], dataset_name: str = None, **kwargs):
        """__init__

        Args:
            fp_names (List[str]): List of
            dataset_name (str) : Name of dataset
            nbits (int): Number of bits

        """
        super().__init__(**kwargs)
        self._fp_cache = {}
        self._morgan_projection = np.random.randn(50, 2048)
        self.fp_names = fp_names

        # Only for csi fp
        self.dataset_name = dataset_name
        self._root_dir = Path().resolve()

    @staticmethod
    def collate_fn(mols: List[dict]) -> dict:
        fp_ar = torch.tensor(mols)
        return {"mols": fp_ar}

    def featurize_smiles(self, smiles: str, **kwargs) -> np.ndarray:
        """featurize_smiles.

        Args:
            smiles (str): smiles
            kwargs:

        Returns:
            Dict:
        """

        mol_obj = data.Mol.MolFromSmiles(smiles)
        return self._featurize(mol_obj)

    def _featurize(self, mol: data.Mol, **kwargs) -> Dict:
        """featurize.

        Args:
            mol (Mol)

        """
        fp_list = []
        for fp_name in self.fp_names:
            ## Get all fingerprint bits
            fingerprint = self._get_fingerprint(mol, fp_name)
            fp_list.append(fingerprint)

        fp = np.concatenate(fp_list)
        return fp

    ##### Fingerprint functions
    def _get_morgan_fp_base(self, mol: data.Mol, nbits: int = 2048, radius=2):
        """get morgan fingeprprint"""
        fp_fn = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
        mol = mol.get_rdkit_mol()
        fingerprint = fp_fn(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _get_morgan_2048(self, mol: data.Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=2048)

    def _get_morgan_projection(self, mol: data.Mol):
        """get morgan fingeprprint"""

        morgan_fp = self._get_morgan_fp_base(mol, nbits=2048)

        output_fp = np.einsum("ij,j->i", self._morgan_projection, morgan_fp)
        return output_fp

    def _get_morgan_1024(self, mol: data.Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=1024)

    def _get_morgan_512(self, mol: data.Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=512)

    def _get_morgan_256(self, mol: data.Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=256)

    def _get_morgan_4096(self, mol: data.Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=4096)

    def _get_morgan_4096_3(self, mol: data.Mol):
        """get morgan fingeprprint"""
        return self._get_morgan_fp_base(mol, nbits=4096, radius=3)

    def _get_maccs(self, mol: data.Mol):
        """get maccs fingerprint"""
        mol = mol.get_rdkit_mol()
        fingerprint = GetMACCSKeysFingerprint(mol)
        array = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _fill_precomputed_cache_hdf5(self, file_prefix):
        """Get precomputed fp cache"""
        if file_prefix not in self._fp_cache:
            hdf5_file = Path(f"{file_prefix}.hdf5")
            index_file = Path(f"{file_prefix}_index.p")
            if not hdf5_file.exists() or not index_file.exists():
                raise ValueError(f"Cannot find file {hdf5_file} or file {index_file}")

            ## First get pickle
            logging.info("Loading h5 indices")
            index_obj = pickle.load(open(index_file, "rb"))

            ## Then get hdf5
            logging.info("Loading h5 features")
            dataset = h5py.File(hdf5_file, "r")["features"]
            logging.info("Stored in fp_cache")
            self._fp_cache[file_prefix] = (index_obj, dataset)

    def _get_precomputed_hdf5(self, mol, file_location):
        """Get precomputed hdf5 of a single molecule"""
        self._fill_precomputed_cache_hdf5(file_location)
        index, cache = self._fp_cache[file_location]
        inchikey = mol.get_inchikey()
        if inchikey in index:
            return cache[index[inchikey]]
        else:
            logging.info(f"Unable to find inchikey {inchikey} in {file_location}")
            # Create empty vector
            return np.zeros(cache[0].shape)

    def _get_map_2048(self, mol):
        """get precomputed map2048 fingerprint"""
        precompute_file = self._root_dir / Path(
            "fingerprints/precomputed_fp/cache_map4_2048.p"
        )
        return self._get_precomputed_hdf5(mol, precompute_file)

    def _get_pyfingerprint(self, mol, name, nbit=None):
        """Get any fingerprint from the pyfingerprint family"""
        precompute_file = self._root_dir / Path(
            f"fingerprints/precomputed_fp/cache_{name}"
        )
        return self._get_precomputed_hdf5(mol, precompute_file)

    def _get_klekota_roth(self, mol):
        return self._get_pyfingerprint(mol, "klekota-roth")

    def _get_cdk(self, mol):
        return self._get_pyfingerprint(mol, "cdk")

    def _get_csi(self, mol):
        precompute_file = self._root_dir / Path(
            f"fingerprints/precomputed_fp/cache_csi_{self.dataset_name}"
        )
        return self._get_precomputed_hdf5(mol, precompute_file)

    def _get_pubchem(self, mol):
        return self._get_pyfingerprint(mol, "pubchem")

    def _get_fp3(self, mol):
        return self._get_pyfingerprint(mol, "FP3", nbit=55)

    def _get_contextgin(self, mol):
        return self._get_pyfingerprint(mol, "gin_supervised_contextpred")

    def _get_maccs_cdk(self, mol):
        return self._get_pyfingerprint(mol, "maccs-cdk")

    @classmethod
    def get_fingerprint_size(cls, fp_names: list = [], **kwargs):
        """Get list of fingerprint size"""
        fp_name_to_bits = {
            "morgan256": 256,
            "morgan512": 512,
            "morgan1024": 1024,
            "morgan2048": 2048,
            "morgan_project": 50,
            "morgan4096": 4096,
            "morgan4096_3": 4096,
            "map2048": 2048,
            "maccs": 167,
            "maccs-cdk": 167,
            "klekota-roth": 4860,
            "cdk": 307,
            "pubchem": 881,
            "FP3": 55,
            "contextgin": 300,
            "csi": 5496,
        }
        num_bits = 0
        for fp_name in fp_names:
            num_bits += fp_name_to_bits.get(fp_name)
        return num_bits

    def _get_fingerprint(self, mol: data.Mol, fp_name: str):
        """_get_fingerprint_fn"""
        return {
            "morgan256": self._get_morgan_256,
            "morgan512": self._get_morgan_512,
            "morgan1024": self._get_morgan_1024,
            "morgan2048": self._get_morgan_2048,
            "morgan_project": self._get_morgan_projection,
            "morgan4096": self._get_morgan_4096,
            "morgan4096_3": self._get_morgan_4096_3,
            "map2048": self._get_map_2048,
            "maccs": self._get_maccs,
            "maccs-cdk": self._get_maccs_cdk,
            "klekota-roth": self._get_klekota_roth,
            "cdk": self._get_cdk,
            "csi": self._get_csi,
            "pubchem": self._get_pubchem,
            "FP3": self._get_fp3,
            "contextgin": self._get_contextgin,
        }[fp_name](mol)

    def dist(self, mol_1, mol_2) -> np.ndarray:
        """Return 2048 bit molecular fingerprint"""
        fp1 = self.featurize(mol_1)
        fp2 = self.featurize(mol_2)
        tani = 1 - (((fp1 & fp2).sum()) / (fp1 | fp2).sum())
        return tani

    def dist_batch(self, mol_list) -> np.ndarray:
        """Return 2048 bit molecular fingerprint"""

        fps = []
        if len(mol_list) == 0:
            return np.array([[]])

        for mol_temp in mol_list:
            fps.append(self.featurize(mol_temp))

        fps = np.vstack(fps)

        fps_a = fps[:, None, :]
        fps_b = fps[None, :, :]

        intersect = (fps_a & fps_b).sum(-1)
        union = (fps_a | fps_b).sum(-1)
        tani = 1 - intersect / union
        return tani

    def dist_one_to_many(self, mol, mol_list) -> np.ndarray:
        """Return 2048 bit molecular fingerprint"""

        fps = []
        if len(mol_list) == 0:
            return np.array([[]])

        for mol_temp in mol_list:
            fps.append(self.featurize(mol_temp))

        fp_a = self.featurize(mol)

        fps = np.vstack(fps)

        fps_a = fp_a[None, :]
        fps_b = fps

        intersect = (fps_a & fps_b).sum(-1)
        union = (fps_a | fps_b).sum(-1)

        # Compute dist
        tani = 1 - intersect / union
        return tani


class BinnedFeaturizer(SpecFeaturizer):
    """BinnedFeaturizer"""

    def __init__(
        self,
        cleaned_peaks: bool = False,
        upper_limit: int = 1500,
        num_bins: int = 2000,
        **kwargs,
    ):
        """__init__"""
        super().__init__(**kwargs)

        self.cleaned_peaks = cleaned_peaks
        self.sirius_folder = Path(
            data_utils.paired_get_sirius_folder(self.dataset_name)
        )

        if self.cleaned_peaks:
            peak_file_summary = data_utils.paired_get_sirius_summary(self.dataset_name)
            summary_df = pd.read_csv(peak_file_summary, sep="\t", index_col=0)
            self.spec_name_to_file = dict(summary_df[["spec_name", "spec_file"]].values)
            self.spec_name_to_file = {
                k: Path(v).resolve() for k, v in self.spec_name_to_file.items()
            }
        self.upper_limit = upper_limit
        self.num_bins = num_bins

    @staticmethod
    def collate_fn(input_list: List[dict]) -> Dict:
        """collate_fn.

        Input list of dataset outputs

        Args:
            input_list (List[Spectra]): Input list containing spectra to be
                collated
        Return:
            Dictionary containing batched results and list of how many channels are
            in each tensor
        """
        # Determines the number of channels
        names = [j["name"] for j in input_list]
        input_list = [j["spec"] for j in input_list]
        stacked_batch = torch.vstack([torch.tensor(spectra) for spectra in input_list])
        return_dict = {
            "spectra": stacked_batch,
            "names": names,
        }
        return return_dict

    def convert_spectra_to_ar(self, spec, **kwargs) -> np.ndarray:
        """Converts the spectra to a normalized ar

        Args:
            spec

        Returns:
            np.ndarray of shape where each channel has
        """
        spectra_ar = spec.get_spec()
        if self.cleaned_peaks:
            spec_name = spec.get_spec_name()
            cleaned_tsv = Path(self.spec_name_to_file.get(spec_name))
            if cleaned_tsv is not None and cleaned_tsv.exists():
                _, spectra_ar = zip(*utils.parse_tsv_spectra(cleaned_tsv))
            else:
                logging.info(f"Skipping cleaned spec for {spec_name}")

        binned_spec = utils.bin_spectra(
            spectra_ar, num_bins=self.num_bins, upper_limit=self.upper_limit
        )
        normed_spec = utils.norm_spectrum(binned_spec)

        # Mean over 0 channel
        normed_spec = normed_spec.mean(0)
        return normed_spec

    def _featurize(self, spec: data.Spectra, **kwargs) -> Dict:
        """featurize.

        Args:
            spec (Spectra)

        """
        # return binned spectra
        normed_spec = self.convert_spectra_to_ar(spec, **kwargs)
        return {"spec": normed_spec, "name": spec.get_spec_name()}


class MZFeaturizer(SpecFeaturizer):
    """MZFeaturizer"""

    def __init__(
        self,
        cleaned_peaks: bool = False,
        upper_limit: int = 1500,
        max_peaks: int = 50,
        **kwargs,
    ):
        """__init__"""
        super().__init__(**kwargs)

        self.cleaned_peaks = cleaned_peaks
        self.sirius_folder = Path(
            data_utils.paired_get_sirius_folder(self.dataset_name)
        )
        self.max_peaks = max_peaks

        if self.cleaned_peaks:
            peak_file_summary = data_utils.paired_get_sirius_summary(self.dataset_name)
            summary_df = pd.read_csv(peak_file_summary, sep="\t", index_col=0)
            self.spec_name_to_file = dict(summary_df[["spec_name", "spec_file"]].values)
            self.spec_name_to_file = {
                k: Path(v).resolve() for k, v in self.spec_name_to_file.items()
            }
        self.upper_limit = upper_limit

    @staticmethod
    def collate_fn(input_list: List[dict]) -> Dict:
        """collate_fn.

        Input list of dataset outputs

        Args:
            input_list (List[Spectra]): Input list containing spectra to be
                collated
        Return:
            Dictionary containing batched results and list of how many channels are
            in each tensor
        """
        # Determines the number of channels
        names = [j["name"] for j in input_list]
        input_list = [torch.from_numpy(j["spec"]).float() for j in input_list]

        # Define tensor of input lens
        input_lens = torch.tensor([len(spectra) for spectra in input_list])

        # Pad the input list using torch function
        input_list_padded = torch.nn.utils.rnn.pad_sequence(
            input_list, batch_first=True, padding_value=0
        )

        return_dict = {
            "spectra": input_list_padded,
            "input_lens": input_lens,
            "names": names,
        }
        return return_dict

    def convert_spectra_to_mz(self, spec, **kwargs) -> np.ndarray:
        """Converts the spectra to a normalized ar

        Args:
            spec

        Returns:
            np.ndarray of shape where each channel has
        """
        if self.cleaned_peaks:
            spec_name = spec.get_spec_name()
            cleaned_tsv = Path(self.spec_name_to_file.get(spec_name))
            if cleaned_tsv is not None and cleaned_tsv.exists():
                _, spectra_ar = zip(*utils.parse_tsv_spectra(cleaned_tsv))
            else:
                logging.info(f"Skipping cleaned spec for {spec_name}")
            merged = spectra_ar
        else:
            spectra_ar = spec.get_spec()

            merged = utils.merge_norm_spectra(spectra_ar)

        # Sort the merged peaks by intensity ([:, 1]) and limit to self.maxpeaks
        merged = merged[merged[:, 1].argsort()[::-1][: self.max_peaks]]

        parentmass = spec.parentmass
        # Make sure MS1 is on top with intensity 2
        merged = np.vstack([[parentmass, 2], merged])
        return merged

    def _featurize(self, spec: data.Spectra, **kwargs) -> Dict:
        """featurize.

        Args:
            spec (Spectra)

        """
        # return binned spectra
        normed_spec = self.convert_spectra_to_mz(spec, **kwargs)
        return {"spec": normed_spec, "name": spec.get_spec_name()}


class PeakFormula(SpecFeaturizer):
    """PeakFormula."""

    cat_types = {"frags": 0, "loss": 1, "ab_loss": 2, "cls": 3}
    num_inten_bins = 10
    num_types = len(cat_types)
    cls_type = cat_types.get("cls")
    aug_nbits = 2048

    def __init__(
        self,
        add_forward_specs: bool = False,
        augment_data: bool = False,
        augment_prob: float = 1,
        remove_prob: float = 0.1,
        remove_weights: float = "uniform",
        inten_prob: float = 0.1,
        use_cls: bool = False,
        cls_type: str = "ms1",
        magma_aux_loss: bool = False,
        forward_aug_folder: str = None,
        max_peaks: int = None,
        inten_transform: str = "float",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_cls = use_cls
        self.cls_type = cls_type
        self.add_forward_specs = add_forward_specs
        self.augment_data = augment_data
        self.remove_prob = remove_prob
        self.augment_prob = augment_prob
        self.remove_weights = remove_weights
        self.inten_prob = inten_prob
        self.magma_aux_loss = magma_aux_loss
        self.forward_aug_folder = forward_aug_folder
        self.max_peaks = max_peaks
        self.inten_transform = inten_transform

        # Get sirius folder
        self.sirius_folder = Path(
            data_utils.paired_get_sirius_folder(self.dataset_name)
        )
        peak_file_summary = data_utils.paired_get_sirius_summary(self.dataset_name)
        summary_df = pd.read_csv(peak_file_summary, sep="\t", index_col=0)
        self.spec_name_to_tree_file = dict(
            summary_df[["spec_name", "tree_file"]].values
        )

        # If add magma specs, make sure to also add this tree file mapping
        if self.add_forward_specs:
            magma_file_summary = (
                Path(self.forward_aug_folder) / "forward_trees/forward_tree_summary.tsv"
            )
            base_folder = magma_file_summary.parent
            summary_df = pd.read_csv(magma_file_summary, sep="\t", index_col=0)
            addt_map = dict(summary_df[["name", "file_loc"]].values)
            addt_map = {i: base_folder / j for i, j in addt_map.items()}
            self.spec_name_to_tree_file.update(addt_map)

        if self.magma_aux_loss:
            # Load smile-fingerprint dict
            magma_folder = data_utils.paired_get_magma_folder(self.dataset_name)
            index_file = Path(magma_folder) / "magma_smiles_fp_index.p"
            fp_file = Path(magma_folder) / "magma_smiles_fp.hdf5"
            mapping_file = Path(magma_folder) / "magma_file_mapping_df.csv"

            # Load in summary files
            summary_df = pd.read_csv(mapping_file, index_col=0)
            self.spec_name_to_magma_file = dict(
                summary_df[["spec_name", "magma_file"]].values
            )

            if not (index_file.exists() and fp_file.exists() and mapping_file.exists()):
                raise ValueError(
                    f"Expected to find pre-computed files {index_file}, {fp_file}, {mapping_file}"
                )

            # Load in fp files
            self.fp_index_obj = pickle.load(open(index_file, "rb"))
            self.fp_dataset = h5py.File(fp_file, "r")["features"]

    def _get_peak_dict(self, spec: data.Spectra) -> dict:
        """get_peak_formula.

        Args:
            spec (data.Spectra): Spectra data type

        Return:
            Indices
        """
        spec_name = spec.get_spec_name()
        frag_tree_file = Path(self.spec_name_to_tree_file[spec_name])

        if not frag_tree_file.exists():
            return {}

        tree = json.load(open(frag_tree_file, "r"))
        root_form = tree["molecularFormula"]

        out_dict = {"frags": [], "loss": [], "ab_loss": [], "root_form": root_form}
        node_ind_to_inten = {}
        inten_list = []

        for j in tree["fragments"]:

            # Compute fragments
            mol_form = j["molecularFormula"]
            mol_ind = j["id"]
            mol_inten = j["relativeIntensity"]
            # Replace nan with 1
            if str(float(mol_inten)) == "nan":
                mol_inten = 1
            mz = j["mz"]
            node_ind_to_inten[mol_ind] = float(mol_inten)
            inten_list.append(node_ind_to_inten[mol_ind])

            ind = None
            out_dict["frags"].append((ind, mz, mol_form, mol_inten))

            # Compute abs losses
            ab_loss = utils.formula_difference(root_form, mol_form)
            if ab_loss.strip() == "":
                continue

            ind = None
            out_dict["ab_loss"].append((ind, None, ab_loss, mol_inten))

        for j in tree["losses"]:
            mol_form = j["molecularFormula"]
            ind = None
            src, trg = j["source"], j["target"]
            inten = (node_ind_to_inten[src] + node_ind_to_inten[trg]) / 2
            out_dict["loss"].append((ind, None, mol_form, inten))

        if self.max_peaks is not None:
            inten_list = sorted(inten_list)[::-1]
            cutoff_ind = min(len(inten_list) - 1, self.max_peaks)
            cutoff_inten = inten_list[cutoff_ind]
            out_dict["frags"] = [i for i in out_dict["frags"] if i[-1] > cutoff_inten]
        return out_dict

    def augment_peak_dict(self, peak_dict: dict, **kwargs):
        """augment_peak_dict.

        Args:
            peak_dict (dict): Dictionary containing peak dict info to augment

        Return:
            peak_dict
        """
        ### Operations:
        # 1. Add
        # 2. Remove
        # 3. Scale

        ## Only scale frags
        frag_tuples = peak_dict["frags"]
        keep_tuples = []
        # cur_formulae = np.vstack(list(zip(*frag_tuples))[-1])
        start_intens = np.vstack(list(zip(*frag_tuples))[-2])
        max_inten = 1e-6

        ## Compute removal probability
        # Temp
        num_modify_peaks = len(frag_tuples)  # - 1
        keep_prob = 1 - self.remove_prob
        num_to_keep = np.random.binomial(
            n=num_modify_peaks,
            p=keep_prob,
        )
        # Temp
        keep_inds = np.arange(0, num_modify_peaks)  # + 1)

        # Quadratic probability weighting
        if self.remove_weights == "quadratic":
            # Temp
            # keep_probs = start_intens[1:].reshape(-1) ** 2 + 1e-9
            keep_probs = start_intens[0:].reshape(-1) ** 2 + 1e-9
            keep_probs = keep_probs / keep_probs.sum()
        elif self.remove_weights == "uniform":
            ## Temp
            # keep_probs = start_intens[1:] + 1e-9
            keep_probs = start_intens[0:] + 1e-9
            keep_probs = np.ones(len(keep_probs)) / len(keep_probs)
        elif self.remove_weights == "exp":
            ## Temp
            # keep_probs = start_intens[1:] + 1e-9
            keep_probs = np.exp(start_intens[0:].reshape(-1) + 1e-5)
            keep_probs = keep_probs / keep_probs.sum()
        else:
            raise NotImplementedError()

        # Keep indices
        # Add root
        ind_samples = np.random.choice(
            keep_inds, size=num_to_keep, replace=False, p=keep_probs
        )
        keep_indices = set(ind_samples)
        for ind, frag_tuple in enumerate(frag_tuples):
            new_tuple = frag_tuple
            ## Scale
            if np.random.random() < self.inten_prob:
                old_inten = new_tuple[3]
                inten_scalar = max(np.random.normal(loc=1), 0)
                new_inten = old_inten * inten_scalar
                new_tuple = (
                    new_tuple[0],
                    new_tuple[1],
                    new_tuple[2],
                    new_inten,
                    new_tuple[4],
                )
            else:
                new_inten = frag_tuple[-2]

            # Remove
            # Add in rescaled tuple only if > remove prob _or_ root of tree (ms1)
            if ind in keep_indices:
                keep_tuples.append(new_tuple)
            else:
                continue

            if new_inten > max_inten:
                max_inten = new_inten

        frag_tuples = keep_tuples

        # Rescale intensities by new max intensity
        new_tuples = []
        for frag_tuple in frag_tuples:
            new_tuple = (
                frag_tuple[0],
                frag_tuple[1],
                frag_tuple[2],
                frag_tuple[3] / max_inten,
                frag_tuple[4],
            )
            new_tuples.append(new_tuple)
        frag_tuples = new_tuples

        peak_dict["frags"] = frag_tuples
        return peak_dict

    def _featurize(
        self, spec: data.Spectra, train_mode: bool = False, **kwargs
    ) -> Dict:
        """featurize.

        Args:
            spec (Spectra)

        """
        spec_name = spec.get_spec_name()
        # Return get_peak_formulas output
        peak_dict = self._get_peak_dict(spec)

        # Augment peak dict with chem formulae
        peak_dict_new = {}
        for k, v in peak_dict.items():
            if k == "root_form":
                peak_dict_new[k] = v
                continue

            new_items = []
            for j in v:
                form_vec = utils.formula_to_dense_mass(j[2])
                new_items.append((*j, form_vec))
            peak_dict_new[k] = new_items

        peak_dict = peak_dict_new

        if train_mode and self.augment_data:
            # Only augment certain select peaks
            augment_peak = np.random.random() < self.augment_prob
            if augment_peak:
                peak_dict = self.augment_peak_dict(peak_dict)

        root = peak_dict["root_form"]
        form_vec, inten_vec, type_vec, mz_vec = [], [], [], []

        ## Only add in fragments for this model!
        ind, k = (0, "frags")
        peak_dict_vals = peak_dict.get(k, [])
        type_vec = len(peak_dict_vals) * [ind]

        if len(peak_dict_vals) > 0:
            _, mz_vec, forms, inten_vec, form_vec = zip(*peak_dict_vals)
            inten_vec = list(inten_vec)
            form_vec = list(form_vec)
            mz_vec = list(mz_vec)

        if self.use_cls:
            if self.cls_type == "ms1":
                cls_form = utils.formula_to_dense_mass(root)
                cls_ind = self.cat_types.get("cls")
                inten_vec.append(1.0)
                type_vec.append(cls_ind)
                form_vec.append(cls_form)
                mz_vec.append(cls_form[-1])
            elif self.cls_type == "zeros":
                cls_form = np.zeros_like(form_vec[0])
                cls_ind = self.cat_types.get("cls")
                inten_vec.append(0.0)
                type_vec.append(cls_ind)
                form_vec.append(cls_form)
                mz_vec.append(0)
            else:
                raise NotImplementedError()
        # Featurize all formulae
        form_vec = np.vstack(form_vec) / utils.NORM_VEC[None, :]
        mz_dict = dict(zip(mz_vec, form_vec))

        # Inten transforms
        inten_vec = np.array(inten_vec)
        if self.inten_transform == "float":
            pass
        elif self.inten_transform == "zero": 
            inten_vec = np.zeros_like(inten_vec)
        elif self.inten_transform == "log":
            inten_vec = np.log(inten_vec + 1e-5)
        elif self.inten_transform == "cat":
            bins = np.linspace(0, 1, self.num_inten_bins)
            # Digitize inten vec
            inten_vec = np.digitize(inten_vec, bins)
        else:
            raise NotImplementedError()

        ## Add in magma supervision!
        fingerprints = []
        if self.magma_aux_loss:
            magma_frag_file = self.spec_name_to_magma_file.get(spec_name)
            magma_df = magma_utils.read_magma_file(magma_frag_file)
            if len(magma_df) > 0:
                form_smiles = tuple(magma_df[["chemical_formula", "smiles"]].values)
                mass_to_tuples = dict(
                    zip(magma_df["mass_to_charge"].values, form_smiles)
                )
            else:
                mass_to_tuples = {}

            peak_smiles_list = []
            for mz, chem_form in mz_dict.items():
                chem_form_list, smiles_list = mass_to_tuples.get(mz, (None, None))
                if chem_form_list is None or len(chem_form_list) == 0:
                    peak_smiles_list.append(None)
                else:

                    # Figure out which of the chem formulae is _closest_
                    parent_form_ar = chem_form[None, :]

                    child_form_ar = np.vstack(
                        [utils.formula_to_dense_mass(i) for i in chem_form_list]
                    )
                    child_form_ar /= utils.NORM_VEC[None, :]

                    dist = np.abs(parent_form_ar - child_form_ar).sum(1)
                    argmin = np.argmin(dist)
                    smiles = smiles_list[argmin]
                    peak_smiles_list.append(smiles)

            # Get fingerprints
            fingerprints = []
            for smi in peak_smiles_list:
                if smi is None:
                    fingerprints.append(np.zeros(self.aug_nbits) - 1)
                else:
                    fingerprints.append(self._extract_fingerprint(smi))

            if len(fingerprints) > 0:
                fingerprints = np.vstack(fingerprints)
        out_dict = {
            "peak_type": np.array(type_vec),
            "form_vec": form_vec,
            "frag_intens": inten_vec,
            "name": spec_name,
            "magma_fps": fingerprints,
            "magma_aux_loss": self.magma_aux_loss,
        }
        return out_dict

    def _extract_fingerprint(self, smiles):
        """extract_fingerprints."""
        index = self.fp_index_obj.get(smiles)
        return self.fp_dataset[index]

    def featurize(self, spec: data.Spectra, train_mode=False, **kwargs) -> Dict:
        """Featurizer a single object"""

        encoded_obj = self._encode(spec)
        if train_mode:
            featurized = self._featurize(spec, train_mode=train_mode)
        else:
            if self.cache_featurizers:
                if encoded_obj in self.cache:
                    featurized = self.cache[encoded_obj]
                else:
                    featurized = self._featurize(spec)
                    self.cache[encoded_obj] = featurized
            else:
                featurized = self._featurize(spec)

        return featurized

    @staticmethod
    def collate_fn(input_list: List[dict]) -> Dict:
        """collate_fn.

        Input list of dataset outputs

        Args:
            input_list (List[data.Spectra]): Input list containing spectra to be
                collated
        Return:
            Dictionary containing batched results
        """
        # Determines the number of channels
        names = [j["name"] for j in input_list]
        peak_form_tensors = [torch.from_numpy(j["form_vec"]) for j in input_list]
        inten_tensors = [torch.from_numpy(j["frag_intens"]) for j in input_list]
        type_tensors = [torch.from_numpy(j["peak_type"]) for j in input_list]

        peak_form_lens = np.array([i.shape[0] for i in peak_form_tensors])
        max_len = np.max(peak_form_lens)
        padding_amts = max_len - peak_form_lens

        type_tensors = [
            torch.nn.functional.pad(i, (0, pad_len))
            for i, pad_len in zip(type_tensors, padding_amts)
        ]
        inten_tensors = [
            torch.nn.functional.pad(i, (0, pad_len))
            for i, pad_len in zip(inten_tensors, padding_amts)
        ]
        peak_form_tensors = [
            torch.nn.functional.pad(i, (0, 0, 0, pad_len))
            for i, pad_len in zip(peak_form_tensors, padding_amts)
        ]

        # Stack everything (bxd for root, bxp for others)
        type_tensors = torch.stack(type_tensors, dim=0).long()
        peak_form_tensors = torch.stack(peak_form_tensors, dim=0).float()

        inten_tensors = torch.stack(inten_tensors, dim=0).float()
        num_peaks = torch.from_numpy(peak_form_lens).long()

        # magma_fps
        use_magma = np.any([i["magma_aux_loss"] for i in input_list])
        magma_dict = {}
        if use_magma:
            magma_fingerprints = [i["magma_fps"] for i in input_list]

            # fingerprints: Batch x max num peaks x fingerprint dimension
            for i in range(len(magma_fingerprints)):
                padded_fp = np.zeros((max_len, magma_fingerprints[0].shape[1]))
                padded_fp[: magma_fingerprints[i].shape[0], :] = magma_fingerprints[i]
                magma_fingerprints[i] = padded_fp

            magma_fingerprints = np.stack(magma_fingerprints, axis=0)
            magma_fingerprints = torch.tensor(magma_fingerprints, dtype=torch.float)
            magma_dict["fingerprints"] = magma_fingerprints

            # Mask for where the spectra doesn't have a peak or has a peak but not a fingerprint
            fingerprint_sum = magma_fingerprints.sum(2)
            fingerprint_mask = fingerprint_sum > 0
            magma_dict["fingerprint_mask"] = fingerprint_mask

        return_dict = {
            "types": type_tensors,
            "form_vec": peak_form_tensors,
            "intens": inten_tensors,
            "names": names,
            "num_peaks": num_peaks,
        }

        return_dict.update(magma_dict)
        return return_dict


class PeakFormulaTest(PeakFormula):
    """PeakFormula with no Magma"""

    def __init__(self, **kwargs):
        kwargs["magma_aux_loss"] = False
        kwargs["add_forward_specs"] = False
        super().__init__(**kwargs)
