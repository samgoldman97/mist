import logging
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

import torch
from torch.utils.data.dataset import Dataset
from mist import utils


class BinnedDataset(Dataset):
    """BinnedDataset."""

    def __init__(
        self,
        df,
        data_dir,
        num_bins,
        num_workers=0,
        upper_limit=1000,
        fingerprinter=None,
        **kwargs,
    ):
        """__init__.

        Args:
            df:
            data_dir:
            num_bins:
            num_workers:
            upper_limit:
            fingerprinter:
            kwargs:
        """
        self.df = df
        self.num_bins = num_bins
        self.num_workers = num_workers
        self.upper_limit = upper_limit
        self.bins = np.linspace(0, self.upper_limit, self.num_bins)
        self.fingerprinter = fingerprinter

        # Read in all molecules
        self.smiles = self.df["smiles"].values
        if fingerprinter.fp_type == "csi":
            # Get weights
            if self.num_workers == 0:
                self.fps = [fingerprinter.get_fp_smi_wt(i) for i in self.smiles]
            else:

                def get_inchikey_wt(smiles):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return None, None
                    ikey = Chem.MolToInchiKey(mol)
                    wt = MolWt(mol)
                    return ikey, wt

                out_tuples = utils.chunked_parallel(
                    self.smiles,
                    get_inchikey_wt,
                    chunks=100,
                    max_cpu=self.num_workers,
                    timeout=4000,
                    max_retries=3,
                )
                # Get weights, inchikeys
                inchikeys, wts = zip(*out_tuples)

                # Get fingerprints (in singular)
                fps = [fingerprinter.get_fp_inchikey(i) for i, _ in out_tuples]
                self.fps = list(zip(fps, wts))
        else:
            if self.num_workers == 0:
                self.fps = [fingerprinter.get_fp_smi_wt(i) for i in self.smiles]
            else:
                self.fps = utils.chunked_parallel(
                    self.smiles,
                    fingerprinter.get_fp_smi_wt,
                    chunks=100,
                    max_cpu=self.num_workers,
                    timeout=4000,
                    max_retries=3,
                )
        # Extract
        fps, weights = zip(*[(i, j) for i, j in self.fps])
        self.fps = fps
        self.weights = np.vstack(weights).squeeze()

        # Read in all specs
        self.spec_names = self.df["spec"].values

        spec_name_df = data_dir / "sirius_outputs/summary_statistics/summary_df.tsv"
        spec_name_df = pd.read_csv(spec_name_df, sep="\t")
        spec_name_to_tsv = dict(spec_name_df[["spec_name", "spec_file"]].values)

        def process_spec_file(
            spec_name, num_bins: int = num_bins, upper_limit: int = upper_limit
        ):
            """process_spec_file."""

            spec_file = spec_name_to_tsv.get(spec_name)
            if spec_file is None:
                return None

            spec_tbl = pd.read_csv(spec_file, sep="\t")
            if (spec_tbl["intensity"].max() == 0) or len(spec_tbl) == 0:
                return None

            formulae, inten = zip(*spec_tbl[["formula", "intensity"]].values)
            masses = [utils.formula_mass(i) for i in formulae]

            # Shape 1 x num peaks x 2
            spectrum = np.vstack([masses, inten]).transpose(1, 0)[None, :, :]
            binned = utils.bin_spectra(spectrum, num_bins, upper_limit)
            normed = utils.norm_spectrum(binned)
            avged = normed.mean(0)
            return avged

        logging.info(f"Iterating with {self.num_workers} workers")
        if self.num_workers == 0:
            spec_outputs = [
                process_spec_file(spec_name) for spec_name in self.spec_names
            ]
        else:
            spec_outputs = utils.chunked_parallel(
                self.spec_names,
                process_spec_file,
                chunks=100,
                max_cpu=self.num_workers,
                timeout=4000,
                max_retries=3,
            )
        # Filter out where nothing is None
        zipped = [
            (spec, fp, weight, name)
            for spec, fp, weight, name in zip(
                spec_outputs, self.fps, self.weights, self.spec_names
            )
            if fp is not None and spec is not None
        ]

        self.spec_ars, self.fps, self.weights, self.spec_names = zip(*zipped)
        self.spec_ars = np.array(self.spec_ars)
        self.spec_fps = np.vstack(self.fps)
        self.spec_weights = np.array(self.weights)

    def __len__(self):
        return len(self.spec_names)

    def __getitem__(self, idx: int):
        name = self.spec_names[idx]
        ar = self.spec_ars[idx]
        fp = self.fps[idx]
        full_weight = self.weights[idx]
        outdict = {
            "name": name,
            "binned": ar,
            "full_weight": full_weight,
            "fp": fp,
        }
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return BinnedDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["name"] for j in input_list]
        spec_ars = [j["binned"] for j in input_list]
        fp_ars = [j["fp"] for j in input_list]
        full_weight = [j["full_weight"] for j in input_list]

        # Now pad everything else to the max channel dim
        spectra_tensors = torch.stack([torch.tensor(spectra) for spectra in spec_ars])
        fp_tensors = torch.stack([torch.tensor(fp) for fp in fp_ars])
        full_weight = torch.FloatTensor(full_weight)
        return_dict = {
            "spectra": spectra_tensors,
            "fps": fp_tensors,
            "names": names,
            "full_weight": full_weight,
        }
        return return_dict


class MolDataset(Dataset):
    """MolDataset."""

    def __init__(self, smiles, num_workers: int = 0, fingerprinter=None, **kwargs):
        """__init__.

        Args:
            smiles:
            num_workers (int): num_workers
            fingerprinter:
            kwargs:
        """

        self.smiles = smiles
        self.num_workers = num_workers
        self.fingerprinter = fingerprinter

        # Read in all molecules
        if fingerprinter.fp_type == "csi":
            # Get weights
            if self.num_workers == 0:
                self.fps = [fingerprinter.get_fp_smi_wt(i) for i in self.smiles]
            else:

                def get_inchikey_wt(smiles):
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return None, None
                    ikey = Chem.MolToInchiKey(mol)
                    wt = MolWt(mol)
                    return ikey, wt

                out_tuples = utils.chunked_parallel(
                    self.smiles,
                    get_inchikey_wt,
                    chunks=100,
                    max_cpu=self.num_workers,
                    timeout=4000,
                    max_retries=3,
                )
                # Get weights, inchikeys
                inchikeys, wts = zip(*out_tuples)

                # Get fingerprints (in singular)
                fps = [fingerprinter.get_fp_inchikey(i) for i, _ in out_tuples]
                self.fps = list(zip(fps, wts))
        else:
            if self.num_workers == 0:
                self.fps = [fingerprinter.get_fp_smi_wt(i) for i in self.smiles]
            else:
                self.fps = utils.chunked_parallel(
                    self.smiles,
                    fingerprinter.get_fp_smi_wt,
                    chunks=100,
                    max_cpu=self.num_workers,
                    timeout=4000,
                    max_retries=3,
                )
        # Extract
        fps, weights, smiles = zip(
            *[
                (i, j, smi)
                for (i, j), smi in zip(self.fps, self.smiles)
                if i is not None
            ]
        )
        self.fps = np.vstack(fps)
        self.smiles = np.array(smiles)
        self.weights = np.vstack(weights).squeeze()

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx: int):
        smi = self.smiles[idx]
        fp = self.fps[idx]
        full_weight = self.weights[idx]
        outdict = {"smi": smi, "fp": fp, "full_weight": full_weight}
        return outdict

    @classmethod
    def get_collate_fn(cls):
        return MolDataset.collate_fn

    @staticmethod
    def collate_fn(input_list):
        """collate_fn"""
        names = [j["smi"] for j in input_list]
        fp_ars = [j["fp"] for j in input_list]
        full_weight = [j["full_weight"] for j in input_list]

        # Now pad everything else to the max channel dim
        fp_tensors = torch.stack([torch.tensor(fp) for fp in fp_ars])
        full_weight = torch.FloatTensor(full_weight)
        return_dict = {
            "fps": fp_tensors,
            "names": names,
            "full_weight": full_weight,
        }
        return return_dict
