""" data.py """
import logging
from typing import Optional
import re

from rdkit import Chem
from rdkit.Chem import Descriptors
from mist import utils


class Spectra(object):
    """

    Object to store all spectra associated with a *single* molecule.

    Attributes:
        spectra: List of the actual spectrum containing N x 2 numpy array
        meta: metadata dictionary associated with this spectrum,most likely of
            the form {"collision": "${collision energy}"}
        compound: backlink to the compound containing the spectrum, containing
            more metadata (including the molecular structure as InChI/SMILES)
        num_spectr: Num spectra associated with this mol
    """

    def __init__(
        self,
        spectra_name: str = "",
        spectra_file: str = "",
        spectra_formula: str = "",
        **kwargs,
    ):
        """__init__.

        Args:
            spectra_name (str): Name of spectra to store as a property. Useful
                for splitters
            spectra_file (str): Location of spec file
            spectra_formula (str): Chemical formula
            **kwargs
        """

        self.spectra_name = spectra_name
        self.spectra_file = spectra_file
        self.formula = spectra_formula

        ##
        self._is_loaded = False
        self.parentmass = None
        self.num_spectra = None
        self.meta = None
        self.spectrum_names = None
        self.spectra = None

    def _load_spectra(self):
        """Load the spectra from files"""
        meta, spectrum_tuples = utils.parse_spectra(self.spectra_file)

        self.meta = meta
        self.parentmass = None
        for parent_kw in ["parentmass", "PEPMASS"]:
            self.parentmass = self.meta.get(parent_kw, None)
            if self.parentmass is not None:
                break

        if self.parentmass is None:
            logging.info(f"Unable to find precursor mass for {self.spectrum_name}")
            self.parentmass = 0
        else:
            self.parentmass = float(self.parentmass)

        # Store all the spectrum names (e.g., e.v.) and spectra arrays
        self.spectrum_names, self.spectra = zip(*spectrum_tuples)
        self.num_spectra = len(self.spectra)
        self._is_loaded = True

    def get_spec_name(self, **kwargs):
        """get_spec_name."""
        return self.spectra_name

    def get_spec(self, **kwargs):
        """get_spec."""
        if not self._is_loaded:
            self._load_spectra()

        return self.spectra

    def get_meta(self, **kwargs):
        """get_meta."""
        if not self._is_loaded:
            self._load_spectra()
        return self.meta

    def get_spectra_formula(self):
        """Get spectra formula."""
        return self.formula


class Mol(object):
    """
    Object to store a compound, including possibly multiple mass spectrometry
    spectra.
    """

    def __init__(
        self,
        mol: Chem.Mol,
        smiles: Optional[str] = None,
        inchikey: Optional[str] = None,
        mol_formula: Optional[str] = None,
    ):
        """__init__.
        Args:
            mol (Chem.Mol): input molecule to store
            smiles (Optional[str]): Input SMILES string
            inchikey (Optional[str]): Input inchikey string
            mol_formula (Optional[str]): Molecule formula
        """
        self.mol = mol

        self.smiles = smiles
        if self.smiles is None:
            # Isomeric smiles handled in preprocessing
            self.smiles = Chem.MolToSmiles(mol)

        self.inchikey = inchikey
        if self.inchikey is None:
            self.inchikey = Chem.MolToInchiKey(mol)

        self.mol_formula = mol_formula
        if self.mol_formula is None:
            self.mol_formula = utils.uncharged_formula(self.mol, mol_type="mol")
        self.num_hs = None

    @classmethod
    def MolFromInchi(cls, inchi: str, **kwargs):
        """__init__.

        Args:
            inchi (str): inchi string

        """
        mol = Chem.MolFromInchi(inchi)

        # Catch exception
        if mol is None:
            return None

        return cls(mol=mol, smiles=None, **kwargs)

    @classmethod
    def MolFromSmiles(cls, smiles: str, **kwargs):
        """__init__.

        Args:
            smiles (str): SMILES string

        """
        if not smiles or isinstance(smiles, float):
            smiles = ""

        mol = Chem.MolFromSmiles(smiles)
        # Catch exception
        if mol is None:
            return None

        return cls(mol=mol, smiles=smiles, **kwargs)

    def get_smiles(self) -> str:
        return self.smiles

    def get_inchikey(self) -> str:
        return self.inchikey

    def get_molform(self) -> str:
        return self.mol_formula

    def get_num_hs(self):
        """get_num_hs."""
        if self.num_hs is None:
            num = re.findall("H([0-9]*)", self.mol_formula)
            if num is None:
                out_num_hs = 0
            else:
                if len(num) == 0:
                    out_num_hs = 0
                elif len(num) == 1:
                    num = num[0]
                    out_num_hs = 1 if num == "" else int(num)
                else:
                    raise ValueError()
            self.num_hs = out_num_hs
        else:
            out_num_hs = self.num_hs

        return out_num_hs

    def get_mol_mass(self):
        return Descriptors.MolWt(self.mol)

    def get_rdkit_mol(self) -> Chem.Mol:
        return self.mol
