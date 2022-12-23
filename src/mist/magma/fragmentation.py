"""fragmentation.py

Code snippets taken from the MAGMa github project

https://github.com/NLeSC/MAGMa

"""


import numpy
from rdkit import Chem

typew = {
    Chem.rdchem.BondType.names["AROMATIC"]: 3.0,
    Chem.rdchem.BondType.names["DOUBLE"]: 2.0,
    Chem.rdchem.BondType.names["TRIPLE"]: 3.0,
    Chem.rdchem.BondType.names["SINGLE"]: 1.0,
}
heterow = {False: 2, True: 1}
missingfragmentpenalty = 10


mims = {
    "H": 1.0078250321,
    "C": 12.0000000,
    "N": 14.0030740052,
    "O": 15.9949146221,
    "F": 18.99840320,
    "Na": 22.9897692809,
    "P": 30.97376151,
    "S": 31.97207069,
    "Cl": 34.96885271,
    "K": 38.96370668,
    "Br": 78.9183376,
    "I": 126.904468,
    "Si": 28.0855,
    "B": 10.811,
    "Se": 78.97,
    "Fe": 55.845,
    "Co": 58.933,
}

# Mass of hydrogen atom
Hmass = mims["H"]
elmass = 0.0005486

ionmasses = {
    1: {
        "+H": mims["H"],
        "+NH4": mims["N"] + 4 * mims["H"],
        "+Na": mims["Na"],
        "+K": mims["K"],
    },
    -1: {"-H": -mims["H"], "+Cl": mims["Cl"]},
}


class FragmentEngine(object):
    def __init__(
        self,
        smiles,
        max_broken_bonds,
        max_water_losses,
        ionisation_mode,
        skip_fragmentation,
        molcharge,
    ):
        try:
            # self.mol = Chem.MolFromMolBlock(str(mol))
            # self.mol = Chem.MolFromSmiles(smiles)
            self.mol = Chem.MolFromSmiles(smiles)
            self.accept = True
            self.natoms = self.mol.GetNumAtoms()
        except:
            self.accept = False
            return
        self.max_broken_bonds = max_broken_bonds
        self.max_water_losses = max_water_losses
        self.ionisation_mode = ionisation_mode
        self.skip_fragmentation = skip_fragmentation
        self.molcharge = molcharge
        self.atom_masses = []
        self.atomHs = []
        self.neutral_loss_atoms = []
        self.bonded_atoms = []  # [[list of atom numbers]]
        self.bonds = set([])
        self.bondscore = {}
        self.new_fragment = 0
        self.template_fragment = 0
        self.fragment_masses = ((max_broken_bonds + max_water_losses) * 2 + 1) * [0]
        self.fragment_info = [[0, 0, 0]]
        self.avg_score = None

        for x in range(self.natoms):
            self.bonded_atoms.append([])
            atom = self.mol.GetAtomWithIdx(x)
            self.atomHs.append(atom.GetNumImplicitHs() + atom.GetNumExplicitHs())
            self.atom_masses.append(mims[atom.GetSymbol()] + Hmass * (self.atomHs[x]))
            if (
                atom.GetSymbol() == "O"
                and self.atomHs[x] == 1
                and len(atom.GetBonds()) == 1
            ):
                self.neutral_loss_atoms.append(x)
            if (
                atom.GetSymbol() == "N"
                and self.atomHs[x] == 2
                and len(atom.GetBonds()) == 1
            ):
                self.neutral_loss_atoms.append(x)
        for bond in self.mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self.bonded_atoms[a1].append(a2)
            self.bonded_atoms[a2].append(a1)
            bondbits = 1 << a1 | 1 << a2
            bondscore = (
                typew[bond.GetBondType()]
                * heterow[
                    bond.GetBeginAtom().GetSymbol() != "C"
                    or bond.GetEndAtom().GetSymbol() != "C"
                ]
            )
            self.bonds.add(bondbits)
            self.bondscore[bondbits] = bondscore

    def extend(self, atom):
        for a in self.bonded_atoms[atom]:
            atombit = 1 << a
            if atombit & self.template_fragment and not atombit & self.new_fragment:
                self.new_fragment = self.new_fragment | atombit
                self.extend(a)

    def generate_fragments(self):
        frag = (1 << self.natoms) - 1
        all_fragments = set([frag])
        total_fragments = set([frag])
        current_fragments = set([frag])
        new_fragments = set([frag])
        self.add_fragment(frag, self.calc_fragment_mass(frag), 0, 0)

        if self.skip_fragmentation:
            self.convert_fragments_table()
            return len(self.fragment_info)

        # generate fragments for max_broken_bond steps
        for step in range(self.max_broken_bonds):
            # loop over all fragments to be fragmented
            for fragment in current_fragments:
                # loop over all atoms
                for atom in range(self.natoms):
                    # in the fragment
                    if (1 << atom) & fragment:
                        # remove the atom
                        self.template_fragment = fragment ^ (1 << atom)
                        list_ext_atoms = set([])
                        extended_fragments = set([])
                        # find all its neighbor atoms
                        for a in self.bonded_atoms[atom]:
                            # present in the fragment
                            if (1 << a) & self.template_fragment:
                                list_ext_atoms.add(a)
                        # in case of one bonded atom, the new fragment is the remainder of the old fragment
                        if len(list_ext_atoms) == 1:
                            extended_fragments.add(self.template_fragment)
                        else:
                            # otherwise extend each neighbor atom to a complete fragment
                            for a in list_ext_atoms:
                                # except when deleted atom is in a ring and a previous extended
                                # fragment already contains this neighbor atom, then
                                # calculate fragment only once
                                for frag in extended_fragments:
                                    if (1 << a) & frag:
                                        break
                                else:
                                    # extend atom to complete fragment
                                    self.new_fragment = 1 << a
                                    self.extend(a)
                                    extended_fragments.add(self.new_fragment)
                        for frag in extended_fragments:
                            # add extended fragments, if not yet present, to the collection
                            if frag not in all_fragments:
                                all_fragments.add(frag)
                                bondbreaks, score = self.score_fragment(frag)
                                if bondbreaks <= self.max_broken_bonds and score < (
                                    missingfragmentpenalty + 5
                                ):
                                    new_fragments.add(frag)
                                    total_fragments.add(frag)
                                    self.add_fragment(
                                        frag,
                                        self.calc_fragment_mass(frag),
                                        score,
                                        bondbreaks,
                                    )
            current_fragments = new_fragments
            new_fragments = set([])
        # number of OH losses
        for step in range(self.max_water_losses):
            # loop of all fragments
            for fi in self.fragment_info:
                # on which to apply neutral loss rules
                if fi[2] == self.max_broken_bonds + step:
                    fragment = fi[0]
                    # loop over all atoms in the fragment
                    for atom in self.neutral_loss_atoms:
                        if (1 << atom) & fragment:
                            frag = fragment ^ (1 << atom)
                            # add extended fragments, if not yet present, to the collection
                            if frag not in total_fragments:
                                total_fragments.add(frag)
                                bondbreaks, score = self.score_fragment(frag)
                                if score < (missingfragmentpenalty + 5):
                                    self.add_fragment(
                                        frag,
                                        self.calc_fragment_mass(frag),
                                        score,
                                        bondbreaks,
                                    )
        self.convert_fragments_table()
        return len(self.fragment_info)

    def score_fragment(self, fragment):
        score = 0
        bondbreaks = 0
        for bond in self.bonds:
            if 0 < (fragment & bond) < bond:
                score += self.bondscore[bond]
                bondbreaks += 1
        if score == 0:
            print("score=0: ", fragment, bondbreaks)
        return bondbreaks, score

    def score_fragment_rel2parent(self, fragment, parent):
        score = 0
        for bond in self.bonds:
            if 0 < (fragment & bond) < (bond & parent):
                score += self.bondscore[bond]
        return score

    def calc_fragment_mass(self, fragment):
        fragment_mass = 0.0
        for atom in range(self.natoms):
            if fragment & (1 << atom):
                fragment_mass += self.atom_masses[atom]
        return fragment_mass

    def add_fragment(self, fragment, fragmentmass, score, bondbreaks):
        mass_range = (
            (self.max_broken_bonds + self.max_water_losses - bondbreaks) * [0]
            + list(
                numpy.arange(
                    -bondbreaks + self.ionisation_mode * (1 - self.molcharge),
                    bondbreaks + self.ionisation_mode * (1 - self.molcharge) + 1,
                )
                * Hmass
                + fragmentmass
            )
            + (self.max_broken_bonds + self.max_water_losses - bondbreaks) * [0]
        )
        if bondbreaks == 0:
            # make sure that fragmentmass is included
            mass_range[
                self.max_broken_bonds + self.max_water_losses - self.ionisation_mode
            ] = fragmentmass
        self.fragment_masses += mass_range
        self.fragment_info.append([fragment, score, bondbreaks])

    def convert_fragments_table(self):
        self.fragment_masses_np = numpy.array(self.fragment_masses).reshape(
            len(self.fragment_info),
            (self.max_broken_bonds + self.max_water_losses) * 2 + 1,
        )

    def calc_avg_score(self):
        self.avg_score = numpy.average(self.scores)

    def get_avg_score(self):
        return self.avg_score

    def find_fragments(self, mass, parent, precision, mz_precision_abs):
        result = numpy.where(
            numpy.where(
                self.fragment_masses_np
                < max(mass * precision, mass + mz_precision_abs),
                self.fragment_masses_np,
                0,
            )
            > min(mass / precision, mass - mz_precision_abs)
        )
        fragment_set = []
        for i in range(len(result[0])):
            fid = result[0][i]
            fragment_set.append(
                self.fragment_info[fid]
                + [
                    self.fragment_masses_np[fid][
                        self.max_broken_bonds
                        + self.max_water_losses
                        - self.ionisation_mode * (1 - self.molcharge)
                    ]
                ]
                + [
                    self.ionisation_mode * (1 - self.molcharge)
                    + result[1][i]
                    - self.max_broken_bonds
                    - self.max_water_losses
                ]
            )
        return fragment_set

    def get_fragment_info(self, fragment, deltaH):
        atomlist = []
        elements = {
            "C": 0,
            "H": 0,
            "N": 0,
            "O": 0,
            "F": 0,
            "P": 0,
            "S": 0,
            "Cl": 0,
            "Br": 0,
            "I": 0,
            "Si": 0,
            "B": 0,
            "Se": 0,
            "Fe": 0,
            "Co": 0,
        }
        for atom in range(self.natoms):
            if (1 << atom) & fragment:
                atomlist.append(atom)
                elements[self.mol.GetAtomWithIdx(atom).GetSymbol()] += 1
                elements["H"] += self.atomHs[atom]
        formula = ""
        for el in (
            "C",
            "H",
            "N",
            "O",
            "F",
            "P",
            "S",
            "Cl",
            "Br",
            "I",
            "Si",
            "B",
            "Se",
            "Fe",
            "Co",
        ):
            nel = elements[el]
            if nel > 0:
                formula += el
            if nel > 1:
                formula += str(nel)
        atomstring = ",".join(str(a) for a in atomlist)
        return atomstring, atomlist, formula, fragment2smiles(self.mol, atomlist)

    def get_natoms(self):
        return self.natoms

    def accepted(self):
        return self.accept


def fragment2smiles(mol, atomlist):
    emol = Chem.EditableMol(mol)
    for atom in reversed(range(mol.GetNumAtoms())):
        if atom not in atomlist:
            emol.RemoveAtom(atom)
    frag = emol.GetMol()
    return Chem.MolToSmiles(frag)
