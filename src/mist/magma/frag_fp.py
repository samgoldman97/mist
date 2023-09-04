""" frag_fp.py

Define a fingerprint function over a fragment graph

"""

import hashlib
import numpy as np

def hash_fn(x):
    """ hash_fn. 

        We want 64 bytes to fit into c type long, so cap at 15
        Each hex char is 4 bits; 16 * 4 = 64 bit

    """
    return int(hashlib.md5(x.encode()).hexdigest()[:15], 16)


def fp_from_frag(
    frag: int,
    atom_symbols: list,
    bonded_atoms: list,
    bonded_types: list,
    bonds_per_atom: list,
    radius: int = 3,
    modulo: int = 2048,
) -> dict:
    """_summary_

    Args:
        frag (int): _description_
        atom_symbols (list): _description_
        bonded_atoms (list): _description_
        bonded_types (list): _description_
        bonds_per_atom (list): _description_
        radius (int, optional): _description_. Defaults to 3.
        modulo (int, optional): _description_. Defaults to 2048.

    Returns:
        dict: _description_
    """
    # Define new_to_old that's a litle too big
    num_atoms = len(atom_symbols)
    all_hashes = np.zeros((num_atoms, radius), dtype=np.int64)
    old_to_new = np.zeros(num_atoms, dtype=np.int64)
    new_to_old = np.zeros(num_atoms, dtype=np.int64)

    # All hashes has dim radius x atom symbols
    # Define the 0 case strings
    for ct in range(radius):
        new_atom_pos = 0
        for atom, atom_symbol in enumerate(atom_symbols):
            atombit = 1 << atom

            # Get all neighobrs
            if not atombit & frag:
                continue

            if ct == 0:
                old_to_new[atom] = new_atom_pos
                new_to_old[new_atom_pos] = atom
                all_hashes[new_atom_pos, ct] = hash_fn(atom_symbol)
            else:
                cur_hash = all_hashes[new_atom_pos, ct - 1]

                # Get local neighbors
                neighbor_labels = []
                num_bonds = bonds_per_atom[atom]
                for targind, bondtype in zip(
                    bonded_atoms[atom, :num_bonds], bonded_types[atom, :num_bonds]
                ):
                    targind = int(targind)
                    targbit = 1 << targind

                    if not targbit & frag:
                        continue

                    targ_atom_pos = old_to_new[targind]
                    targhash = all_hashes[targ_atom_pos, ct - 1]
                    neighbor_label = f"{bondtype}_{targhash}"
                    neighbor_labels.append(neighbor_label)
                neighbor_labels = "".join(sorted(neighbor_labels))
                new_hash_str = f"{cur_hash}_{neighbor_labels}"
                new_hash = hash_fn(new_hash_str)
                all_hashes[new_atom_pos, ct] = new_hash
            new_atom_pos += 1

    if modulo is not None:
        all_hashes = all_hashes % modulo

    flat_list = np.unique(all_hashes[:new_atom_pos].flatten())
    return {
        "all_hashes": all_hashes,
        "flat_list": flat_list,
        "new_to_old": new_to_old,
        "old_to_new": old_to_new,
        "new_atom_pos": new_atom_pos,
    }
