""" run_magma.py

Entry point into running magma program

"""
import sys
import argparse
import logging
import json
from pathlib import Path
import numpy as np
import copy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from rdkit import RDLogger

# Custom import
import mist.utils as utils
import mist.magma.fragmentation as fragmentation
import mist.magma.frag_fp as frag_fp


FRAGMENT_ENGINE_PARAMS = {"max_broken_bonds": 6, "max_tree_depth": 3}


def magma_augmentation(
    spec_file: Path,
    output_dir: Path,
    spec_to_smiles: dict,
    spec_to_adduct: dict,
    max_peaks: int,
    ppm_diff: float = 10,
    debug: bool = False,
):
    """magma_augmentation.

    Args:
        spec_file (Path): spec_file
        output_dir (Path): output_dir
        spec_to_smiles (dict): spec_to_smiles
        spec_to_adduct (dict): Spec to adduct
        max_peaks (int): max_peaks
        ppm_diff (float): Max diff ppm
        debug (bool)
    """
    spectra_name = spec_file.stem
    tsv_dir = output_dir / "magma_tsv"
    tree_dir = output_dir / "magma_tree"
    tsv_dir.mkdir(exist_ok=True)
    tree_dir.mkdir(exist_ok=True)
    tsv_filename = tsv_dir / f"{spectra_name}.magma"

    meta, spectras = utils.parse_spectra(spec_file)

    spectra_smiles = spec_to_smiles.get(spectra_name, None)
    spectra_adduct = spec_to_adduct.get(spectra_name, None)

    # Step 1 - Generate fragmentations inside fragmentation engine
    fe = fragmentation.FragmentEngine(mol_str=spectra_smiles, **FRAGMENT_ENGINE_PARAMS)

    # Outside try except loop
    if debug:
        fe.generate_fragments()
    else:
        try:
            fe.generate_fragments()
        except:
            print(f"Error with generating fragments for spec {spectra_name}")
            return

    # Step 2: Process spec and get comparison points
    # Read in file and filter it down
    spectra = utils.process_spec_file(
        meta, spectras, max_peaks=max_peaks, max_inten=0.001
    )

    if spectra is None:
        print(f"Error with generating fragments for spec {spectra_name}")
        return

    s_m, s_i = spectra[:, 0], spectra[:, 1]

    # Correct for s_m by subtracting it
    adjusted_m = s_m - utils.ion_to_mass[spectra_adduct]

    # Step 3: Make all assignments
    frag_hashes, frag_inds, shift_inds, masses, scores = fe.get_frag_masses()

    # Argsort by bond breaking scores
    # Lower bond scores are better
    new_order = np.argsort(scores)
    frag_hashes, frag_inds, shift_inds, masses, scores = (
        frag_hashes[new_order],
        frag_inds[new_order],
        shift_inds[new_order],
        masses[new_order],
        scores[new_order],
    )
    ppm_diffs = (
        np.abs(masses[None, :] - adjusted_m[:, None]) / adjusted_m[:, None] * 1e6
    )

    # Need to catch _all_ equivalent fragments
    # How do I remove the symmetry problem at each step and avoid branching
    # trees for the same examples??
    min_ppms = ppm_diffs.min(-1)
    is_min = min_ppms[:, None] == ppm_diffs
    peak_mask = min_ppms < ppm_diff

    # Step 4: Make exports
    # Now collect all inds and results
    # Also record a map from hash, hshift to the peak_info
    tsv_export_list = []
    for ind, was_assigned in enumerate(peak_mask):
        new_entry = {
            "mz_observed": s_m[ind],
            "mz_corrected": adjusted_m[ind],
            "inten": s_i[ind],
            "ppm_diff": "",
            "frag_inds": "",
            "frag_mass": "",
            "frag_h_shift": "",
            "frag_base_form": "",
            "frag_hashes": "",
        }
        if was_assigned:
            # Find all the fragments that have min ppm tolerance
            matched_peaks = is_min[ind]
            min_inds = np.argwhere(matched_peaks).flatten()

            # Get min score for this assignment
            min_score = np.min(scores[min_inds])

            # Filter even further down to inds that have min score and min ppm
            min_score_ppm = min_inds[
                np.argwhere(scores[min_inds] == min_score).flatten()
            ]

            frag_inds_temp = [frag_inds[temp_ind] for temp_ind in min_score_ppm]
            frag_masses_temp = [masses[temp_ind] for temp_ind in min_score_ppm]
            frag_hashes_temp = [frag_hashes[temp_ind] for temp_ind in min_score_ppm]
            shift_inds_temp = [shift_inds[temp_ind] for temp_ind in min_score_ppm]
            frag_entries_temp = [
                fe.frag_to_entry[frag_hash] for frag_hash in frag_hashes_temp
            ]
            frag_forms_temp = [frag_entry["form"] for frag_entry in frag_entries_temp]

            # Randomly sample an index
            new_ind = np.random.choice(len(frag_inds_temp))

            # make_str = lambda x: ",".join([str(xx) for xx in x])
            def make_str(x):
                return str(x[new_ind])

            new_entry["ppm_diff"] = min_ppms[ind]
            new_entry["frag_inds"] = make_str(frag_inds_temp)
            new_entry["frag_hashes"] = make_str(frag_hashes_temp)
            new_entry["frag_mass"] = make_str(frag_masses_temp)
            new_entry["frag_h_shift"] = make_str(shift_inds_temp)
            new_entry["frag_base_form"] = make_str(frag_forms_temp)

            targ_frag = int(eval(new_entry["frag_inds"]))
            fp_frags = frag_fp.fp_from_frag(
                frag=targ_frag,
                atom_symbols=fe.atom_symbols_ar,
                bonded_atoms=fe.bonded_atoms_np,
                bonded_types=fe.bonded_types_np,
                bonds_per_atom=fe.num_bonds_np,
                radius=3,
                modulo=2048,
            )
            flat_list = ",".join([str(i) for i in fp_frags["flat_list"]])
            new_entry["frag_fp"] = flat_list
            # print(f"Exception on {spectra_name}")

            # Only add to list if it was assigned
            tsv_export_list.append(new_entry)

    df = pd.DataFrame(tsv_export_list)
    if len(df) == 0:
        # Define empty df with proper keys
        df = pd.DataFrame(
            columns=[
                "mz_observed",
                "mz_corrected",
                "inten",
                "ppm_diff",
                "frag_inds",
                "frag_mass",
                "frag_h_shift",
                "frag_base_form",
                "frag_hashes",
                "frag_fp",
            ]
        )

    # Only keep entries where
    df.sort_values(by="mz_observed", inplace=True)
    df = df.reset_index(drop=True)
    df.to_csv(tsv_filename, sep="\t", index=None)


def run_magma_augmentation(
    spectra_dir: str,
    output_dir: str,
    spec_labels: str,
    max_peaks: int,
    debug: bool = False,
    num_workers: int = 0,
):
    """_summary_

    Args:
        spectra_dir (str): _description_
        output_dir (str): _description_
        spec_labels (str): _description_
        max_peaks (int): _description_
        debug (bool, optional): _description_. Defaults to False.
        num_workers (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    logging.info("Create magma spectra files")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    ms_files = list(Path(spectra_dir).glob("*.ms"))

    # Read in spec to smiles
    df = pd.read_csv(spec_labels, sep="\t")
    spec_to_smiles = dict(df[["spec", "smiles"]].values)
    spec_to_adduct = dict(df[["spec", "ionization"]].values)

    # Run this over all files
    def partial_aug_safe(spec_file):
        return magma_augmentation(
            spec_file,
            output_dir,
            spec_to_smiles,
            spec_to_adduct,
            max_peaks=max_peaks,
            debug=debug,
        )

    if debug:
        [partial_aug_safe(i) for i in tqdm(ms_files[10040:100045])]
    else:
        utils.chunked_parallel(ms_files, partial_aug_safe, max_cpu=max(num_workers, 1))


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectra-dir",
        default="data/paired_data/canopus_train/spec_files",
        help="Directory where spectra are stored",
    )
    parser.add_argument(
        "--spec-labels",
        default="data/paired_data/canopus_train/labels.tsv",
        help="TSV Location containing spectra labels",
    )
    parser.add_argument(
        "--output-dir",
        default="data/paired_data/canopus_train/magma_outputs",
        help="Output directory to save MAGMA files",
    )
    parser.add_argument(
        "--max-peaks",
        default=20,
        help="Maximum number of peaks",
        type=int,
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        help="Maximum number of peaks",
        type=int,
    )
    parser.add_argument("--debug", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    # Define basic logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    RDLogger.DisableLog("rdApp.*")
    args = get_args()
    kwargs = args.__dict__
    run_magma_augmentation(**kwargs)
