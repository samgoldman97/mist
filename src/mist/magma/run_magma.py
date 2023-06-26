""" run_magma.py

Accept input processed spectra and make subformula peak assignments
accordingly.

"""
import logging
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from ast import literal_eval
import pickle
import sys

# Custom import
from mist.utils import chunked_parallel
from mist.magma.fragmentation import FragmentEngine
from mist.magma import magma_utils

# Define basic logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)


FRAGMENT_ENGINE_PARAMS = {
    "max_broken_bonds": 3,
    "max_water_losses": 1,
    "ionisation_mode": 1,
    "skip_fragmentation": 0,
    "molcharge": 0,
}


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectra-dir",
        default="data/paired_spectra/gnps2015/spec_files",
        help="Directory where spectra are stored",
    )
    parser.add_argument(
        "--spec-labels",
        default="data/paired_spectra/gnps2015/labels.tsv",
        help="TSV Location containing spectra labels",
    )
    parser.add_argument(
        "--output-dir",
        default="data/paired_spectra/gnps2015/magma_outputs",
        help="Output directory to save MAGMA files",
    )
    parser.add_argument(
        "--lowest-penalty-filter",
        action="store_true",
        help="If set, magma script will filter candidate smiles for each peak by the penalty score",
    )
    parser.add_argument(
        "--skip-magma",
        default=False,
        action="store_true",
        help="If true, assume magma files already generated",
    )
    parser.add_argument(
        "--constrain-formula",
        default=False,
        action="store_true",
        help="If true, constrain the chemical formula when calling on sirius",
    )
    parser.add_argument(
        "--workers", default=30, action="store", type=int, help="Num workers"
    )
    return parser.parse_args()


def _get_random_unique_formulas(candidate_chemical_formulas, candidate_smiles):
    """_get_random_unique_formulas."""
    unique_formulas = set(candidate_chemical_formulas)
    form_idx_dict = dict.fromkeys(unique_formulas)

    for i, form in enumerate(candidate_chemical_formulas):
        if form_idx_dict[form] is None:
            form_idx_dict[form] = [i]
        else:
            form_idx_dict[form].append(i)

    chosen_idxs = [np.random.choice(idxs, 1)[0] for form, idxs in form_idx_dict.items()]

    filtered_formulas = [candidate_chemical_formulas[i] for i in chosen_idxs]
    filtered_smiles = [candidate_smiles[i] for i in chosen_idxs]
    return filtered_formulas, filtered_smiles


def get_matching_fragment(
    fragment_df, mass_comparison_vector, lowest_penalty_filter: bool
):
    """get_matching_fragment.

    Compare frag

    Args:
        fragment_df
        mass_comparison_vec
        lowest_penalty_filter

    """
    # Step 1 - Determine and filter for fragments whose mass range cover the peak mass
    matched_fragments_df = fragment_df[mass_comparison_vector]

    # If no candidate fragments exist, exit function
    if matched_fragments_df.shape[0] == 0:
        return None

    # Step 2 - If multiple candidate substructures, filter for those with the lowest penalty scores
    if lowest_penalty_filter:
        if matched_fragments_df.shape[0] > 1:
            min_score = matched_fragments_df["score"].min()
            matched_fragments_df = matched_fragments_df[
                matched_fragments_df["score"] == min_score
            ]

    # Step 3 - Save all remaining candidate fragments
    matched_fragment_idxs = list(matched_fragments_df.index)

    return matched_fragment_idxs


def get_fragment_mass_range(fragment_engine, fragment_df, tolerance):
    """get_fragment_mass_range.

    Define min and max masses in the range that are available based upon
    hydrogen diffs.

    Args:
        fragment_engine: Fragment engine
        fragment_df: fragment_df
        tolerance: Tolerance

    """
    fragment_masses_np = fragment_engine.fragment_masses_np

    # Build a list of the min and max mass of each fragment
    fragment_mass_min_max = []

    for fragment_idx in range(fragment_masses_np.shape[0]):
        fragment_masses = fragment_masses_np[fragment_idx, :]

        if np.sum(fragment_masses) == 0:
            min_frag_mass = 0
            max_frag_mass = 0

        else:
            min_frag_mass = (
                fragment_masses[np.nonzero(fragment_masses)[0][0]] - tolerance
            )
            max_frag_mass = max(fragment_masses) + tolerance

        fragment_mass_min_max.append((min_frag_mass, max_frag_mass))

    fragment_mass_min_max = np.array(fragment_mass_min_max)
    fragment_df["min_mass"] = fragment_mass_min_max[:, 0]
    fragment_df["max_mass"] = fragment_mass_min_max[:, 1]

    return fragment_df


def peak_fragment_assignment_sirius(
    spectra,
    fragment_engine,
    constrain_formula=False,
    lowest_penalty_filter: bool = False,
    tolerance=1,
):
    """peak_fragment_assignment_sirius.

    Args:
        spectra: NP array of mz, inten
        fragment_engine: FragmentEngine
        constrain_formula: If true, constrain formula of output
        lowest_penalty_filter (bool): lowest_penalty_filter

    Returns:
        Dataframe with columns: mass/charge, intensity, fragment chemical formula, fragment smile-string
    """

    fragments_info = fragment_engine.fragment_info

    fragment_df = pd.DataFrame(
        fragment_engine.fragment_info, columns=["id", "score", "bond_breaks"]
    )
    fragment_df = get_fragment_mass_range(fragment_engine, fragment_df, tolerance)

    # Need to build comparison values here
    min_fragment_mass = fragment_df["min_mass"].values
    max_fragment_mass = fragment_df["max_mass"].values
    exact_masses = spectra["exactmass"].values

    mass_comparison_matrix = np.logical_and(
        exact_masses[None, :] >= min_fragment_mass[:, None],
        exact_masses[None, :] <= max_fragment_mass[:, None],
    )

    num_peaks = exact_masses.shape[0]
    formulae = spectra["formula"].values

    # Iterate over each peak to find a match
    for peak_idx in range(num_peaks):
        mass_comparison_vector = mass_comparison_matrix[:, peak_idx]

        sirius_pred_formula = formulae[peak_idx]

        matched_fragment_idxs = get_matching_fragment(
            fragment_df,
            mass_comparison_vector,
            lowest_penalty_filter=lowest_penalty_filter,
        )

        if matched_fragment_idxs is None:
            continue

        # Save selected fragments info
        candidate_chemical_formulas = []
        candidate_smiles = []
        for idx in matched_fragment_idxs:
            fragment_info = fragment_engine.get_fragment_info(fragments_info[idx][0], 0)

            if sirius_pred_formula == fragment_info[2] or not constrain_formula:
                candidate_chemical_formulas.append(fragment_info[2])
                candidate_smiles.append(fragment_info[3])

        if len(candidate_smiles) == 0:
            continue

        # Remove duplicate chemical formula randomly (to handle memory constraints)
        filtered_formulas, filtered_smiles = _get_random_unique_formulas(
            candidate_chemical_formulas, candidate_smiles
        )

        spectra.loc[peak_idx, "chemical_formula"] = str(filtered_formulas)
        spectra.loc[peak_idx, "smiles"] = str(filtered_smiles)

    return spectra


def magma_augmentation_sirius(
    spectra_name,
    spec_name_to_spec,
    spectra_dir,
    output_dir,
    lowest_penalty_filter,
    spec_to_smiles,
    constrain_formula,
):
    """ magma_augmentation_sirius. """
    ms_file_name = spec_name_to_spec.get(spectra_name)
    save_filename = (Path(output_dir) / spectra_name).with_suffix(".magma")

    # Parse spectra from tsv
    spec_df = pd.read_csv(ms_file_name, sep="\t")

    spectra_smiles = spec_to_smiles.get(spectra_name, None)
    (
        max_broken_bonds,
        max_water_losses,
        ionisation_mode,
        skip_fragmentation,
        molcharge,
    ) = FRAGMENT_ENGINE_PARAMS.values()

    # Step 1 - Load fragmentation engine
    try:
        engine = FragmentEngine(
            smiles=spectra_smiles,
            max_broken_bonds=max_broken_bonds,
            max_water_losses=max_water_losses,
            ionisation_mode=ionisation_mode,
            skip_fragmentation=skip_fragmentation,
            molcharge=molcharge,
        )
        engine.generate_fragments()
    except:
        logging.info(f"Error for spec {ms_file_name}")
        return None

    # Step 2 - Load MAGMa engine
    # Remove magma engine
    mol_spectra_df = peak_fragment_assignment_sirius(
        spec_df,
        engine,
        constrain_formula=constrain_formula,
        lowest_penalty_filter=lowest_penalty_filter,
    )

    # Step 4 - Save assignments
    save_filename = Path(save_filename)
    save_filename.parent.mkdir(exist_ok=True)
    mol_spectra_df.to_csv(save_filename, sep="\t")


def get_fingerprint_set(summary_df):
    """Return list of smiles, fingeprrints"""
    spec_name_to_magma_file = dict(summary_df[["spec_name", "magma_file"]].values)

    logging.info("Generate set of unique smiles")

    def extract_smiles_set(my_tuple):
        """Read all files"""
        spectra_name, magma_file_path = my_tuple
        try:
            fragment_labels_df = pd.read_csv(magma_file_path, index_col=0, sep="\t")
        except:
            logging.info(f"Skipping file: {magma_file_path} ")
            return set()

        if "smiles" not in fragment_labels_df.columns:
            return set()

        fragment_labels_df = magma_utils._convert_str_to_list(
            fragment_labels_df, "chemical_formula"
        )
        fragment_labels_df = magma_utils._convert_str_to_list(fragment_labels_df, "smiles")

        peaks_smiles = fragment_labels_df["smiles"].values

        curr_smiles_set = set()
        for peak_smiles in peaks_smiles:
            is_list = isinstance(peak_smiles, list)
            if peak_smiles == "NAN" or (not is_list and pd.isna(peak_smiles)):
                continue
            else:
                peak_smiles_set = set(peak_smiles)
                curr_smiles_set = curr_smiles_set.union(peak_smiles_set)
        return curr_smiles_set

    logging.info("Reading in all magma ouptputs")
    all_smiles_sets = chunked_parallel(
        list(spec_name_to_magma_file.items()),
        extract_smiles_set,
        chunks=500,
        max_cpu=30,
    )
    logging.info("Unioning smiles sets")
    smiles_set = set()
    smiles_set.update(*all_smiles_sets)

    smiles_set = list(smiles_set)
    logging.info("Generate sparse representation morgan fingerprint for unique smiles")

    fps = chunked_parallel(
        smiles_set, magma_utils.get_magma_fingerprint, chunks=1000, max_cpu=30
    )
    fps = np.vstack(fps)
    return smiles_set, fps


def get_and_dump_fps(summary_df, output_dir):
    """get_and_dump_fps.

    Args:
        summary_df:
        output_dir:
    """
    smiles_set, smiles_fps = get_fingerprint_set(summary_df)
    keys_to_index = dict(zip(smiles_set, np.arange(len(smiles_set))))
    fp_index_file = Path(output_dir) / "magma_smiles_fp_index.p"
    fp_file = Path(output_dir) / "magma_smiles_fp.hdf5"

    with open(fp_index_file, "wb") as fp:
        pickle.dump(keys_to_index, fp)

    logging.info("Dumping to h5py")
    h = h5py.File(fp_file, "w")
    h.create_dataset("features", data=smiles_fps, dtype=np.uint8)


def run_magma_augmentation_sirius(
    spectra_dir,
    output_dir,
    lowest_penalty_filter,
    spec_labels,
    skip_magma=False,
    workers=20,
    constrain_formula=False,
):
    """run_magma_augmentation_sirius.

    Run magma augmentation but take as input a directory of sirius files. If
    constrain formula, assert that each assigned peak _must_ have a fragment
    that matches the formula.

    Args:
        spectra_dir:
        output_dir:
        lowest_penalty_filter:
        spec_labels:
        skip_magma:
        workers:
        constrain_formula:
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    logging.info("Create magma spectra files")

    # Get all spectra files
    spectra_dir = Path(spectra_dir)
    peak_summary_file = spectra_dir / "summary_statistics/summary_df.tsv"
    summary_df = pd.read_csv(peak_summary_file, sep="\t", index_col=0)
    spec_name_to_spec = dict(summary_df[["spec_name", "spec_file"]].values)

    # Read in spec to smiles
    df = pd.read_csv(spec_labels, sep="\t")
    spec_to_smiles = dict(df[["spec", "smiles"]].values)
    spec_names = list(spec_to_smiles.keys())

    if not skip_magma:
        packaged_args = [
            spec_name_to_spec,
            spectra_dir,
            output_dir,
            lowest_penalty_filter,
            spec_to_smiles,
            constrain_formula,
        ]
        safe_parallel = lambda x: magma_augmentation_sirius(x, *packaged_args)
        chunked_parallel(spec_names, safe_parallel, chunks=1000, max_cpu=workers)

    # Save a summary file
    summary_save_path = output_dir / "magma_file_mapping_df.csv"
    if not summary_save_path.exists():
        logging.info("Create magma to spectra mapping file")
        spectra_names = []
        magma_file_paths = []
        for spectra_name in spec_names:
            magma_file_path = (Path(output_dir) / spectra_name).with_suffix(".magma")

            # Skip if neither file is real
            if not magma_file_path.exists():
                continue

            spectra_names.append(spectra_name)
            magma_file_paths.append(magma_file_path)

        summary_data = {
            "spec_name": spectra_names,
            "magma_file": magma_file_paths,
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_save_path)

    else:
        summary_df = pd.read_csv(summary_save_path, index_col=0)

    get_and_dump_fps(summary_df, output_dir)


if __name__ == "__main__":
    import time

    start_time = time.time()
    args = get_args()
    kwargs = args.__dict__
    run_magma_augmentation_sirius(**kwargs)
    end_time = time.time()
    print(f"Program finished in: {end_time - start_time} seconds")
