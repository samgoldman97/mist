""" 02_reformat_csi2022.py """
from pathlib import Path
import h5py
from tqdm import tqdm
import pickle
import numpy as np
import argparse
from mist import utils
from functools import partial
import pandas as pd

from rdkit import Chem


OUT_DIR = Path("data/paired_spectra/")
FP_LOC = Path("fingerprints/precomputed_fp/")

START_DIR = Path("data/raw/csievaldata/")
INDEP_DIR = START_DIR / Path("independent")
DATA_DIR = START_DIR / Path("crossval")
SPLITS_LOC = START_DIR / Path("crossvalidation_folds.txt")

INDEP_FPS = INDEP_DIR.joinpath("fingerprints.hdf5")
INDEP_FPS_PRED = INDEP_DIR.joinpath("predictions.hdf5")
INDEP_RANKINGS = INDEP_DIR.joinpath("ranking.hdf5")
INDEP_SPECTRA = INDEP_DIR.joinpath("spectra")

DATA_FPS = DATA_DIR.joinpath("fingerprints.hdf5")
DATA_FPS_PRED = DATA_DIR.joinpath("prediction.hdf5")
DATA_RANKINGS = DATA_DIR.joinpath("ranking.hdf5")
DATA_SPECTRA = DATA_DIR.joinpath("spectra")


DATASET_NAME = "csi2022"
EVAL_NAME = "casmi"


def debug_mode():
    """debug_mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()
    return args.debug


# Do bulk copy
def read_write_spec(x, targ_dir):
    """read_write_spec.

    Args:
        x:
        targ_dir:
    """
    meta, spec = utils.parse_spectra(x)
    meta_keys = list(meta.keys())
    meta_keep = ["compound", "formula", "parentmass", "ionization", "InChI", "InChIKey"]
    meta_comment = set(meta_keys).difference(set(meta_keep))

    out_meta = "\n".join([f">{i} {meta.get(i, None)}" for i in meta_keep])
    out_comment = "\n".join([f"#{i} {meta.get(i, None)}" for i in meta_comment])
    peak_list = []
    for k, v in spec:
        peak_entry = []
        peak_entry.append(f">{k}")
        peak_entry.extend([f"{row[0]} {row[1]}" for row in v])
        peak_list.append("\n".join(peak_entry))

    out_peaks = "\n\n".join(peak_list)

    total_str = f"{out_meta}\n{out_comment}\n\n{out_peaks}"
    outpath = Path(targ_dir) / Path(Path(x).name)

    with open(outpath, "w") as fp:
        fp.write(total_str)

    inchikey = meta["InChIKey"]
    inchi = meta["InChI"]
    smiles = meta["smiles"]

    # smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
    inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))
    out_dict = {
        "smiles": smiles,
        "formula": meta["formula"],
        "name": meta["compound"],
        "spec": Path(x).stem,
        "ionization": meta["ionization"].replace(" ", ""),
        "inchikey": inchikey,
        "dataset": None,
    }
    return out_dict


def process_spectra(
    dataset_name: str, out_dir: Path, spectra_input_dir: Path, debug: bool = False
):
    """process_spectra.

    Args:
        dataset_name (str): dataset_name
        out_dir (Path): out_dir
        spectra_input_dir (Path): spectra_input_dir
        debug (bool): debug
    """
    # Create new directories
    out_dataset = out_dir / Path(dataset_name)
    spectra_out_dir = out_dataset.joinpath("spec_files")
    out_dataset.mkdir(exist_ok=True)
    spectra_out_dir.mkdir(exist_ok=True)

    # Parse and loop over all spectra
    input_spectra = [str(i) for i in spectra_input_dir.glob("*.ms")]

    if debug:
        input_spectra = input_spectra[:100]

    # Run single function
    single_func = partial(read_write_spec, targ_dir=spectra_out_dir)
    # [single_func(i) for i in input_spectra]

    # Debug
    dataset_entries = utils.chunked_parallel(
        input_spectra, single_func, chunks=100, max_cpu=32
    )
    # Overwrite dataset name
    df_out = pd.DataFrame(dataset_entries)
    df_out["dataset"] = dataset_name

    # Export labels
    label_trg = out_dataset.joinpath("labels.tsv")
    df_out = df_out[
        ["dataset", "spec", "name", "formula", "ionization", "smiles", "inchikey"]
    ]
    df_out.to_csv(label_trg, sep="\t", index=False)


def reformat_fingerprints(
    dataset_name: str, out_dir: Path, hdf_input: Path, debug: bool = False
):
    """reformat_fingerprints.

    Args:
        dataset_name (str): dataset_name
        out_dir (Path): out_dir
        hdf_input (Path): hdf_input
        debug (bool): debug
    """
    output_name = str(FP_LOC / Path(f"cache_csi_{dataset_name}"))

    labels = out_dir / Path(dataset_name) / "labels.tsv"
    df = pd.read_csv(labels, sep="\t")
    spec_to_key = dict(df[["spec", "inchikey"]].values)

    hdf_obj = h5py.File(hdf_input, "r")
    all_fps = np.vstack([i for i in hdf_obj["fingerprints_masked"]])
    all_specs = np.vstack([i for i in hdf_obj["compound_names"]])

    _temp_index = [j.item() for j in all_specs]
    _temp_index = [j if isinstance(j, str) else j.decode() for j in _temp_index]
    inchikeys = [spec_to_key.get(j) for j in _temp_index]

    print("Dumping outputs")
    # ind = np.where(all_specs.squeeze() == b"CCMSLIB00000579531")[0][0]
    # pickle.dump(all_fps[ind], open("CCMSLIB00000579531_fp.p", "wb"))

    # New output with index and hdf5
    keys_to_index = dict(zip(inchikeys, np.arange(len(all_fps))))
    with open(output_name + "_index.p", "wb") as fp:
        pickle.dump(keys_to_index, fp)

    h = h5py.File(output_name + ".hdf5", "w")
    h.create_dataset("features", data=all_fps, dtype=np.uint8)

    with open(output_name + "_fp_inds.p", "wb") as fp:
        ind_obj = hdf_obj["fingerprint_indizes"][:]
        ind_obj = dict(zip(np.arange(len(ind_obj)).tolist(), ind_obj))
        pickle.dump(ind_obj, fp)


def _get_name_to_fp(dataset_name: str, hdf_obj, debug: bool = False) -> dict:
    """_get_name_to_fp.

    Args:
        dataset_name (str): dataset_name
        hdf_obj:
        debug (bool): debug

    Returns:
        dict:
    """
    hdf_obj = h5py.File(hdf_obj, "r")
    all_fps = np.vstack([i for i in hdf_obj["fingerprints_masked"]])
    all_specs = np.vstack([i for i in hdf_obj["compound_names"]])

    _temp_index = [j.item() for j in all_specs]
    _temp_index = [j if isinstance(j, str) else j.decode() for j in _temp_index]

    return dict(zip(_temp_index, all_fps))


def reformat_preds(
    dataset_name: str,
    out_dir: Path,
    hdf_input: Path,
    debug: bool = False,
    true_fp_hdf=None,
):
    """reformat_preds.

    Args:
        dataset_name (str): dataset_name
        out_dir (Path): out_dir
        hdf_input (Path): hdf_input
        debug (bool): debug
        true_fp_hdf:
    """

    labels = out_dir / Path(dataset_name) / "labels.tsv"
    df = pd.read_csv(labels, sep="\t")

    # If we know how splits were conducted, add this knowledge in
    splits = [0, 1, 2, 3, 4]
    split_files = [
        out_dir / Path(dataset_name) / Path(f"splits/csi_split_{i}.txt") for i in splits
    ]
    folds = {}
    for split_file in split_files:
        if split_file.exists():
            df_split = pd.read_csv(split_file, index_col=False)
            fold_names = set(df_split.keys())
            fold_names.remove("name")
            names = df_split["name"].values
            for i in fold_names:
                fold_names = names[df_split[i] == "test"]
                folds[i] = fold_names

    if len(folds) is None:
        folds = {"CSI_Fold": df["spec"].values}

    hdf_obj = h5py.File(hdf_input, "r")
    preds = np.vstack(list(hdf_obj["predictions"]))
    names = np.array(
        [i if isinstance(i, str) else i.decode() for i in hdf_obj["names"]]
    )

    name_to_fp = {}
    if true_fp_hdf is not None:
        name_to_fp = _get_name_to_fp(dataset_name, true_fp_hdf, debug)

    results_name_base = out_dir / Path(dataset_name) / Path("prev_results")
    results_name_base.mkdir(exist_ok=True)
    for fold, fold_names in folds.items():
        to_include = np.array([j in fold_names for j in names])
        preds_temp = preds[to_include]
        names_temp = names[to_include]

        targs_temp = [name_to_fp.get(i, None) for i in names_temp]
        result = {
            "dataset_name": dataset_name,
            "names": names_temp,
            "preds": preds_temp,
            "targs": targs_temp,
            "args": {"model": "CSI:FingerID"},
            "split_name": fold,
        }

        results_name = results_name_base / Path(
            f"spectra_encoding_{dataset_name}_{fold}.p"
        )
        with open(results_name, "wb") as fp:
            pickle.dump(result, fp)


def reformat_rankings(
    dataset_name: str, out_dir: Path, hdf_input: Path, debug: bool = False
):
    """reformat_rankings.

    Args:
        dataset_name (str): dataset_name
        out_dir (Path): out_dir
        hdf_input (Path): hdf_input
        debug (bool): debug
    """

    labels = out_dir / Path(dataset_name) / "labels.tsv"
    retrieval_hdf = out_dir / Path(dataset_name) / "retrieval_hdf"
    retrieval_hdf.mkdir(exist_ok=True)

    df = pd.read_csv(labels, sep="\t")
    hdf_obj = h5py.File(hdf_input, "r")

    # Export lookup libraries into new HDF5 format
    all_formulas = [
        i if isinstance(i, str) else i.decode()
        for i in list(hdf_obj["candidateListFormulas"])
    ]
    all_offsets = [int(i) for i in list(hdf_obj["candidateListOffsets"])]
    all_lengths = [int(i) for i in list(hdf_obj["candidateListLengths"])]

    output_indices = {
        i: {"offset": j, "length": k}
        for i, j, k in zip(all_formulas, all_offsets, all_lengths)
    }

    keys_file = retrieval_hdf / Path("pubchem_with_csi_retrieval_db_index.p")
    fp_file = retrieval_hdf / Path("pubchem_with_csi_retrieval_db.hdf5")

    # Establish HDF file
    h = h5py.File(fp_file, "w")
    if not debug:
        # Copy old dataset in pieces
        dataset_shape = hdf_obj["candidateListFingerprints"].shape
        dataset_len = dataset_shape[0]
        new_dataset = h.create_dataset(
            "fingerprints",
            dataset_shape,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=5,
        )
        chunk_size = 300000
        num_chunks = (dataset_len // chunk_size) + 1
        print("Starting to iterate over old dataset and copy in chunks")
        for chunk in tqdm(range(num_chunks)):
            start = chunk * chunk_size
            end = start + chunk_size
            if end > dataset_len:
                end = dataset_len
            if start > dataset_len:
                raise ValueError()
            data_src = hdf_obj["candidateListFingerprints"][start:end]
            new_dataset[start:end] = data_src
        h.close()
    else:
        # If debug
        # 1. Get all formulas we actually want to retrieve
        formulas = pd.unique(df["formula"]).tolist()

        # 2. create a list of all these entries from the HDF dataset
        full_fps = []
        new_offset_dict = {}
        cur_offset = 0
        for formula in formulas:
            _entry = output_indices[formula]
            offset = _entry["offset"]
            length = _entry["length"]
            fps = hdf_obj["candidateListFingerprints"][offset : offset + length]
            full_fps.append(fps)

            new_offset_dict[formula] = {"offset": cur_offset, "length": length}
            cur_offset += length

        full_fps = np.vstack(full_fps)

        # 3. Redo the numberings
        output_indices = new_offset_dict

        # 4. Output to file
        h.create_dataset(
            "fingerprints",
            data=full_fps,
            dtype=np.uint8,
            compression="gzip",
            compression_opts=5,
        )

    # Dump to output
    print("Dumping output ranking indices")
    with open(keys_file, "wb") as fp:
        pickle.dump(output_indices, fp)

    # Export results for retrieval
    methods = {"Bayes", "ModifiedPlatt"}
    entries = []
    spec_names = np.array(
        [i if isinstance(i, str) else i.decode() for i in list(hdf_obj["names"])]
    )
    in_pubchem = np.array(list(hdf_obj["Bayes_found"])).astype(bool)

    print("Pulling hdf5 values")
    # Pull these all at once for speed
    method_to_vals = {
        f"{method_name}{suffix}": list(hdf_obj[f"{method_name}{suffix}"])
        for method_name in methods
        for suffix in ["_from", "_to"]
    }

    print(f"Iterating over ranking hdf5")
    for ind, name in tqdm(enumerate(spec_names)):
        entry = {}
        for method in methods:
            from_key = f"{method}_from"
            _from = int(method_to_vals[from_key][ind])
            entry[method] = f"{_from}"  # "-{_to}"
            entry["Instance"] = name
        entries.append(entry)

    result_folder = out_dir / Path(dataset_name) / Path("prev_results")
    result_folder.mkdir(exist_ok=True)

    out_df_name = result_folder / Path("csi_fingerid_retrieval.txt")
    unfound_names = result_folder / Path("unfound_specs.txt")

    entries = np.array(entries)
    entries_export = entries[in_pubchem].tolist()

    df_out = pd.DataFrame(entries_export)
    df_out.to_csv(out_df_name, sep="\t", index=False)

    # Check on the entries that weren't found
    not_found = ~in_pubchem
    spec_names_unfound = spec_names[not_found]
    open(unfound_names, "w").write("\n".join(spec_names_unfound))


def convert_retrieval_to_pickled(retrieval_file, dataset_name):
    """convert_retrieval_to_pickled.

    Args:
        retrieval_file:
    """
    parent_path = Path(retrieval_file).parent
    df = pd.read_csv(retrieval_file, sep="\t")
    names = df["Instance"]

    for k in df.columns:
        if k == "Instance":
            continue
        pred_ranking_outs = {}
        ind_found = df[k]
        pred_ranking_outs["dataset_name"] = "csi2022"
        pred_ranking_outs["retrieval_settings"] = {"dist_name": k}
        pred_ranking_outs["ind_found"] = ind_found.values
        pred_ranking_outs["names"] = np.array(names)

        file_out = parent_path / f"csi_retrieval_{k}.p"
        with open(file_out, "wb") as fp:
            pickle.dump(pred_ranking_outs, fp)


def reformat_splits(
    dataset_name: str, out_dir: Path, splits_input: Path, debug: bool = False
):
    """reformat_splits.

    Args:
        dataset_name (str): dataset_name
        out_dir (Path): out_dir
        splits_input (Path): splits_input
        debug (bool): debug
    """

    df = pd.read_csv(splits_input, sep="\t", names=["spec", "inchikey", "fold"])
    splits_base = out_dir / Path("splits")
    splits_base.mkdir(exist_ok=True)

    # Copy splits
    fold_mapping = dict(df[["spec", "fold"]].values)

    # train, test
    for i in range(5):
        entries = []
        for k, v in fold_mapping.items():
            # if k not in valid_spec:
            #    continue
            entry = {"name": k}
            entry[f"Fold_{i}"] = "train" if int(v) != i else "test"
            entries.append(entry)

        df_split = pd.DataFrame(entries)
        splits_trg = splits_base / f"csi_split_{i}.txt"
        df_split.to_csv(splits_trg, index=False)


def main():
    """main."""

    # get args
    debug = debug_mode()
    dataset_name = DATASET_NAME
    eval_name = EVAL_NAME

    if debug:
        dataset_name = f"{dataset_name}_debug"
        eval_name = f"{eval_name}_debug"

    # Gets info from the smiles
    process_spectra(dataset_name, OUT_DIR, DATA_SPECTRA, debug=debug)
    reformat_splits(dataset_name, OUT_DIR.joinpath(dataset_name), SPLITS_LOC, debug)

    reformat_fingerprints(dataset_name, OUT_DIR, DATA_FPS, debug)
    reformat_preds(
        dataset_name,
        OUT_DIR,
        DATA_FPS_PRED,
        debug,
        true_fp_hdf=DATA_FPS,
    )
    reformat_rankings(dataset_name, OUT_DIR, DATA_RANKINGS, debug)
    out_res_names = (
        OUT_DIR / DATASET_NAME / "prev_results" / "csi_fingerid_retrieval.txt"
    )
    convert_retrieval_to_pickled(out_res_names, "csi2022")

    process_spectra(eval_name, OUT_DIR, INDEP_SPECTRA, debug=debug)
    reformat_fingerprints(eval_name, OUT_DIR, INDEP_FPS, debug)
    reformat_preds(eval_name, OUT_DIR, INDEP_FPS_PRED, debug, true_fp_hdf=INDEP_FPS)
    reformat_rankings(eval_name, OUT_DIR, INDEP_RANKINGS, debug)
    out_res_names = OUT_DIR / EVAL_NAME / "prev_results" / "csi_fingerid_retrieval.txt"
    convert_retrieval_to_pickled(out_res_names, "csi2022")


if __name__ == "__main__":
    main()
