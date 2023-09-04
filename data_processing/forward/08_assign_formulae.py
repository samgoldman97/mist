""" 06_assign_formulae.py

Given output of a model of predictions, try to assign formula substructures to
the forward prediction results


"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from functools import partial

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from mist import utils

# Use to create vectors of chem formulae


def bin_frags(frags, upper_limit=1000, num_bins=1000):
    bins = np.linspace(0, upper_limit, num=num_bins)
    # Convert to digitized spectra
    digitized = np.digitize(frags, bins=bins)
    return digitized


def h_ct(form_vec):
    # If there's carbon
    return (utils.element_to_position["H"] * form_vec).sum(-1)


def c_ct(form_vec):
    # If there's carbon
    return (utils.element_to_position["C"] * form_vec).sum(-1)


def extract_table(
    smi_pred_tuple,
    is_sparse=True,
    max_keep=50,
    num_bins=1000,
    upper_limit=1000,
    no_filter=False,
):
    """Make table assignments"""

    smi, pred = smi_pred_tuple
    mol = Chem.MolFromSmiles(smi)
    full_form = CalcMolFormula(mol)
    inchikey = Chem.MolToInchiKey(mol)
    cross_prod, masses = utils.get_all_subsets(full_form)

    # Filter by H and C cts
    if no_filter:
        pass
    else:
        num_hs = h_ct(cross_prod)
        num_cs = c_ct(cross_prod)
        h_over_c = num_hs / (num_cs + 1e-22)
        passes_ch_test = np.logical_and(h_over_c < 6.0, h_over_c > 0.1)
        no_carbon = num_hs > 0
        h_c_filter = np.logical_or(no_carbon, passes_ch_test)
        cross_prod = cross_prod[h_c_filter]
        masses = masses[h_c_filter]

    # Convert masses into bins
    # Look for overlap
    if not is_sparse:
        pred_inds = np.argwhere(pred).flatten()
        pred_intens = pred[pred_inds]
        pred_ind_to_inten = dict(zip(pred_inds, pred_intens))
    else:
        pred_inds, pred_intens = pred[0], pred[1]
        pred_ind_to_inten = dict(zip(pred_inds, pred_intens))

    # 1.0 Convert this into bin indices
    mass_inds = bin_frags(masses, num_bins=num_bins, upper_limit=upper_limit)

    # 2.0 Look for overlap in bin indices
    overlap_inds = set(mass_inds).intersection(pred_inds)

    # 3.0 Find all elements that correspond to the bin
    formulas, mzs, intensities = [], [], []
    for j in sorted(overlap_inds):
        inten = pred_ind_to_inten[j]

        arg_options = np.argwhere(mass_inds == j).flatten()
        cross_prod_ind = np.random.choice(arg_options)
        formula = utils.vec_to_formula(cross_prod[cross_prod_ind])
        mz = masses[cross_prod_ind]
        formulas.append(formula)
        mzs.append(mz)
        intensities.append(inten)

    formulas, mzs, intensities = (
        np.array(formulas),
        np.array(mzs),
        np.array(intensities),
    )

    # If > 50, argsort by intensities
    top_intens = np.argsort(intensities)[::-1][:max_keep]
    formulas, mzs, intensities = (
        formulas[top_intens],
        mzs[top_intens],
        intensities[top_intens],
    )

    df = pd.DataFrame(
        {"mz": mzs, "intensity": intensities, "chemicalFormula": formulas}
    )
    if len(df) == 0:
        return None

    df = df.sort_values(axis=0, by="mz").reset_index(drop=True)

    # Add scaling factor to get logarithmic peak affect
    df["intensity"] = np.exp(df["intensity"].values * 4)
    max_inten = (df["intensity"]).max()
    df["intensity"] = df["intensity"] / max_inten
    out = {"tbl": df, "full_form": full_form, "inchikey": inchikey}
    return out


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--preds",
        help="pickle preds (assume sparse format)",
        default="results/2022_08_01_forward_gnn_best/subsample_preds.p",
    )
    parser.add_argument("--output-dir", help="Name of output dir")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--no-filter", action="store_true", default=False)
    parser.add_argument(
        "--num-bins",
        type=int,
        default=10000,
        help="Num bins used in prediction forward module",
    )
    parser.add_argument(
        "--upper-limit",
        type=int,
        default=1000,
        help="Num bins used in prediction forward module",
    )
    parser.add_argument("--orig-labels", action="store", default=None)
    parser.add_argument("--split-file", action="store", default=None)
    return parser.parse_args()


def safe_extract_table(
    smi_pred_tuple,
    is_sparse=True,
    max_keep=50,
    num_bins=1000,
    upper_limit=1000,
    no_filter=False,
):
    try:
        output = extract_table(
            smi_pred_tuple,
            is_sparse=is_sparse,
            max_keep=max_keep,
            num_bins=num_bins,
            upper_limit=upper_limit,
            no_filter=no_filter,
        )
        return output
    except:
        # print(f"Failed on smi: {smi_pred_tuple[0]}")
        return None


def main():
    args = get_args()
    no_filter = args.no_filter
    pred_file = Path(args.preds)
    outdir = Path(args.output_dir)
    outdir.mkdir(exist_ok=True)
    preds = pickle.load(open(pred_file, "rb"))

    orig_labels = args.orig_labels
    split_file = args.split_file
    exclude_ikeys = []
    if orig_labels is not None and split_file is not None:
        # Create a list of inchikeys to exclude by val/test
        spec_to_ikey = dict(
            pd.read_csv(orig_labels, sep="\t")[["spec", "inchikey"]].values
        )
        split_df = pd.read_csv(split_file, sep=",")
        name_cols = set(split_df.keys())
        name_cols = name_cols.difference(["name"])
        assert len(name_cols) == 1
        fold_name = list(name_cols)[0]

        fold_name = list(name_cols)[0]
        cur_labels = split_df[fold_name].values
        test_inds = cur_labels == "test"
        val_inds = cur_labels == "val"
        exclude_inds = np.logical_or(val_inds, test_inds)
        exclude_names = split_df["name"].values[exclude_inds]
        exclude_ikeys = [spec_to_ikey[i] for i in exclude_names]
        # exclude_ikeys = []

    # smi, pred = preds['names'][0], preds['preds'][0]
    # df = extract_table(smi, pred, is_sparse=True, max_keep=1000,
    #                   num_bins=args.num_bins,
    #                   upper_limit=1000)

    # Extract tables from forward predictions
    spec_tuples = list(zip(preds["names"], preds["preds"]))
    safe_extract = partial(
        safe_extract_table,
        is_sparse=True,
        max_keep=50,
        num_bins=args.num_bins,
        upper_limit=args.upper_limit,
        no_filter=no_filter,
    )

    if args.debug:
        spec_tuples = spec_tuples[:20]
    all_tables = utils.chunked_parallel(spec_tuples, safe_extract, max_cpu=64)
    output_labels = []
    spec_prefix = "forward_pseudo"
    spec_names = [f"{spec_prefix}_{i}" for i in np.arange(1, len(preds["names"]) + 1)]
    tsv_dir = outdir / "spectra"
    tsv_dir.mkdir(exist_ok=True)
    skip_ctr = 0
    for spec_name, tbl, smi in zip(spec_names, all_tables, preds["names"]):

        if tbl is None or len(tbl["tbl"]) < 5 or tbl is None:
            continue

        out_loc = tsv_dir / f"{spec_name}.tsv"
        tbl["tbl"].to_csv(out_loc, sep="\t")
        out_entry = {
            "dataset": "unpaired",
            "spec": spec_name,
            "name": spec_name,
            "ionization": "[M+H]+",  # Use M+H by default
            "smiles": smi,
            "formula": tbl["full_form"],
            "inchikey": tbl["inchikey"],
        }
        if tbl["inchikey"] in exclude_ikeys:
            skip_ctr += 1
            continue

        output_labels.append(out_entry)
        # I need everything that's in the labels file

    print(f"Skipped {skip_ctr} due to ikey being found in val/test")
    out_labels_df = pd.DataFrame(output_labels)
    out_labels_df.to_csv(tsv_dir.parent / "labels.tsv", sep="\t")


if __name__ == "__main__":
    main()
