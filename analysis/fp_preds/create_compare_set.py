"""create_compare_set.py

Simplescript to take the outputs from one of our model predictions and replace
a desired set of bits with the predictions from CSI:FingerID in preparation for
retrieval comparisons

Example call:

python3 process_results/create_compare_set.py --in-file results/2022_05_13_growing_ms_worst_high_lambda/2022_05_13-071210_783e0ba6950ce9146c97770c1c1195fc/spectra_encoding_csi2022.p --csi-input data/paired_spectra/csi2022/prev_results/spectra_encoding_csi2022.p --out-dir 2022_05_15_order_study --debug --replace-strategies random likelihood_diff_desc likelihood_diff_asc likelihood_desc likelihood_asc var_desc var_asc norm_ll_asc norm_ll_desc 

"""

import numpy as np
import pickle
from pathlib import Path
import argparse
import copy


def clamped_log(x, clamp_val=-100):
    """Compute log and clamp output value"""
    res = np.log(x)
    res[res <= clamp_val] = clamp_val
    return res


def clamp(x, min=1e-7, max=1 - 1e-7, in_place=False):
    """Clamp values"""
    if not in_place:
        x = x.copy()
    x[x <= min] = min
    x[x >= max] = max
    return x


def clamped_likelihood(targs, preds):
    """Clamped likelihood"""
    return targs * clamped_log(preds) + (1 - targs) * clamped_log(1 - preds)


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-file",
        help="Name of initial predictions",
        default="results/2022_05_08_growing_ms/2022_05_08-015724_826c48eee06bf3487d79ba7e22909cd5/spectra_encoding_csi2022.p",
    )
    parser.add_argument(
        "--csi-input",
        help="Name of csi inputs to replace with",
        default="data/paired_spectra/csi2022/prev_results/spectra_encoding_csi2022.p",
    )
    parser.add_argument(
        "--out-dir",
        help="Name of csi inputs to replace with",
        default="2022_05_08_replace_preds",
    )
    parser.add_argument(
        "--replace-strategies",
        help="Name of strategy for replacing bits",
        action="store",
        nargs="+",
        default=["random"],
        choices=[
            "random",
            "likelihood_diff_desc",
            "likelihood_diff_asc",
            "likelihood_desc",
            "likelihood_asc",
            "var_desc",
            "var_asc",
            "norm_ll_asc",
            "norm_ll_desc",
        ],
    )
    parser.add_argument(
        "--subs",
        help="List of amount of substitutions to make",
        action="store",
        nargs="+",
        default=[0, 10, 25, 50, 100, 500, 1000, 1500, 3000, 5496],
        type=int,
    )
    parser.add_argument(
        "--debug",
        help="If true, debug",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--clip-outs",
        help="If true, clip outputs to not be 0 or 1",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def random_order(my_targs, my_preds, csi_targs, csi_preds, descending=False):
    """Return a random ordering over the bits"""
    bit_order = np.arange(my_preds.shape[1])
    np.random.shuffle(bit_order)
    return bit_order


def likelihood_diff_order(my_targs, my_preds, csi_targs, csi_preds, descending=False):
    """Order based upon likelihood difference in their preds and ours"""
    # Compute all the bit-wise likelihoods
    my_likelihoods = clamped_likelihood(my_targs, my_preds).mean(0)
    csi_likelihoods = clamped_likelihood(csi_targs, csi_preds).mean(0)

    # Compute largest magnitude differences
    likelihood_diffs = my_likelihoods - csi_likelihoods
    sorted_bits = np.argsort(likelihood_diffs)
    if descending:
        sorted_bits = sorted_bits[::-1]

    return sorted_bits


def norm_likelihood(my_targs, my_preds, csi_targs, csi_preds, descending=False):
    """Order based upon likelihood normalized by the background likelihood due to class bias"""

    # Compute log likelihoods by bit
    my_likelihoods = clamped_likelihood(my_targs, my_preds).mean(0)
    frac_on = my_targs.mean(0)

    # Compute expected log likelihood
    e_ll = frac_on * clamped_log(frac_on) + (1 - frac_on) * clamped_log(1 - frac_on)

    # Compute log odds ratio
    log_odds_ratio = my_likelihoods - e_ll
    sorted_bits = np.argsort(log_odds_ratio)
    if descending:
        sorted_bits = sorted_bits[::-1]
    return sorted_bits


def likelihood_order(my_targs, my_preds, csi_targs, csi_preds, descending=False):
    """Order based upon our likelihoods per bit"""
    # Compute all the bit-wise likelihoods
    my_likelihoods = clamped_likelihood(my_targs, my_preds).mean(0)
    sorted_bits = np.argsort(my_likelihoods)
    if descending:
        sorted_bits = sorted_bits[::-1]
    return sorted_bits


def var_order(my_targs, my_preds, csi_targs, csi_preds, descending=False):
    """Order based upon variance in target values"""
    # Compute all the bit-wise likelihoods
    targ_var = np.var(my_targs, 0)
    sorted_bits = np.argsort(targ_var)
    if descending:
        sorted_bits = sorted_bits[::-1]
    return sorted_bits


if __name__ == "__main__":
    args = parse_args()

    in_file = args.in_file
    csi_input = args.csi_input
    debug = args.debug
    clip = args.clip_outs
    replace_strategies = args.replace_strategies

    out_dir = Path(f"results/{args.out_dir}")
    out_dir.mkdir(exist_ok=True)

    # Load both our outputs and CSI outputs
    pickled_obj = pickle.load(open(in_file, "rb"))
    csi_inputs = pickle.load(open(csi_input, "rb"))

    # Filter both down to only 0 element
    csi_inputs = [i for i in csi_inputs if i["split_name"] == "Fold_0"][0]
    my_inputs = pickled_obj[0]

    # Extract csi inputs & argsort
    csi_targs = np.vstack(csi_inputs["targs"])
    csi_preds = csi_inputs["preds"]
    csi_names = np.array(csi_inputs["names"])
    csi_new_inds = np.argsort(csi_names)
    csi_targs = csi_targs[csi_new_inds]
    csi_preds = csi_preds[csi_new_inds]
    csi_names = csi_names[csi_new_inds]

    # Extract my inputs & argsort
    my_targs = np.vstack(my_inputs["targs"])
    my_preds = my_inputs["preds"]
    my_names = np.array(my_inputs["names"])
    my_new_inds = np.argsort(my_names)
    my_targs = my_targs[my_new_inds]
    my_preds = my_preds[my_new_inds]
    my_names = my_names[my_new_inds]

    assert np.all(my_names == csi_names)

    # Define the intervals to loop over
    replace_intervals = args.subs

    if debug:
        replace_intervals = replace_intervals[:2]

    # For each of these, replace the fingerprints and make a new one
    output_entries = []
    for replace_strategy in replace_strategies:

        # Compute different strategies
        if replace_strategy == "random":
            sorted_bits = random_order(my_targs, my_preds, csi_targs, csi_preds)
        elif replace_strategy == "likelihood_diff_desc":
            sorted_bits = likelihood_diff_order(
                my_targs, my_preds, csi_targs, csi_preds, descending=True
            )
        elif replace_strategy == "likelihood_diff_asc":
            sorted_bits = likelihood_diff_order(
                my_targs, my_preds, csi_targs, csi_preds, descending=False
            )
        elif replace_strategy == "likelihood_desc":
            sorted_bits = likelihood_order(
                my_targs, my_preds, csi_targs, csi_preds, descending=True
            )
        elif replace_strategy == "likelihood_asc":
            sorted_bits = likelihood_order(
                my_targs, my_preds, csi_targs, csi_preds, descending=False
            )
        elif replace_strategy == "var_desc":
            sorted_bits = var_order(
                my_targs, my_preds, csi_targs, csi_preds, descending=True
            )
        elif replace_strategy == "var_asc":
            sorted_bits = var_order(
                my_targs, my_preds, csi_targs, csi_preds, descending=False
            )
        elif replace_strategy == "norm_ll_desc":
            sorted_bits = norm_likelihood(
                my_targs, my_preds, csi_targs, csi_preds, descending=True
            )
        elif replace_strategy == "norm_ll_asc":
            sorted_bits = norm_likelihood(
                my_targs, my_preds, csi_targs, csi_preds, descending=False
            )
        else:
            raise NotImplementedError()

        for replace_amt in replace_intervals:

            # Copy and replace with csi bit preds
            pred_copy = my_preds.copy()
            if clip:
                pred_copy = clamp(pred_copy, min=1e-7, max=1 - 1e-7)

            replace_inds = sorted_bits[:replace_amt]
            pred_copy[:, replace_inds] = csi_preds[:, replace_inds]

            # Create output entry dict
            new_output = copy.deepcopy(my_inputs)
            if debug:
                new_output["targs"] = my_targs[:10]
                new_output["preds"] = pred_copy[:10]
                new_output["names"] = my_names[:10]
            else:
                new_output["targs"] = my_targs
                new_output["preds"] = pred_copy
                new_output["names"] = my_names

            # Change model
            new_output["args"]["model"] = f"{replace_strategy}_{replace_amt}"
            output_entries.append(new_output)

    out_name = out_dir / "spectra_encoding_replacements.p"
    pickle.dump(output_entries, open(out_name, "wb"))
