""" summarize_model_ablation.py


Create a summary of the data file for different ablation experiments.


"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

from functools import partial
from mist import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir", default="results/2022_09_08_canopus_ablations_lr_fixed"
    )
    return parser.parse_args()


def cos_sim(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    pred

    sim = cosine_similarity(pred, targ)
    sim = np.diag(sim)
    return sim[:, None]


def ll(pred, targ):
    """nll.

    Args:
        pred:
        targ:
    """
    log = partial(utils.clamped_log_np, _min=-5)
    ll = targ * log(pred) + (1 - targ) * log(1 - pred)
    return ll


def main():
    """main."""
    args = get_args()
    res_dir = Path(args.dir)
    full_res = []
    for pred_file in res_dir.rglob("preds/*.p"):
        pred_name = pred_file.parent.parent.name
        with open(pred_file, "rb") as fp:
            res = pickle.load(fp)
            names, preds, targs = res["names"], res["preds"], res["targs"]
            preds = np.vstack(preds)

            # Output is going to be
            cos_sims = cos_sim(preds, targs).mean(-1)
            ll_ = ll(preds, targs).mean(-1)

            outputs = [
                {"cos_sim": j, "ll": k, "name": i, "method": pred_name}
                for i, j, k in zip(names, cos_sims, ll_)
            ]
            full_res.extend(outputs)

    df = pd.DataFrame(full_res)
    method, split = list(zip(*[i.rsplit("_", 1) for i in df["method"].values]))
    df["method"] = method
    df["split"] = split

    df.to_csv(res_dir / "fp_pred_summary.tsv", sep="\t", index=None)

    mean_vals = df.groupby(["method"]).mean()  # , "split"]).mean()
    # mean_ct = mean_vals.groupby("method").count()
    mean_ct = df.groupby("method").count()[["cos_sim", "ll"]]

    # mean_std = mean_vals.groupby("method").std() mean_ct
    mean_std = df.groupby("method").std()
    mean_se = mean_std / np.sqrt(mean_ct)
    mean_mean = mean_vals.groupby("method").mean()

    print("Mean of each run:")
    print(mean_mean)

    print("SE of each run:")
    print(mean_se)


if __name__ == "__main__":
    main()
