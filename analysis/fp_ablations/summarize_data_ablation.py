""" summarize_ablation_data.py


Create a summary of the data file


"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import yaml

from functools import partial
from mist import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results/2022_09_08_canopus_data_ablation")
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
        args_file = pred_file.parent.parent / "args.yaml"
        args_dict = yaml.safe_load(open(args_file, "r"))
        split_file = Path(args_dict.get("split_file"))
        split_name = split_file.name
        split_num = int(split_file.stem.split("_")[-2])
        print(split_num)

        with open(pred_file, "rb") as fp:
            res = pickle.load(fp)
            names, preds, targs = res["names"], res["preds"], res["targs"]
            preds = np.vstack(preds)

            # Output is going to be
            cos_sims = cos_sim(preds, targs).mean(-1)
            ll_ = ll(preds, targs).mean(-1)

            outputs = [
                {
                    "cos_sim": j,
                    "name": i,
                    "ll": k,
                    "split_file": split_name,
                    "split_frac": split_num,
                }
                for i, j, k in zip(names, cos_sims, ll_)
            ]

            full_res.extend(outputs)

    df = pd.DataFrame(full_res)
    df.to_csv(res_dir / "fp_pred_summary.tsv", sep="\t", index=None)
    mean = df.groupby("split_frac").mean()
    ct = df.groupby("split_frac").count()
    std = df.groupby("split_frac").std()
    print("mean", mean)
    print("se", std / np.sqrt(ct))


if __name__ == "__main__":
    main()
