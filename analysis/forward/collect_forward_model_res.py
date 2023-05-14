""" Collect results from cluster and turn into top 10 table """
from pathlib import Path
import yaml
import argparse
import json
import pandas as pd
import numpy as np

FIELDS_TO_KEEP = None


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir")
    parser.add_argument("--out-name", default=None)
    return parser.parse_args()


def collect_results(args):
    """collect_results."""
    results_dir = Path(args.results_dir)
    res_files = list(results_dir.rglob("test_results.yaml"))

    outs = []
    for res_file in res_files:
        yaml_out = yaml.safe_load(open(res_file, "r"))
        out_dict = {}
        out_dict.update(yaml_out["args"])
        out_dict.update(yaml_out["test_metrics"])
        outs.append(out_dict)

    df = pd.DataFrame(outs)
    df = df.sort_values(
        by="test_top_10_peak_overlap", axis=0, ascending=False
    ).reset_index(drop=True)
    out_name = results_dir / "collected_res.tsv"

    df.to_csv(out_name, sep="\t", index=False)


if __name__ == "__main__":
    args = get_args()
    collect_results(args)
