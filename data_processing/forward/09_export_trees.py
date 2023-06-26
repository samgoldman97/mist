""" 07_export_trees.py
convert the new spec files into trees 

Note: this is copied from the magma file verbatim

"""
import argparse
from typing import List
import numpy as np
from pathlib import Path
from functools import partial
import pandas as pd
import json

from mist import utils


def get_args():
    """get args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-spec-dir")
    return parser.parse_args()


def process_tree(input_tuple, outdir) -> List[dict]:
    spec_name, magma_file, full_formula = input_tuple
    df = pd.read_csv(magma_file, sep="\t", index_col=0)
    max_inten = np.max(df["intensity"].values)
    if max_inten == 0:
        return None

    df = df.sort_values(by="mz", ascending=False).reset_index(drop=True)

    output_tree = {}
    output_tree["molecularFormula"] = full_formula
    output_tree["root"] = full_formula
    output_tree["fragments"] = []
    output_tree["losses"] = []

    cur_ind = 0
    for _, entry in df.iterrows():
        new_dict = {
            "id": cur_ind,
            "molecularFormula": entry["chemicalFormula"],
            "relativeIntensity": entry["intensity"] / max_inten,
            "mz": entry["mz"],
        }
        output_tree["fragments"].append(new_dict)
        cur_ind += 1
    output_json = json.dumps(output_tree)  # indent=2)
    output_name = outdir / f"{spec_name}.json"

    with open(output_name, "w") as fp:
        fp.write(output_json)
    return {"name": spec_name, "file_loc": output_name.name}


def main(args):

    # Pipeline is to get info from the tsv file
    # Define a new labels file and convert everything into trees
    tsv_dir = Path(args.new_spec_dir)
    new_outdir = tsv_dir.parent / "forward_trees"
    new_outdir.mkdir(exist_ok=True)
    tsv_files = list(tsv_dir.glob("*.tsv"))
    spec_names = [i.stem for i in tsv_files]

    labels_file = tsv_dir.parent / "labels.tsv"
    labels_df = pd.read_csv(labels_file, sep="\t")
    new_entry_to_full_form = dict(labels_df[["spec", "formula"]].values)

    # Second create tree exports for everything
    spec_name_file_tuples = [
        (k, v, new_entry_to_full_form[k])
        for k, v in zip(spec_names, tsv_files)
        if k in new_entry_to_full_form
    ]

    # Process these
    process_spec_full = partial(process_tree, outdir=new_outdir)

    # full_output = [process_spec_full(i) for i in spec_name_file_tuples]
    full_output = utils.chunked_parallel(
        spec_name_file_tuples, process_spec_full, max_cpu=30
    )
    # Create forward tree summary
    labels_out = pd.DataFrame([i for i in full_output if i is not None])
    out_label_file = new_outdir / "forward_tree_summary.tsv"
    labels_out["spec_name"] = labels_out["name"]
    labels_out["tree_file"] = labels_out["file_loc"]
    labels_out.to_csv(out_label_file, sep="\t")


if __name__ == "__main__":
    args = get_args()
    main(args)
