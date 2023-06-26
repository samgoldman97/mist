""" summarize_sirius.py

Wrangle results from sirius run output.

"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from mist.utils import chunked_parallel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-file", default="data/paired_spectra/gnps2015/labels.tsv"
    )
    parser.add_argument(
        "--sirius-folder", default="data/paired_spectra/gnps2015/sirius_outputs"
    )
    return parser.parse_args()


escape_adduct = lambda x: x.replace(" ", "")


def extract_top_specs(labels_file, sirius_folder):
    """extract_top_specs.

    Extract the top predicted formula from the set of sirius outputs

    Args:
        labels_file (str): Name of labels file outptu
        sirius_folder (str): Name of sirius folder output

    Return:
        pd.DataFrame containing outputs
    """
    ## Map ID
    df = pd.read_csv(labels_file, sep="\t") if labels_file is not None else {}
    if "formula" in df.keys():
        id_to_formula = dict(df[["spec", "formula"]].values)
    else:
        id_to_formula = {}
    def process_example(directory_full):

        # Continue if not dir
        if not directory_full.is_dir() or "_" not in directory_full.stem:
            return None

        # Get the true formula for the spectra
        num, file_id = directory_full.stem.split("_", 1)
        true_formula = id_to_formula.get(file_id, None)

        score_dir = directory_full / "scores"
        if not score_dir.is_dir():
            print(f"Cannot find: {score_dir}")
            return None

        def extract_score_from_file(score_file):
            """ calculate tree score """
            lines = [i.strip() for i in open(score_file, "r").readlines()]
            return float(lines[1].split()[-1])

        all_formulae_files = list(score_dir.glob("*.info"))
        score_tuples = [(extract_score_from_file(i), i)
                        for i in all_formulae_files]
        top_score, top_file = sorted(score_tuples, key=lambda x: x[0])[::-1][0]
        precursor_formula, adduct = top_file.stem.split("_")

        # Find all tree sand spectra
        spectra_subdir = directory_full / "spectra"
        tree_subdir = directory_full / "trees"
        info_file = directory_full / "compound.info"
        # Info file has a set of k,v pairs separated by a space. Parse these and put into a dict
        with open(info_file, "r") as fp:
            lines = fp.readlines()
            out_dict = {}
            for line in lines:
                line = line.strip()
                if "\t" in line and len(line) > 1:
                    k, v = line.split("\t", 1)
                    out_dict[k] = v
            parentmass = out_dict["ionMass"]
            parentmass = float(parentmass)



        adduct_escaped = escape_adduct(adduct)
        file_extension = f"{precursor_formula}_{adduct_escaped}"
        tree_loc = tree_subdir / f"{file_extension}.json"
        spectra_loc = spectra_subdir / f"{file_extension}.tsv"

        new_entry = {
            "spec_name": file_id,
            "spec_file": spectra_loc,
            "tree_file": tree_loc,
            "adduct": adduct, 
            "pred_formula": precursor_formula,
            "true_formula": true_formula,
            "parentmass": parentmass
        }
        return new_entry

    #[process_example(i) for i in list(Path(sirius_folder).glob("*"))]
    full_outputs = chunked_parallel(list(Path(sirius_folder).glob("*")),
                                    process_example, 100,
                                    max_cpu=30)
    results_list = [i for i in full_outputs if i is not None]
    output = pd.DataFrame(results_list)
    return output


def process_statistics(spec_df, result_dir):
    """Create summary statistics for sirius.

    Args:
        spec_df (str): Name of spectra data frame
        result_dir (str): Name of output folder

    """

    correct_formulas = []
    row_list = [j for _, j  in spec_df.iterrows()]
    result_dir = Path(result_dir)

    def extract_num_explained(row): 
        return len(open(row["spec_file"], "r").readlines()) - 1

    peaks_explained = chunked_parallel(row_list, 
                                       extract_num_explained, 100, 
                                       max_cpu=30)

    for row in row_list:
        correct_formula = row["pred_formula"] == row["true_formula"]
        correct_formulas.append(correct_formula)

    spec_df["Correct Formula"] = correct_formulas
    spec_df["Peaks Explained"] = peaks_explained

    summary_df_loc = result_dir / "summary_df.tsv"
    summary_statistics_loc = result_dir / "summary_stats.txt"
    spec_df.sort_values(by="Peaks Explained", inplace=True)
    spec_df.to_csv(summary_df_loc, sep="\t")

    with open(summary_statistics_loc, "w") as fp:
        frac_correct = np.mean(spec_df["Correct Formula"].values)
        fp.write(f"Frac correct formulas: {frac_correct}")


def main(labels_file: str, sirius_folder: str, ):
    """main."""
    # Get all outputs from sirius
    spec_df = extract_top_specs(labels_file, sirius_folder)

    base_dir = Path(sirius_folder)
    # Create output folder
    out_dir = base_dir / "summary_statistics"
    out_dir.mkdir(exist_ok=True)

    # Create summary statistics
    process_statistics(spec_df, out_dir)

def build_new_labels(sirius_summary: str, dataset_name=""):
    """build_new_labels."""

    summary = pd.read_csv(sirius_summary, sep="\t")
    name_to_pred_form = dict(summary[["spec_name", "pred_formula"]].values)
    name_to_pred_adduct = dict(summary[["spec_name", "adduct"]].values)
    name_to_parentmass = dict(summary[["spec_name", "parentmass"]].values)


    new_df = []
    for k in name_to_pred_adduct:
        new_entry = {}
        new_entry['spec'] = k
        new_entry['formula'] = name_to_pred_form[k]
        new_entry['ionization'] = name_to_pred_adduct[k]
        new_entry['dataset'] = dataset_name
        new_entry['compound'] = k
        new_entry['parentmass'] =  name_to_parentmass[k]
        new_df.append(new_entry)

    labels = pd.DataFrame(new_df)
    has_formula = labels[labels["ionization"] != "[M+?]+"]
    has_formula = has_formula.reset_index(drop=True)
    return has_formula



if __name__ == "__main__":
    args = get_args()
    kwargs = args.__dict__
    main(**kwargs)
