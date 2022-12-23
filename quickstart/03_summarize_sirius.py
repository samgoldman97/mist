""" 03_summarize_sirius

Converts sirius outputs into summary file and
adds putative chem formula from sirius to the quickstart labels file


"""
from pathlib import Path
import pandas as pd
import mist.sirius.summarize_sirius as summarize_sirius


def main():
    """main process"""
    extract_name = "quickstart"
    input_dir = Path(f"data/paired_spectra/{extract_name}")
    sirius_out = input_dir / "sirius_outputs"
    labels_file = input_dir / "labels.tsv"
    labels_file_form_out = input_dir / "labels_with_putative_form.tsv"
    sirius_summary = sirius_out / "summary_statistics/summary_df.tsv"

    summarize_sirius.main(labels_file=labels_file,
                          sirius_folder=sirius_out)

    labels = pd.read_csv(labels_file, sep="\t")
    summary = pd.read_csv(sirius_summary, sep="\t")

    name_to_pred_form = dict(summary[["spec_name", "pred_formula"]].values)
    name_to_pred_adduct = dict(summary[["spec_name", "adduct"]].values)

    pred_form = [name_to_pred_form.get(i, "") for i in labels["spec"]]
    ionization = [name_to_pred_adduct.get(i, "[M+?]+") for i in labels["spec"]]

    labels["formula"] = pred_form
    labels["ionization"] = ionization

    has_formula = labels[labels["ionization"] != "[M+?]+"]
    has_formula = has_formula.reset_index(drop=True)
    has_formula.to_csv(labels_file_form_out, sep="\t", index=None)


if __name__ == "__main__":
    main()
