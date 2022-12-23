""" 03_add_putative_formulae.py """

import pandas as pd

dataset_name = "mills"
labels_file = f"data/paired_spectra/{dataset_name}/labels.tsv"
labels_file_out = f"data/paired_spectra/{dataset_name}/labels_putative.tsv"
labels_file_hplus_out = f"data/paired_spectra/{dataset_name}/labels_putative_h_plus.tsv"
labels_file_form_out = (
    f"data/paired_spectra/{dataset_name}/labels_with_putative_form.tsv"
)
sirius_summary = f"data/paired_spectra/{dataset_name}/sirius_outputs/summary_statistics/summary_df.tsv"


def main():
    labels = pd.read_csv(labels_file, sep="\t")
    summary = pd.read_csv(sirius_summary, sep="\t")

    name_to_pred_form = dict(summary[["spec_name", "pred_formula"]].values)
    name_to_pred_adduct = dict(summary[["spec_name", "adduct"]].values)

    pred_form = [name_to_pred_form.get(i, "") for i in labels["spec"]]
    ionization = [name_to_pred_adduct.get(i, "[M+?]+") for i in labels["spec"]]

    labels["formula"] = pred_form
    labels["ionization"] = ionization
    labels.to_csv(labels_file_out, sep="\t", index=None)
    h_plus_labels = labels[labels["ionization"] == "[M+H]+"].reset_index(drop=True)
    h_plus_labels.to_csv(labels_file_hplus_out, sep="\t", index=None)

    has_formula = labels[labels["ionization"] != "[M+?]+"].reset_index(drop=True)
    has_formula.to_csv(labels_file_form_out, sep="\t", index=None)


if __name__ == "__main__":
    main()
