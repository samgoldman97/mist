""" reformat_sirius_mills_revision.py

Given the revised version and output of the mills dataset, go through and pull
the new chemical formulae annotations for structure and attribute them.

"""
import pandas as pd
import shutil
from pathlib import Path
from mist import utils


dataset_name = "mills"
sirius_targ_dir = Path("data/paired_spectra/mills/sirius_outputs")
sirius_targ_dir.mkdir(exist_ok=True)

sirius_src_dir = Path("results/2023_05_08_sirius_check/sirius5s_out/")
sirius_src_summary_dir = Path("results/2023_05_08_sirius_check/sirius5s_out_summary/")


def rename_entrant(name):
    compound_id = name.rsplit("_", 1)[-1]
    orig_id = name.split("_")[0]
    return f"{compound_id}_mills_{compound_id}"


def main():
    # Step 1 can we read in the formula annotations from structure
    annots = sirius_src_summary_dir / "compound_identifications_adducts.tsv"
    annots = sirius_src_summary_dir / "compound_identifications.tsv"
    df = pd.read_csv(annots, sep="\t")
    new_outs = []
    for spec_id, formula, adduct, smiles in df[["id", "molecularFormula",
                                                "adduct", "smiles"]].values:
        adduct = adduct.replace(' ', '')
        if adduct != "[M+H]+": continue
        new_name = rename_entrant(spec_id)
        old_folder = sirius_src_dir / spec_id 

        #print(list(old_folder.glob("*")))
        json_save = old_folder / f"trees/{formula}_{adduct}.json"
        tbl_save = old_folder / f"spectra/{formula}_{adduct}.tsv"

        # Start to copy these into the new folder and make them consistent with
        # sirius outputs...
        new_dir = sirius_targ_dir / new_name
        new_tree_loc = new_dir / f"trees/{json_save.name}"
        new_spec_loc = new_dir / f"spectra/{tbl_save.name}"

        new_spec_loc.parent.mkdir(parents=True, exist_ok=True)
        new_tree_loc.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(json_save, new_tree_loc)
        shutil.copy2(tbl_save, new_spec_loc)

        num, file_id = new_name.split("_", 1)

        new_entry = {
            "spec_name": file_id,
            "spec_file": new_spec_loc,
            "tree_file": new_tree_loc,
            "adduct": adduct, 
            "pred_formula": formula,
            "true_formula": formula,
        }
        new_outs.append(new_entry)


    out_df = pd.DataFrame(new_outs)
    out_summary = sirius_targ_dir / "summary_statistics/summary_df.tsv"
    out_summary.parent.mkdir(exist_ok=True)
    out_df.to_csv(out_summary, sep="\t")


if __name__ == "__main__":
    main()
