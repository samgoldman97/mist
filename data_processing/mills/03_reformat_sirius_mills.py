""" 03_reformat_sirius_mills.py

Reformat the results from the sirius outptus

"""
from pathlib import Path

dataset_name = "mills"
sirius_dir = Path("data/paired_spectra/mills/sirius_outputs")


def main():
    for i in sirius_dir.glob("*"):
        if "mgf_export_sirius" in i.stem:
            compound_id = i.stem.rsplit("_", 1)[-1]
            orig_id = i.stem.split("_")[0]
            # Renumber arbitrary
            new_name = i.parent / f"{orig_id}_mills_{compound_id}"
            i.rename(new_name)

        # if "mills_" in i.stem:
        #    # Renumber arbitrary
        #    new_name = i.parent / f"0_{i.stem}"
        #    print(new_name)

        #    i.rename(new_name)


if __name__ == "__main__":
    main()
