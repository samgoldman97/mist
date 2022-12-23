""" 01_reformat_mills

Convert protect dataset into rename

Outputs:
1. spec files
2. Labels

"""
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from mist import utils

DEBUG = False

MILLS_MGF = Path("data/raw/mills/Mills_mzxml/gnps_fbn_export.mgf")
TARGET_DIRECTORY = Path("data/paired_spectra/mills/")
TARGET_SPEC = TARGET_DIRECTORY / "spec_files"
TARGET_LABELS = TARGET_DIRECTORY / "labels.tsv"


def main():
    """main process"""

    # Make target directories
    TARGET_DIRECTORY.mkdir(exist_ok=True)
    TARGET_SPEC.mkdir(exist_ok=True)

    # Load meta file
    mgf_names = ["mills"]  # ["mills"]
    mgf_files = [MILLS_MGF]
    entries = []
    for mgf_name, mgf_file in zip(mgf_names, mgf_files):
        parsed_spec = utils.parse_spectra_mgf(
            mgf_file, max_num=None if not DEBUG else 10
        )
        for spec_meta, spec in tqdm(parsed_spec):
            feat_id = int(spec_meta["FEATURE_ID"])
            out_spec_id = f"{mgf_name}_{feat_id}"
            mass = spec_meta["PEPMASS"]
            full_loc = TARGET_SPEC / f"{out_spec_id}.ms"
            with open(full_loc, "w") as fp:
                # Need to have an MS MS write file
                essential_keys = {
                    "compound": out_spec_id,
                    "ionization": "[M+?]+",
                    "parentmass": mass,
                }
                out_str = utils.spec_to_ms_str(
                    spec=spec, essential_keys=essential_keys, comments=spec_meta
                )
                fp.write(out_str)

            # Create output label entries
            entry = {
                "dataset": mgf_name,
                "spec": out_spec_id,
            }
            entries.append(entry)
        df = pd.DataFrame(entries)
        df.to_csv(TARGET_LABELS, sep="\t", index=False)


if __name__ == "__main__":
    main()
