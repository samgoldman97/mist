""" Analyze ranking outputs """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

ranking_files = [
    "results/canopus_retrieval_compare/ffn_fp_retrieval/canopus_hplus_100_0/retrieval_fp_intpubchem_with_morgan4096_retrieval_db_canopus_train_cosine.p",
    "results/canopus_retrieval_compare/mist_fp_retrieval/canopus_hplus_100_0/retrieval_fp_intpubchem_with_morgan4096_retrieval_db_canopus_train_cosine.p",
    "results/canopus_retrieval_compare/mist_contrastive_retrieval/canopus_hplus_100_0/retrieval_contrastive_intpubchem_with_morgan4096_retrieval_db_canopus_train_cosine.p",
]

labels = "data/paired_spectra/csi2022/labels.tsv"
analysis_base = f"python analysis/analyze_rankings.py --labels {labels}"

for ranking_file in ranking_files:
    ranking_file = Path(ranking_file)
    out_file = ranking_file.parent / f"{ranking_file.stem}_analysis.yaml"
    cmd = f"{analysis_base} --rankings {ranking_file} --output {out_file}"
    print(cmd)
    subprocess.call(cmd, shell=True)
