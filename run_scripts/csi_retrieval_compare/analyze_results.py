""" Analyze ranking outputs """
import yaml
import pandas as pd
from pathlib import Path
import subprocess
import re
import yaml

ranking_files = [
    "results/csi_retrieval_compare/mist_fp_retrieval/csi_split_0/retrieval_fp_pubchem_with_csi_retrieval_db_csi2022_cosine.p",
    "results/csi_retrieval_compare/mist_contrastive_retrieval/csi_split_0/retrieval_contrastive_pubchem_with_csi_retrieval_db_csi2022_cosine.p",
    "results/csi_retrieval_compare/sirius_fp_retrieval/csi_split_0/retrieval_fp_pubchem_with_csi_retrieval_db_csi2022_cosine.p",
]

labels = "data/paired_spectra/csi2022/labels.tsv"
analysis_base = f"python analysis/analyze_rankings.py --labels {labels}"

for ranking_file in ranking_files:
    ranking_file = Path(ranking_file)
    out_file = ranking_file.parent / f"{ranking_file.stem}_analysis.yaml"
    cmd = f"{analysis_base} --rankings {ranking_file} --output {out_file}"
    print(cmd)
    subprocess.call(cmd, shell=True)
