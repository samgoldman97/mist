"""createe_smi_output.py

Given a set of retrieval distances, this file creates a full prediction file
with the top k smiles from database retrieval.


Cmd:

```
python3 analysis/retrieval/create_smi_output.py --names-file data/paired_spectra/csi2022/retrieval_hdf/intpubchem_with_morgan4096_retrieval_db_names.p --ranking results/2022_08_27_mist_no_aug_morgan/2022_08_27-1256_011159_589fafcd07beca2529bd6deab4354946/retrieval/retrieval_fp_intpubchem_with_morgan4096_retrieval_db_csi2022_cosine_0.p --k 10
```
"""
import pickle
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from mist import utils

parser = argparse.ArgumentParser()
parser.add_argument("--names-file", help="Name file linking db rankings to names")
parser.add_argument("--ranking", help="Pred ranking file")
parser.add_argument("--k", help="Num to export", default=10, type=int)
parser.add_argument(
    "--save-name", help="Name of save output", action="store", default=None
)
args = parser.parse_args()

name_file = Path(args.names_file)
pred_ranking = Path(args.ranking)
save_name = args.save_name

if save_name is None:
    save_name = pred_ranking.parent / f"{pred_ranking.stem}_k_smi.tsv"

save_name = Path(save_name)
save_name.parent.mkdir(exist_ok=True)
k = args.k

if not pred_ranking.exists():
    print(f"File {pred_ranking} does not exist")

if not name_file.exists():
    print(f"File {name_file} does not exist")

# Load in the true ranking file
with open(name_file, "rb") as fp:
    ind_to_smi = pickle.load(fp)

with open(pred_ranking, "rb") as fp:
    pred_ranking_outs = pickle.load(fp)

name_to_ranking = dict(zip(pred_ranking_outs["names"], pred_ranking_outs["ranking"]))
name_to_dists = dict(zip(pred_ranking_outs["names"], pred_ranking_outs["dists"]))

out_entries = []
for name, ranking in tqdm(name_to_ranking.items()):
    subsets = name_to_ranking.get(name)
    dists = name_to_dists.get(name)

    order = np.argsort(dists)
    dists_sorted = dists[order]
    inds_sorted = subsets[order]
    top_k = inds_sorted[:k]
    top_k_smi = [ind_to_smi[i] for i in top_k]

    for smi_rank, smi in enumerate(top_k_smi):
        form = utils.form_from_smi(smi)
        out_entry = {
            "name": name,
            "form": form.split("+")[0].split("-")[0],
            "smi": smi,
            "rank": smi_rank,
        }
        out_entries.append(out_entry)

df = pd.DataFrame(out_entries)
df.to_csv(save_name, sep="\t", index=None)
