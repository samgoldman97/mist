""" pull_examples.py

For help making schematic, generate spectra and mol pairs

"""
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import Draw
from mist import utils

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["font.family"] = "sans-serif"
sns.set(
    context="paper",
    style="white",
    font_scale=2,
    rc={"figure.figsize": (10, 5)},
)

cur_date = datetime.now().strftime("%Y_%m_%d")
output_folder = Path(f"results/{cur_date}_spec_examples")
output_folder.mkdir(exist_ok=True)

dataset = "gnps2015"
labels = Path(f"data/paired_spectra/{dataset}/labels.tsv")
df = pd.read_csv(labels, sep="\t")

# Extract example subs
smi_lens = df["smiles"].apply(len).values
inds = np.arange(len(smi_lens))
inds = inds[np.logical_and(smi_lens > 15, smi_lens < 40)]
examples = np.random.choice(inds, 30)
df_sub = df.loc[examples]

# Can also extract by names
# names = ["CCMSLIB00006699969", "nist_1966820", "CCMSLIB00003134676"]
# inds = [ind for ind, i in enumerate(df['spec'].values)
#        if i in names]
# df_sub = df.loc[inds]

# Export df sub
df_sub.to_csv(output_folder / "sub_df.tsv", sep="\t")

spectra_names = df_sub["spec"]
spectra_formula = df_sub["formula"]
spectra_smiles = df_sub["smiles"]

spec_folder = labels.parent / "spec_files"
spec_files = [spec_folder / f"{i}.ms" for i in spectra_names]

num_bins = 10000
upper_lim = 1000
bins = np.linspace(0, upper_lim, num_bins)

# Parse spec files
parsed_spec_ars = [
    list(zip(*utils.parse_spectra(i)[1]))[1] for i in spec_files if i.exists()
]
np.vstack(parsed_spec_ars[0]).shape
binned = [
    utils.norm_spectrum(
        utils.bin_spectra(
            parsed_spec_ar,
            num_bins=num_bins,
            upper_limit=upper_lim,
        ).mean(0)[None, :]
    ).squeeze()
    for parsed_spec_ar in parsed_spec_ars
]

for binned_spec, smiles, formula, name in zip(
    binned, spectra_smiles, spectra_formula, spectra_names
):
    # Draw molecule
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToImageFile(mol, output_folder / f"{name}_{smiles}.png")

    # Plot spectrum
    fig = plt.figure(figsize=(10, 5))
    ax = fig.gca()
    inds_temp = np.nonzero(binned_spec)[0].flatten()
    for i in inds_temp:
        ax.axvline(bins[i], ymin=0, ymax=binned_spec[i])

    ax.set_xlabel("M/Z")
    ax.set_ylabel("I")
    ax.set_ylim([0, 1.08])
    ax.set_xlim([0, 1000])
    ax.set_title(f"{name}\n{formula}\n{smiles}")

    fig.savefig(output_folder / f"{name}.pdf", bbox_inches="tight")
    fig.clf()
