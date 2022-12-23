"""
num_bins_simulation.py

Validate our choice of number of bins in the forward problem by computing the
number of collisions at various number of bins

"""
import numpy as np
import pandas as pd

from mist import utils


def bin_frags(frags, upper_limit=1000, num_bins=1000):
    bins = np.linspace(0, upper_limit, num=num_bins)
    # Convert to digitized spectra
    digitized = np.digitize(frags, bins=bins)
    return digitized


if __name__ == "__main__":
    labels = "data/paired_spectra/csi2022/labels.tsv"
    formulae = set(pd.unique(pd.read_csv(labels, sep="\t")["formula"]))

    formulae = list(formulae)

    ub = 15000
    nb = 10000

    for form in formulae:
        cross_prod, masses = utils.get_all_subsets(form)
        digitized = bin_frags(masses, upper_limit=ub, num_bins=nb)

        # Get bincount
        counts = np.bincount(digitized, minlength=ub)
        raise NotImplementedError()
