""" 05_max_sim_calcualtor.py

Calculate the maximum similarity between biomolecules and the forward training
set to try to filter down based upon tanimoto similarity

"""

from pathlib import Path
import pandas as pd
import pickle

from mist import utils
from mist.data import featurizers
import numpy as np
from tqdm import tqdm

from functools import partial


def main():

    debug = False
    labels_file = "data/paired_spectra/csi2022/labels.tsv"
    unpaired_file = "data/unpaired_mols/bio_mols/all_smis_unfiltered.txt"
    unpaired_file_path = Path(unpaired_file)
    dataset = Path(labels_file).parent.name

    # Bank of molecules to compare as ref
    labels_df = pd.read_csv(labels_file, sep="\t")
    smiles_bank = list(pd.unique(labels_df["smiles"]))

    # Test molecules
    smiles_test = [i.strip() for i in open(unpaired_file, "r").readlines()]

    if debug:
        smiles_bank = smiles_bank
        smiles_test = smiles_test[:20000]

    # Create fingerprints of the entire bank of smiles
    featurizer = featurizers.FingerprintFeaturizer(fp_names=["morgan1024"])
    # bank_fps = np.vstack([featurizer.featurize_smiles(i) for i in smiles_bank])
    bank_fps = utils.chunked_parallel(smiles_bank, featurizer.featurize_smiles)
    bank_fps = np.vstack(bank_fps)

    # Create fingerprints of entire test bank
    test_fps = utils.chunked_parallel(smiles_test, featurizer.featurize_smiles)
    test_fps = np.vstack(test_fps)

    test_fp_smi_pairs = list(zip(smiles_test, test_fps))

    def batch_calculation(input_batch: tuple, featurizer, bank_fps):
        """batch_calculation"""
        test_smis, test_fps = zip(*input_batch)
        fp_x = bank_fps
        fp_y = np.vstack(test_fps)
        einsum_intersect = np.einsum("x i, y i -> xy", fp_x, fp_y)
        einsum_union = fp_x.sum(-1)[:, None] + fp_y.sum(-1)[None, :]
        einsum_union_less_intersect = einsum_union - einsum_intersect
        tani_pairwise = einsum_intersect / einsum_union_less_intersect
        tani_max = tani_pairwise.max(0)
        temp_dict = dict(zip(test_smis, tani_max))
        return temp_dict

    partial_fn = partial(batch_calculation, bank_fps=bank_fps, featurizer=featurizer)
    batch_size = 500
    parallel_input = list(utils.batches(test_fp_smi_pairs, batch_size))

    out_map_list = utils.chunked_parallel(parallel_input, partial_fn, max_cpu=10)
    out_map = {}
    for out_map_temp in out_map_list:
        out_map.update(out_map_temp)

    with open(unpaired_file_path.parent / f"max_sim_{dataset}.p", "wb") as fp:
        pickle.dump(out_map, fp)


if __name__ == "__main__":
    main()
