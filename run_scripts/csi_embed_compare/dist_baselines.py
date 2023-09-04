""" run_baselines.py

Run methods
- ms2deepscore
- modified cosine
- spec2vec
"""

from tqdm import tqdm
from pathlib import Path

import pickle
import numpy as np
import pandas as pd
from mist.data import featurizers

# Matchms filtering
import matchms
from matchms import importing
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_intensity
from matchms.filtering import select_by_mz
import matchms.filtering as msfilters

from matchms.similarity import ModifiedCosine
from matchms import calculate_scores

import gensim
from spec2vec import Spec2Vec

import ms2deepscore
from ms2deepscore.models import load_model

matchms.set_matchms_logger_level(loglevel="ERROR")


def peak_processing(spectrum):
    """peak_processing.

    Taken directly from tutorial

    """
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = select_by_mz(spectrum, mz_from=10, mz_to=1500)
    return spectrum


if __name__ == "__main__":
    debug = False
    outfolder = Path("results/csi_dist_compare/")
    model_folder = outfolder / "saved_models"
    f = "data/paired_spectra/csi2022/csi2022.mgf"
    f_split = "data/paired_spectra/csi2022/splits/csi_split_0.tsv"
    labels = "data/paired_spectra/csi2022/labels.tsv"
    labels_df = pd.read_csv(labels, sep="\t")
    name_to_smi = dict(zip(labels_df["spec"], labels_df["smiles"]))

    split_df = pd.read_csv(f_split, sep="\t")
    test_names = set(split_df[split_df["split"] == "test"]["name"].values)
    input_specs = importing.load_from_mgf(f, metadata_harmonization=True)
    new_names, new_specs = [], []
    for ind, i in enumerate(tqdm(input_specs)):
        if debug and ind > 100:
            break

        f_name = i.metadata["_file"]
        if f_name not in test_names:
            continue

        new_specs.append(i)
        new_names.append(f_name)

    percentile = 0.01
    input_specs, names = new_specs, new_names
    num_pairs = np.triu(np.ones((len(names), len(names))), 1).sum()
    keep_pairs = int(percentile * num_pairs)
    spectra = [peak_processing(s) for s in input_specs]
    # for model_name in ["ms2deepscore", "spec2vec", "cosine", "random", "tanimoto"]:
    for model_name in ["tanimoto"]:
        out_file = outfolder / f"{model_name}_csi2022.p"
        if model_name == "spec2vec":
            spec_model = model_folder / "spec2vec.model"
            model = gensim.models.Word2Vec.load(str(spec_model))
            embed_model = Spec2Vec(
                model=model,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=5.0,
            )
            embed_fn = lambda x, y: embed_model._calculate_embedding(x)
            pairwise = False
        elif model_name == "ms2deepscore":
            spec_model = (
                model_folder / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
            )
            tf_model = load_model(str(spec_model))
            model = ms2deepscore.MS2DeepScore(tf_model)
            embed_fn = lambda x, y: model.calculate_vectors([x])[0]
            pairwise = False
        elif model_name == "cosine":
            similarity_measure = ModifiedCosine(tolerance=0.005)
            pairwise_fn = lambda x, y: calculate_scores(
                x, x, similarity_measure, is_symmetric=True
            ).scores["score"]
            pairwise = True
        elif model_name == "random":
            pairwise_fn = lambda x, y: np.random.rand(len(x), len(x))
            pairwise = True
        elif model_name == "tanimoto":

            def pairwise_fn(specs, names):
                featurizer = featurizers.FingerprintFeaturizer(fp_names=["morgan2048"])
                test_fps = [featurizer.featurize_smiles(name_to_smi[i]) for i in names]
                test_fps = np.vstack(test_fps)
                intersect = np.einsum("ij,kj->ik", test_fps, test_fps)
                union = (
                    test_fps.sum(-1)[None, :] + test_fps.sum(-1)[:, None] - intersect
                )
                tani_sim = intersect / union
                return tani_sim

            pairwise = True
        else:
            raise ValueError("Unknown model")

        if pairwise:
            pair_sims = pairwise_fn(spectra, names)
        else:
            new_embeddings = []
            new_names = []
            for i, j in tqdm(zip(spectra, names)):
                if i is None:
                    continue
                new_embedding = embed_fn(i, j)
                new_embeddings.append(new_embedding)
                new_names.append(j)
            new_embeddings = np.vstack(new_embeddings)

            # Compare pairwise cosine similarities with new embeddings
            mults = new_embeddings[:, None, :] * new_embeddings[None, :, :]
            norms = np.linalg.norm(new_embeddings, axis=1)
            norms_pairs = norms[:, None] * norms[None, :] + 1e-8
            pair_sims = np.sum(mults, axis=2) / norms_pairs

        # Set upper triangle to 0
        pair_sims[np.triu_indices(pair_sims.shape[0])] = 0

        # Get indices of top pairs
        most_sim_inds = np.argsort(pair_sims.ravel())[::-1][:keep_pairs]
        most_sim_vals = pair_sims.ravel()[most_sim_inds]

        # Get row and col inds by unraveling
        row_inds, col_inds = np.unravel_index(most_sim_inds, pair_sims.shape)
        pair_1 = np.array(new_names)[row_inds]
        pair_2 = np.array(new_names)[col_inds]
        top_pairs_sorted = list(zip(pair_1, pair_2))

        out = {
            "top_pairs_sorted": top_pairs_sorted,
            "model": model_name,
        }
        out_file = outfolder / f"{model_name}_csi2022_split_0.p"
        with open(out_file, "wb") as f:
            pickle.dump(out, f)
