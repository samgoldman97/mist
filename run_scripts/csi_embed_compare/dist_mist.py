from pathlib import Path
import subprocess
import pickle
import numpy as np

dataset_name = "csi2022"
labels_file = "data/paired_spectra/csi2022/labels.tsv"
subform_folder = "data/paired_spectra/csi2022/subformulae/subformulae_default/"
num_workers = 10
base_script = f"python3 src/mist/embed_contrast.py --dataset-name {dataset_name} --num-workers {num_workers} --gpu --subset-datasets test_only --labels-file {labels_file} --subform-folder {subform_folder}"
model_ckpt = "results/csi_contrastive_mist/split_0/csi_split_0/best.ckpt"

savedir = Path("results/csi_dist_compare/mist_embed/")
savedir.mkdir(exist_ok=True)
cmd = f"{base_script} --model-ckpt {str(model_ckpt)} --save-dir {savedir}"
subprocess.call(cmd, shell=True)

embed_file = savedir / "embed_csi2022.p"
output_file = savedir.parent / "mist_csi2022_split_0.p"

embedded_output = pickle.load(open(embed_file, "rb"))
new_embeddings = embedded_output["embeds"]
names = embedded_output["names"]

# Compare pairwise cosine similarities with new embeddings
mults = new_embeddings[:, None, :] * new_embeddings[None, :, :]
norms = np.linalg.norm(new_embeddings, axis=1)
norms_pairs = norms[:, None] * norms[None, :] + 1e-8
pair_sims = np.sum(mults, axis=2) / norms_pairs

percentile = 0.01
num_pairs = np.triu(np.ones((len(names), len(names))), 1).sum()
keep_pairs = int(percentile * num_pairs)

# Set upper triangle to 0
pair_sims[np.triu_indices(pair_sims.shape[0])] = 0

# Get indices of top pairs
most_sim_inds = np.argsort(pair_sims.ravel())[::-1][:keep_pairs]
most_sim_vals = pair_sims.ravel()[most_sim_inds]

# Get row and col inds by unraveling
row_inds, col_inds = np.unravel_index(most_sim_inds, pair_sims.shape)
pair_1 = np.array(names)[row_inds]
pair_2 = np.array(names)[col_inds]
top_pairs_sorted = list(zip(pair_1, pair_2))

out = {
    "top_pairs_sorted": top_pairs_sorted,
    "model": "mist",
}
with open(output_file, "wb") as f:
    pickle.dump(out, f)
