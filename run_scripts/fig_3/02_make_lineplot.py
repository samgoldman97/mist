"""02_make_lineplot
"""
import subprocess
from pathlib import Path

contrast_dir = "results/2022_10_27_contrastive_best_ensemble"
num_workers = 16
dataset_name = "csi2022"
labels_file = f"data/paired_spectra/{dataset_name}/labels.tsv"
save_dir = Path(contrast_dir) / "embeds"
save_dir.mkdir(exist_ok=True)
print(save_dir)

contrast_outdir = Path(contrast_dir) / f"out_retrieval_all"
contrast_outdir.mkdir(exist_ok=True)
for ckpt in Path(contrast_dir).rglob("*.ckpt"):
    if "Fold_0" in str(ckpt):
        cmd = f"python3 run_scripts/embed_contrastive.py --gpu --num-workers {num_workers} --dataset-name csi2022 --model-ckpt {ckpt} --save-dir {save_dir} --subset test_only"
        print(cmd)
        subprocess.run(cmd, shell=True)
        embed_out = list(save_dir.glob("*.p"))[0]
        break
    else:
        pass

# Make lineplot
lineplot_cmd = f"python analysis/embedding/tani_lineplot.py --mist-embed-file {embed_out} --save-name {save_dir / 'sim_lineplot.pdf'}"
print(lineplot_cmd)
subprocess.run(lineplot_cmd, shell=True)

umap_cmd = f"python analysis/embedding/embed_umap.py --umap-embeddings {embed_out} --save-name {save_dir / 'umap.png'} --png"
#subprocess.run(umap_cmd, shell=True)
