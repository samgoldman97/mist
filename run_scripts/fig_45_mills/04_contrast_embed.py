"""  04_contrast_embed.py

Mills contrast embed

"""
import subprocess
from pathlib import Path

contrastive_model = "results/2022_10_30_contrastive_csi_prospective/2022_10_30-1336_906319_0cec3289d692c28163b57ffecc0f7767/prospective/best.ckpt"

dataset_names = ["mills", "csi2022"]
labels_names = ["labels_with_putative_form.tsv", "labels.tsv"] # Prefix for labels

res_dir = Path("results/2022_12_01_prospective_analysis")
res_dir = Path("results/2023_05_10_prospective_reanalysis_forms/")
num_workers = 16
res_dir.mkdir(exist_ok=True)

res_dir_fp = res_dir / "contrast_embed"
base_script = f"python3 run_scripts/embed_contrastive.py --num-workers {num_workers} --gpu --model {contrastive_model}"


for dataset_name, labels_name  in zip(dataset_names, labels_names):
    new_folder = res_dir / dataset_name
    new_folder.mkdir(exist_ok=True)
    cmd = f"{base_script} --dataset-name {dataset_name} --save-dir {new_folder} --labels-name {labels_name}"
    print(cmd)
    subprocess.run(cmd, shell=True)

# Custom smiles to embed

#smi_encoding_dir = res_dir / "embedded_smiles"
#smi_encoding_dir.mkdir(exist_ok=True)
#ref_encoded_smis = smi_encoding_dir / "embedded_smis.tsv"
#muramyl_smiles = "CC(C(=O)NC(CCC(=O)O)C(=O)N)NC(=O)C(C)OC1C(C(OC(C1O)CO)O)NC(=O)C"
#smiles_to_encode = {"smiles": [muramyl_smiles], 
#                    "inchikey": [utils.inchikey_from_smiles(muramyl_smiles)],
#                    "name": ["muramyl"]}
#pd.DataFrame(smiles_to_encode).to_csv(ref_encoded_smis, sep="\t", index=None)
#
#embed_cmd = f"python run_scripts/embed_smis.py --model {contrastive_model} --gpu --save-dir {smi_encoding_dir} --smiles-list {ref_encoded_smis}"
#subprocess.run(embed_cmd, shell=True)



