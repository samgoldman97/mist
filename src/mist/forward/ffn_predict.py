"""ffn_predict.py

Make predictions with trained model

"""

import logging
import yaml
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from mist import utils
from mist.forward import ffn_data, ffn_model, fingerprint


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    parser.add_argument(
        "--save-name", default="results/2022_06_22_pretrain/spec_pred.p"
    )
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
        default="results/2022_06_22_pretrain/version_3/epoch=99-val_loss=0.87.ckpt",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument(
        "--save-tuples",
        default=False,
        action="store_true",
        help="Export as tuples of non zero els",
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__

    debug = kwargs["debug"]
    kwargs["save_dir"] = str(Path(kwargs["save_name"]).parent)
    save_dir = kwargs["save_dir"]
    utils.setup_logger(save_dir, log_name="gnn_pred.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles for pred
    dataset_name = kwargs["dataset_name"]
    if not dataset_name.endswith(".txt"):
        data_dir = Path("data/paired_spectra") / dataset_name
        labels = data_dir / "labels.tsv"
        smiles = pd.read_csv(labels, sep="\t")["smiles"].values
    else:
        smiles = [i.strip() for i in open(dataset_name, "r").readlines()]

    if debug:
        smiles = smiles[:200]

    # Load from checkpoint
    # Create model and load
    best_checkpoint = kwargs["checkpoint_pth"]
    model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")

    # Get train, val, test inds
    num_workers = kwargs.get("num_workers", 0)
    fingerprinter = fingerprint.Fingerprinter(model.fp_type)
    pred_dataset = ffn_data.MolDataset(
        smiles, num_workers=num_workers, fingerprinter=fingerprinter
    )

    print(f"Len of dataset: {len(pred_dataset)}")
    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    gpu = kwargs["gpu"]
    if gpu:
        model = model.cuda()

    names, preds = [], []
    with torch.no_grad():
        for batch in pred_loader:
            fps, batch_names, weights = (
                batch["fps"],
                batch["names"],
                batch["full_weight"],
            )
            if gpu:
                fps = fps.cuda()
                weights = weights.cuda()
            output = model(fps, weights).cpu().detach().numpy()

            if kwargs["save_tuples"]:
                nonzero_out = output > 0.2  # 1e-3  # set 1e-7 thresh
                for j in range(output.shape[0]):
                    inds = np.argwhere(nonzero_out[j]).flatten()
                    preds.append((inds, output[j, inds]))
            else:
                preds.append(output)
            names.append(batch_names)

        names = [j for i in names for j in i]

        if not kwargs["save_tuples"]:
            preds = np.vstack(preds)

        output = {"preds": preds, "names": names}

        with open(kwargs["save_name"], "wb") as fp:
            pickle.dump(output, fp)
