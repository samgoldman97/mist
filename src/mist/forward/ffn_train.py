"""ffn_train.py

Train ffn to predict binned specs

"""
import logging
import yaml
import json
import argparse
from pathlib import Path
import pandas as pd

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from mist import utils
from mist.forward import ffn_data, ffn_model, splitter, fingerprint


def add_forward_args(parser):
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--scheduler", default=False, action="store_true")
    parser.add_argument("--lr-decay-rate", default=0.9)
    parser.add_argument("--learning-rate", default=7e-4, action="store", type=float)
    parser.add_argument("--seed", default=42, action="store", type=int)
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=128, action="store", type=int)
    parser.add_argument("--max-epochs", default=300, action="store", type=int)
    parser.add_argument("--save-dir", default="results/2022_06_22_pretrain/")

    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--split-name", default="csi_split.txt")

    parser.add_argument("--num-bins", default=1000, action="store", type=int)
    parser.add_argument("--upper-limit", default=1500, action="store", type=int)
    parser.add_argument("--layers", default=3, action="store", type=int)
    parser.add_argument("--dropout", default=0.1, action="store", type=float)
    parser.add_argument("--hidden-size", default=256, action="store", type=int)
    parser.add_argument("--use-reverse", default=False, action="store_true")
    parser.add_argument("--overfit-train", default=False, action="store_true")
    parser.add_argument("--loss-fn", default="cosine", choices=["cosine", "bce"])
    parser.add_argument(
        "--fp-type",
        default="morgan4096_2",
        choices=[
            "csi",
            "morgan2048_2",
            "morgan2048_3",
            "morgan4096_2",
            "morgan4096_3",
            "morgan_form",
        ],
    )
    parser.add_argument("--growing", default="none", action="store", type=str)
    parser.add_argument("--growing-weight", default=0.5, action="store", type=float)
    parser.add_argument("--growing-layers", default=3, action="store", type=int)
    parser.add_argument(
        "--growing-scheme", default="interleave", action="store", type=str
    )


def get_args():
    parser = argparse.ArgumentParser()
    add_forward_args(parser)
    return parser.parse_args()


def train_model():
    args = get_args()
    kwargs = args.__dict__

    save_dir = kwargs["save_dir"]
    utils.setup_logger(save_dir, log_name="ffn_train.log", debug=kwargs["debug"])
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2)
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    logging.info(json.dumps(kwargs, indent=2))

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/paired_spectra") / dataset_name
    labels = data_dir / "labels.tsv"
    split_file = data_dir / "splits" / kwargs["split_name"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")
    spec_names = df["spec"].values
    train_inds, val_inds, test_inds = splitter.get_splits(spec_names, split_file)
    train_df = df.iloc[train_inds]
    val_df = df.iloc[val_inds]
    test_df = df.iloc[test_inds]

    if kwargs.get("debug", False):
        train_df = train_df[:20]
        val_df = val_df[:20]
        test_df = test_df[:20]

    num_bins = kwargs.get("num_bins")
    upper_limit = kwargs.get("upper_limit")
    num_workers = kwargs.get("num_workers", 0)
    logging.info("Making datasets")
    # Remove duplicates!
    # train_df = train_df.drop_duplicates(subset="inchikey").reset_index(drop=True)
    fingerprinter = fingerprint.Fingerprinter(
        kwargs["fp_type"], dataset_name=dataset_name
    )
    train_dataset = ffn_data.BinnedDataset(
        train_df,
        data_dir=data_dir,
        num_bins=num_bins,
        upper_limit=upper_limit,
        fingerprinter=fingerprinter,
        num_workers=num_workers,
    )
    val_dataset = ffn_data.BinnedDataset(
        val_df,
        data_dir=data_dir,
        fingerprinter=fingerprinter,
        num_bins=num_bins,
        upper_limit=upper_limit,
        num_workers=num_workers,
    )
    test_dataset = ffn_data.BinnedDataset(
        test_df,
        data_dir=data_dir,
        upper_limit=upper_limit,
        fingerprinter=fingerprinter,
        num_bins=num_bins,
        num_workers=num_workers,
    )
    # Define dataloaders
    collate_fn = train_dataset.get_collate_fn()
    train_loader = DataLoader(
        train_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,  # Temp turn off shuffle
        batch_size=kwargs["batch_size"],
    )
    val_loader = DataLoader(
        val_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )
    test_loader = DataLoader(
        test_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    # Define model
    test_batch = next(iter(train_loader))
    logging.info("Building model")
    model = ffn_model.ForwardFFN(
        hidden_size=kwargs["hidden_size"],
        layers=kwargs["layers"],
        dropout=kwargs["dropout"],
        output_dim=num_bins,
        use_reverse=kwargs["use_reverse"],
        learning_rate=kwargs["learning_rate"],
        decay_rate=kwargs["lr_decay_rate"],
        scheduler=kwargs["scheduler"],
        upper_limit=kwargs["upper_limit"],
        loss_name=kwargs["loss_fn"],
        fp_type=kwargs["fp_type"],
        input_dim=fingerprinter.get_nbits(),
        growing=kwargs["growing"],
        growing_weight=kwargs["growing_weight"],
        growing_layers=kwargs["growing_layers"],
        growing_scheme=kwargs["growing_scheme"],
    )

    # outputs = model(test_batch['fps'])

    # Create trainer
    tb_logger = pl_loggers.TensorBoardLogger(save_dir, name="")
    console_logger = utils.ConsoleLogger()

    if args.overfit_train:
        monitor = "train_loss_epoch"
    else:
        monitor = "val_loss"

    tb_path = tb_logger.log_dir
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=tb_path,
        filename="{epoch}-{val_loss:.5f}",
        save_weights_only=False,
        # every_n_epochs=10,
    )
    earlystop_callback = EarlyStopping(monitor=monitor, patience=10)
    callbacks = [earlystop_callback, checkpoint_callback]

    logging.info("Starting train")
    trainer = pl.Trainer(
        logger=[tb_logger, console_logger],
        accelerator="gpu" if kwargs["gpu"] else None,
        devices=1 if kwargs["gpu"] else None,
        callbacks=callbacks,
        gradient_clip_val=5,
        max_epochs=kwargs["max_epochs"],
        gradient_clip_algorithm="value",
        check_val_every_n_epoch=10 if args.overfit_train else 1,
    )

    trainer.fit(model, train_loader, val_loader)
    checkpoint_callback = trainer.checkpoint_callback
    best_checkpoint = checkpoint_callback.best_model_path
    best_checkpoint_score = checkpoint_callback.best_model_score.item()

    # Load from checkpoint
    model = ffn_model.ForwardFFN.load_from_checkpoint(best_checkpoint)
    logging.info(
        f"Loaded model with from {best_checkpoint} with val loss of {best_checkpoint_score}"
    )

    model.eval()
    test_out = trainer.test(dataloaders=test_loader)
    out_yaml = {"args": kwargs, "test_metrics": test_out[0]}
    out_str = yaml.dump(out_yaml, indent=2, default_flow_style=False)

    with open(Path(save_dir) / "test_results.yaml", "w") as fp:
        fp.write(out_str)
