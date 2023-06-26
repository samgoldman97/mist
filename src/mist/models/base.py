""" base.py

This file will contain all common operations for the models
including certain superclasses of models

"""
import logging
from typing import List, Optional

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from mist.data.datasets import SpectraMolDataset, SpecDataModule
from mist import utils

# Define model_types
model_registry = {}


def register_model(cls):
    """register_model.

    Add an argument to the model_registry.
    Use this as a decorator on classes defined in the rest of the directory.

    """
    model_registry[cls.__name__] = cls
    return cls


def get_model_class(model, **kwargs):
    return model_registry[model]


def build_model(model, **kwargs):
    """build_model"""
    return get_model_class(model)(**kwargs)


class TorchModel(pl.LightningModule):
    """TorchModel.

    Parent class to hold SpectraModels.

    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        min_lr: float = 1e-6,
        weight_decay: float = 0.0,
        optim_name: str = "adam",
        lr_decay_frac: float = 0.99,
        scheduler: bool = False,
        lr_decay_time: int = 10000,
        **kwargs,
    ):
        """Call init"""
        pl.LightningModule.__init__(self)
        self.learning_rate = learning_rate
        self.optim_name = optim_name
        self.results_dir = ""
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.lr_decay_time = lr_decay_time
        self.lr_decay_frac = lr_decay_frac

    @staticmethod
    def dataset_type(mode: Optional[str] = None) -> str:
        """dataset_type."""
        return "default"

    def configure_optimizers(self):
        if self.optim_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optim_name == "radam":
            from radam.radam import RAdam

            optimizer = RAdam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        ## Scheduler
        if not self.scheduler:
            return optimizer
        else:
            # Step learning rate scheduler
            decay_rate = self.lr_decay_frac
            start_lr = self.learning_rate
            steps_to_decay = self.lr_decay_time
            min_decay_rate = self.min_lr / start_lr
            lr_lambda = lambda epoch: (
                np.maximum(decay_rate ** (epoch // steps_to_decay), min_decay_rate)
            )
            interval = "step"
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": interval,
                },
            }
            return ret

    def forward(self, batch):
        raise NotImplementedError()

    def set_results_dir(self, dir_):
        """Set the results dir to store misc info"""
        self.results_dir = dir_

    def batch_to_device(self, batch: dict) -> None:
        """batch_to_device.

        Convert batch tensors to same device as the model


        Args:
            batch (dict): Batch from data loader

        """
        # Port to cuda
        device = self.device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = batch[key].to(device)

    def encode_all_spectras(
        self, spectras: SpectraMolDataset, no_grad=False, logits=False, **kwargs
    ) -> torch.tensor:
        """encode_all_spectras."""
        with torch.set_grad_enabled(not no_grad):
            spectra_loader = SpecDataModule.get_paired_loader(
                spectras, shuffle=False, **kwargs
            )
            spectra_outputs = []
            for spectra_batch in spectra_loader:
                self.batch_to_device(spectra_batch)

                # Convert to cpu!
                if not logits:
                    model_encoding = self.encode_spectra(spectra_batch)[0].cpu()
                else:
                    model_encoding = self.encode_spectra(
                        spectra_batch,
                        logits=True,
                    )[0].cpu()

                spectra_outputs.append(model_encoding)

            stacked_spectra = torch.cat(spectra_outputs, 0)
        return stacked_spectra

    def encode_all_mols(
        self, mol_library: SpectraMolDataset, no_grad=False, **kwargs
    ) -> torch.tensor:
        """encode_all_mols."""

        # Modify to use only mol  loader

        with torch.set_grad_enabled(not no_grad):
            mol_loader = SpecDataModule.get_mol_loader(
                mol_library, shuffle=False, **kwargs
            )
            mol_outputs = []
            for mol_batch in mol_loader:
                self.batch_to_device(mol_batch)

                # Convert to cpu!
                model_encoding = self.encode_mol(mol_batch)[0].cpu()
                mol_outputs.append(model_encoding)

            stacked_mols = torch.Tensor([])
            if len(mol_outputs) > 0:
                stacked_mols = torch.cat(mol_outputs, 0)
        return stacked_mols

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def post_train_modify(self, _checkpoint_callback, debug=False, **kwargs):
        """On fit end, load the best checkpoint"""

        debug_overfit = debug == "test_overfit"
        if debug_overfit:
            logging.info("Not loading best model")
            return

        # Load from checkpoint
        best_checkpoint = _checkpoint_callback.best_model_path

        best_checkpoint_score = None
        if _checkpoint_callback.best_model_score is not None:
            best_checkpoint_score = _checkpoint_callback.best_model_score.item()

        loaded_checkpoint = torch.load(best_checkpoint)
        best_epoch = loaded_checkpoint["epoch"]

        logging.info(
            f"Loading from epoch {best_epoch} with val loss of {best_checkpoint_score}"
        )

        # model = model.load_from_checkpoint(best_checkpoint)
        self.load_state_dict(loaded_checkpoint["state_dict"])

    def load_from_ckpt(self, ckpt_file, **kwargs):

        loaded_checkpoint = torch.load(ckpt_file)
        best_epoch = loaded_checkpoint["epoch"]

        logging.info(f"Loading from epoch {best_epoch} (strict=False)")

        # model = model.load_from_checkpoint(best_checkpoint)
        self.load_state_dict(loaded_checkpoint["state_dict"], strict=False)

    def train_model(
        self,
        module,
        save_dir,
        max_epochs=None,
        gradient_clip_val=5,
        min_epochs=None,
        gpus=0,
        log_version=None,
        log_name="",
        patience: int = 20,
        tune: bool = False,
        tune_save: bool = False,
        debug: str = None,
        **kwargs,
    ) -> List[dict]:
        """train_model.

        Args:
            module:
            save_dir:
            max_epochs:
            gradient_clip_val:
            min_epochs:
            gpus:
            log_version:
            log_name:
            patience (int): patience
            tune (bool): False
            tune_save (bool): If true, save model
            kwargs:

        Returns:
            List[dict]:
        """

        tb_logger = pl_loggers.TensorBoardLogger(
            save_dir, name=log_name, version=log_version
        )

        tb_path = tb_logger.log_dir
        self.set_results_dir(tb_path)

        monitor = "val_loss"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            dirpath=tb_path,
            filename="best",
            save_weights_only=True,
        )
        callbacks = []
        loggers = [tb_logger]

        if not tune:
            callbacks.append(checkpoint_callback)
            console_logger = utils.ConsoleLogger()
            loggers.append(console_logger)
            val_check_interval = None
            check_val_every_n_epoch = 1
        else:
            tune_callback = utils.TuneReportCallback(["val_loss"], on="validation_end")
            callbacks.append(tune_callback)
            if tune_save:
                callbacks.append(checkpoint_callback)
            val_check_interval = None  # 2000 #2000
            check_val_every_n_epoch = 1

        # Set results dir
        earlystop_callback = EarlyStopping(monitor="val_loss", patience=patience)
        callbacks.append(earlystop_callback)

        # Add in LR Monitor
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gradient_clip_val=gradient_clip_val,
            min_epochs=min_epochs,
            accelerator="gpu" if gpus >= 1 else None,
            devices=gpus if gpus >= 1 else None,
            logger=loggers,
            reload_dataloaders_every_n_epochs=1,
            callbacks=callbacks,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            enable_checkpointing=False if (tune and not tune_save) else True,
        )
        trainer.fit(self, module)
        # if debug == "test":
        # else:
        #    trainer.fit(self, module)

        if tune:
            return None

        # Modify the model after fit with callbacks as inputs
        # This involves loading the model frorm the best checkpoint
        self.post_train_modify(_checkpoint_callback=checkpoint_callback, **kwargs)

        # Test
        # List of losses
        test_loss = trainer.test(self, module.test_dataloader())
        logging.info(test_loss)
        test_losses = test_loss

        # Turn on eval mode
        self.eval()
        return test_losses
