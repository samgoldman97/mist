""" binned_ffn_model.py"""
from typing import Optional, List, Tuple
import torch
import torch.nn as nn

from mist.data import featurizers
from mist.models.base import TorchModel, register_model
from mist.models import modules


@register_model
class FingerIDFFN(TorchModel):
    def __init__(
        self, fp_names: List[str] = ["morgan2048"], loss_fn: str = "bce", **kwargs
    ):
        """__init__"""
        super().__init__(**kwargs)
        self.output_size = featurizers.FingerprintFeaturizer.get_fingerprint_size(
            fp_names
        )

        # BCE loss
        self.bce_loss = nn.BCELoss(reduction="none")
        self.loss_name = loss_fn

        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.cosine_loss = lambda x, y: 1 - cosine_sim(
            x.expand(y.shape), y.float()
        ).unsqueeze(-1)

        if self.loss_name == "bce":
            self.loss_fn = self.bce_loss
        elif self.loss_name == "mse":
            mse_loss = nn.MSELoss(reduction="none")
            self.loss_fn = mse_loss
        elif self.loss_name == "cosine":
            self.loss_fn = self.cosine_loss
        else:
            raise NotImplementedError()

        self._build_model(**kwargs)
        self.save_hyperparameters()

    def _build_model(
        self,
        hidden_size: int = 50,
        spectra_dropout: float = 0.0,
        num_spec_layers: int = 3,
        num_bins: int = 100000,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.spectra_encoder_main = modules.MLPBlocks(
            input_size=num_bins,
            hidden_size=hidden_size,
            dropout=spectra_dropout,
            num_layers=num_spec_layers,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, self.output_size), nn.Sigmoid()
        )

    @staticmethod
    def spec_features(mode: Optional[str] = None) -> str:
        return "binned"

    @staticmethod
    def mol_features(mode: Optional[str] = None) -> str:
        return "fingerprint"

    def test_step(self, batch, batch_idx):
        """Test step"""
        pred_fp, aux_outputs = self.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = self.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]
        loss_dict = self.compute_loss(pred_fp, target_fp, train_step=False)

        for k, v in loss_dict.items():
            self.log(
                f"test_{k}", v, logger=True, batch_size=len(pred_fp)  # on_epoch=True,
            )

        return loss_dict

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        pred_fp, aux_outputs = self.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = self.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]
        loss_dict = self.compute_loss(pred_fp, target_fp, train_step=False)

        for k, v in loss_dict.items():
            self.log(
                f"val_{k}", v, batch_size=len(pred_fp), logger=True  # on_epoch=True,
            )

        return loss_dict

    def training_step(self, batch, batch_idx):
        """training_step.

        This is called by lightning trainer.

        Returns loss obj
        """
        # Sum pool over channels for simplicity
        pred_fp, aux_outputs_spec = self.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = self.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]

        loss_dict = self.compute_loss(pred_fp, target_fp, train_step=True)

        for k, v in loss_dict.items():
            self.log(
                f"train_{k}", v, batch_size=len(pred_fp), logger=True  # on_epoch=True,
            )
        return loss_dict

    def compute_loss(self, pred_fp, target_fp, train_step=True, **kwargs):

        # Compute weight of loss function
        fp_loss, iterative_loss = None, None

        # Get FP Loss
        fp_loss = self.loss_fn(pred_fp, target_fp).mean(-1)

        # Pull losses together
        total_loss = fp_loss
        loss_weights = torch.ones_like(total_loss)
        loss_weights = loss_weights / loss_weights.sum()
        if not train_step and iterative_loss is not None:
            total_loss += iterative_loss

        # Weighted mean over batch
        total_loss = (total_loss * loss_weights).sum()

        ret_dict = {
            "loss": total_loss,
            "mol_loss": fp_loss.mean().item(),
        }

        return ret_dict

    def encode_spectra(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        h0 = self.spectra_encoder_main(batch["spectra"].float())
        encoded = self.output_layer(h0)
        aux_outputs = {"h0": h0}

        return encoded, aux_outputs

    def encode_mol(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """encode_mol.

        Identity encoder because we want to predict fingerprints

        """
        return batch["mols"], {}
