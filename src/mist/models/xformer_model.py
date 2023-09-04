""" Sinusoidal spectra model"""
from typing import Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from mist.data import featurizers
from mist.models.base import TorchModel, register_model
from mist.models import modules, transformer_layer
import mist.utils as utils


@register_model
class FingerIDXFormer(TorchModel):
    def __init__(
        self,
        fp_names: List[str] = ["morgan2048"],
        loss_fn: str = "bce",
        embed_instrument: bool = False,
        **kwargs,
    ):
        """__init__"""
        super().__init__(**kwargs)
        self.output_size = featurizers.FingerprintFeaturizer.get_fingerprint_size(
            fp_names
        )

        self.embed_instrument = embed_instrument
        self.instr_dim = utils.max_instr_idx
        self.instrument_embedder = nn.Parameter(torch.eye(self.instr_dim))
        self.inten_dim = 1

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
        **kwargs,
    ):
        """_summary_

        Args:
            hidden_size (int, optional): _description_. Defaults to 50.
            spectra_dropout (float, optional): _description_. Defaults to 0.0.
            num_spec_layers (int, optional): _description_. Defaults to 3.
        """
        self.hidden_size = hidden_size

        # Step 1 is to define a freq embedder
        self.embedder = FourierEmbedder(d=hidden_size)
        self.freq_mlp = nn.Sequential(
            nn.Linear(hidden_size + self.instr_dim + 1, hidden_size),
            nn.Dropout(spectra_dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )

        # Step 2 apply transformer on top of these embeddings
        # Multihead attention block with residuals
        peak_attn_layer = transformer_layer.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=2 * self.hidden_size,
            dropout=spectra_dropout,
            additive_attn=False,
            pairwise_featurization=False,
        )
        self.peak_attn_layers = modules._get_clones(peak_attn_layer, num_spec_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, self.output_size), nn.Sigmoid()
        )

    @staticmethod
    def spec_features(mode: Optional[str] = None) -> str:
        return "mz_xformer"

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
                # on_epoch=True,
                f"test_{k}",
                v,
                logger=True,
                batch_size=len(pred_fp),
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
                # on_epoch=True,
                f"val_{k}",
                v,
                batch_size=len(pred_fp),
                logger=True,
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
                # on_epoch=True,
                f"train_{k}",
                v,
                batch_size=len(pred_fp),
                logger=True,
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

        spec = batch["spectra"]
        num_peaks = batch["input_lens"]
        instruments = batch["instruments"]
        batch_dim, peak_dim, _ = spec.shape
        device = spec.device
        mz_vec = spec[:, :, 0]
        inten_vec = spec[:, :, 1]
        embedded_mz = self.embedder(mz_vec)

        embedded_instruments = self.instrument_embedder[instruments.long()]
        if self.embed_instrument:
            embedded_instruments = embedded_instruments[:, None, :].repeat(
                1, peak_dim, 1
            )
        else:
            embedded_instruments = torch.zeros(batch_dim, peak_dim, self.instr_dim).to(
                device
            )

        cat_vec = [embedded_mz, inten_vec[:, :, None], embedded_instruments]
        peak_tensor = torch.cat(cat_vec, -1)
        peak_tensor = self.freq_mlp(peak_tensor)

        peaks_aranged = torch.arange(peak_dim).to(device)

        # batch x num peaks
        attn_mask = ~(peaks_aranged[None, :] < num_peaks[:, None])

        # Transpose to peaks x batch x features
        peak_tensor = peak_tensor.transpose(0, 1)
        for peak_attn_layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = peak_attn_layer(
                peak_tensor,
                src_key_padding_mask=attn_mask,
            )

        peak_tensor = peak_tensor.transpose(0, 1)

        # Get only the class token
        h0 = peak_tensor[:, 0, :]

        encoded = self.output_layer(h0)
        aux_outputs = {"h0": h0}
        return encoded, aux_outputs

    def encode_mol(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """encode_mol.

        Identity encoder because we want to predict fingerprints

        """
        return batch["mols"], {}


class FourierEmbedder(torch.nn.Module):
    """Embed a set of mz float values using frequencies"""

    def __init__(self, d=512, logmin=-2.5, logmax=3.3, **kwargs):
        super().__init__()
        self.d = d
        self.logmin = logmin
        self.logmax = logmax

        lambda_min = np.power(10, -logmin)
        lambda_max = np.power(10, logmax)
        index = torch.arange(np.ceil(d / 2))
        exp = torch.pow(lambda_max / lambda_min, (2 * index) / (d - 2))
        freqs = 2 * np.pi * (lambda_min * exp) ** (-1)

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # Turn off requires grad for freqs
        self.freqs.requires_grad = False

    def forward(self, mz: torch.FloatTensor):
        """forward

        Args:
            mz: FloatTensor of shape (batch_size, mz values)

        Returns:
            FloatTensor of shape (batch_size, peak len, mz )
        """
        freq_input = torch.einsum("bi,j->bij", mz, self.freqs)
        embedded = torch.cat([torch.sin(freq_input), torch.cos(freq_input)], -1)
        embedded = embedded[:, :, : self.d]
        return embedded
