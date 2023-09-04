""" contrastive_model.py """
import copy
from typing import Optional, List, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from mist.models.base import TorchModel, register_model, build_model
from mist import utils


@register_model
class ContrastiveModel(TorchModel):
    """ContrastiveModel."""

    def __init__(
        self,
        base_model_hparams: dict = {},
        fp_names: List[str] = ["morgan2048"],
        dist_name: str = "bce",
        contrastive_weight: float = 0.0,
        contrastive_scale: float = 1.0,
        contrastive_bias: float = 0.0,
        contrastive_loss: str = "none",
        contrastive_latent: str = "fp",
        contrastive_decoy_pool: str = "mean",
        contrastive_latent_size: int = 256,
        contrastive_latent_dropout: float = 0.0,
        **kwargs,
    ):
        """__init__"""

        super().__init__(**kwargs)
        self.save_hyperparameters()

        # Build base model
        self.main_model = build_model(**base_model_hparams)

        self.contrastive_weight = contrastive_weight
        self.contrastive_scale = contrastive_scale
        self.contrastive_bias = contrastive_bias
        self.contrastive_loss = contrastive_loss
        self.contrastive_latent = contrastive_latent
        self.contrastive_decoy_pool = contrastive_decoy_pool
        self.contrastive_latent_size = contrastive_latent_size
        self.contrastive_latent_dropout = contrastive_latent_dropout
        self.output_size = self.main_model.output_size
        self.hidden_size = self.main_model.hidden_size

        # Add in contrastive module
        if self.contrastive_latent == "fp":
            self.mol_aux_encoder = nn.Identity()
        elif self.contrastive_latent == "h0_learned":
            self.mol_aux_encoder = nn.Identity()
        elif self.contrastive_latent == "h0":
            self.mol_aux_encoder = nn.Linear(self.output_size, self.hidden_size)
        elif self.contrastive_latent == "aux":
            self.mol_aux_encoder = nn.Linear(
                self.output_size, self.contrastive_latent_size
            )
            self.spec_aux_encoder = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(self.contrastive_latent_dropout),
                nn.Linear(self.hidden_size, self.contrastive_latent_size),
            )
        elif self.contrastive_latent == "fp_aux":
            self.mol_aux_encoder = nn.Linear(
                self.output_size, self.contrastive_latent_size
            )
            self.spec_aux_encoder = copy.deepcopy(self.mol_aux_encoder)
        elif self.contrastive_latent == "fp_aux_siamsese":
            self.mol_aux_encoder = nn.Sequential(
                nn.Linear(self.output_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.contrastive_latent_dropout),
                nn.Linear(self.hidden_size, self.contrastive_latent_size),
            )
            self.spec_aux_encoder = self.mol_aux_encoder
        else:
            raise ValueError()

        ## BCE loss
        self.bce_loss = nn.BCELoss(reduction="none")
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.cosine_loss = lambda x, y: 1 - cosine_sim(
            x.expand(y.shape), y.float()
        ).unsqueeze(-1)
        # Try bce loss for distance
        self.dist_name = dist_name
        if self.dist_name == "bce":
            self.dist_fn = lambda x, y: self.bce_loss(
                x.expand(y.shape), y.float()
            ).mean(-1)
        elif self.dist_name == "cosine":
            self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
            self.dist_fn = lambda x, y: 1 - self.cosine_sim(
                x.expand(y.shape), y.float()
            )
        elif self.dist_name == "euclid":
            self.dist_fn = lambda x, y: ((x.expand(y.shape) - y.float()) ** 2).sum(
                -1
            ) ** (0.5)
        elif self.dist_name == "tanimoto":
            self.dist_fn = self._tanimoto_dist
        else:
            raise NotImplementedError()
        # self.spectra_encoder = self._build_model(**kwargs)

    def _tanimoto_dist(self, x, y):
        """compute tanimoto dist"""
        x_bool = x > self.thresh
        y_bool = y.type(torch.bool)
        intersection = (y_bool & x_bool).sum(1)
        union = (y_bool | x_bool).sum(1)
        tani = intersection / union
        return 1 - tani

    def forward(self, batch):
        """forward pass"""
        raise NotImplementedError()

    def encode_spectra(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """encode_spectra."""
        fp_pred, aux = self.main_model.encode_spectra(batch)

        # Transform into contrastive
        if self.contrastive_latent == "fp":
            aux["contrast"] = fp_pred
        elif self.contrastive_latent == "h0":
            aux["contrast"] = aux["h0"]
        elif self.contrastive_latent == "h0_learned":
            aux["contrast"] = aux["h0"]
        elif self.contrastive_latent == "aux":
            aux["contrast"] = self.spec_aux_encoder(aux["h0"])
        elif self.contrastive_latent == "fp_aux":
            aux["contrast"] = self.spec_aux_encoder(fp_pred)
        elif self.contrastive_latent == "fp_aux":
            aux["contrast"] = self.spec_aux_encoder(fp_pred)
        else:
            raise ValueError()

        return fp_pred, aux

    def encode_mol(self, batch: dict) -> Tuple[torch.Tensor, dict]:
        """encode_mol.

        Identity encoder because we want to predict fingerprints
        """
        fp, aux = batch["mols"][:, :], {}
        aux["contrast"] = self.mol_aux_encoder(fp.float())

        return fp, aux

    def compute_contrastive_loss(self, pred_contrast, targ_contrast, batch: dict):
        """Compute contrastive loss"""

        if self.contrastive_loss == "none":
            return 0

        elif self.contrastive_loss == "triplet_rand":

            mol_indices = batch["mol_indices"]
            spec_indices = batch["spec_indices"]

            paired = batch["matched"]
            paired_mol_inds = mol_indices[paired.bool()]
            paired_spec_inds = spec_indices[paired.bool()]

            paired_pred_fps = pred_contrast[paired_spec_inds]
            paired_mol_fps = targ_contrast[paired_mol_inds]

            paired_dist = self.dist_fn(paired_pred_fps, paired_mol_fps)

            # Pred fps

            # All decoys
            unpaired_mol_fps = targ_contrast[spec_indices[~paired.bool()]]

            # Now need to do a weird shuffle
            expanded_unpaired_mol_fps = unpaired_mol_fps[None, :, :]
            expanded_unpaired_mol_fps = expanded_unpaired_mol_fps.repeat(
                paired_pred_fps.shape[0], 1, 1
            )
            expanded_preds = paired_pred_fps[:, None, :]
            expanded_preds = expanded_preds.repeat(1, unpaired_mol_fps.shape[0], 1)

            negative_dist = self.dist_fn(expanded_preds, expanded_unpaired_mol_fps)
            dist_diff = paired_dist[:, None] - negative_dist + self.contrastive_bias

            loss = torch.relu(dist_diff)

            # Mean pool
            if self.contrastive_decoy_pool == "mean":
                batch_loss = loss.mean(-1)
            elif self.contrastive_decoy_pool == "max":
                batch_loss = loss.max(-1)[0]
            elif self.contrastive_decoy_pool == "logsumexp":
                # Compute logsumexp and get a tighter bound with shape
                shape = loss.shape[-1]
                batch_loss = torch.logsumexp(loss * shape, -1) / shape
            else:
                raise NotImplementedError()
            return batch_loss

        elif self.contrastive_loss == "triplet":
            mol_indices = batch["mol_indices"]
            spec_indices = batch["spec_indices"]

            pred_fps = pred_contrast[spec_indices]
            mol_fps = targ_contrast[mol_indices]

            # Compute distance
            # BCE loss is a dist
            distances = self.dist_fn(pred_fps, mol_fps)
            paired = batch["matched"]

            paired_dist = distances[paired.bool()]
            unpaired_dist = distances[~paired.bool()]

            # Count how many decoys for each exapmle
            repeat_interleave_ct = torch.zeros_like(paired_dist)
            one_vec = torch.ones_like(spec_indices).float()
            repeat_interleave_ct = repeat_interleave_ct.index_add(
                0, spec_indices, one_vec
            )

            # Remove 1 to adjust for the paired examples
            repeat_interleave_ct -= 1
            paired_dist_expanded = paired_dist.repeat_interleave(
                repeat_interleave_ct.long()
            )

            # We want to minimize this, which means maximizing unpaired dist
            # Scale factor makes the contribution of far negatives small
            dist_diff = paired_dist_expanded - unpaired_dist + self.contrastive_bias
            loss = torch.relu(dist_diff)

            scatter_inds = spec_indices[~paired.bool()]

            # Expand and pad (slower, more flexible)
            ind_argsort = torch.argsort(scatter_inds)
            loss_argsorted = loss[ind_argsort]

            # Batch x distances
            padded_dists = utils.pad_packed_tensor(
                loss_argsorted, repeat_interleave_ct.long(), 0
            )
            padded_dists = padded_dists

            # Mean pool
            if self.contrastive_decoy_pool == "mean":
                batch_loss = padded_dists.sum(-1) / (repeat_interleave_ct + 1e-12)
                batch_loss[repeat_interleave_ct == 0] = 0
            elif self.contrastive_decoy_pool == "max":
                batch_loss = padded_dists.max(-1)[0]
            elif self.contrastive_decoy_pool == "logsumexp":
                # Compute logsumexp
                batch_loss = torch.logsumexp(padded_dists, -1)
            else:
                raise NotImplementedError()

            # Use scatter operations (faster, less flexible in this torch vzn)
            # batch_loss = torch.zeros_like(paired_dist)
            # batch_loss = batch_loss.index_add(0, scatter_inds, loss)

            # batch_loss = batch_loss / (repeat_interleave_ct + 1e-12)
            # batch_loss[repeat_interleave_ct == 0] = 0
            return batch_loss

        elif self.contrastive_loss == "softmax":
            mol_indices = batch["mol_indices"]
            spec_indices = batch["spec_indices"]

            pred_fps = pred_contrast[spec_indices]
            mol_fps = targ_contrast[mol_indices]

            # Compute distance
            # BCE loss is a dist
            distances = self.dist_fn(pred_fps, mol_fps)
            paired = batch["matched"]

            paired_dist = distances[paired.bool()]
            unpaired_dist = distances[~paired.bool()]

            # Count how many decoys for each exapmle
            repeat_interleave_ct = torch.zeros_like(paired_dist)
            one_vec = torch.ones_like(spec_indices).float()
            repeat_interleave_ct = repeat_interleave_ct.index_add(
                0, spec_indices, one_vec
            )

            # Remove 1 to adjust for the paired examples
            repeat_interleave_ct -= 1
            paired_dist_expanded = paired_dist.repeat_interleave(
                repeat_interleave_ct.long()
            )

            # We want to minimize this, which means maximizing unpaired dist
            # Scale factor makes the contribution of far negatives small
            exp_term = paired_dist_expanded - unpaired_dist
            exp_term = self.contrastive_scale * (exp_term + self.contrastive_bias)
            exp_factors = torch.exp(exp_term)

            # Compute summation
            # Dsetination
            exp_sums = torch.zeros(len(paired_dist)).to(paired_dist.device)

            # Take all the paired examples of positive - negative and sum them in
            # place
            scatter_index = spec_indices[~paired.bool()]
            exp_sums.scatter_add_(0, scatter_index, exp_factors)

            tuplet_loss = torch.log(1 + exp_sums) / self.contrastive_scale
            tuplet_loss = tuplet_loss.mean()
            return tuplet_loss.mean(-1)

        elif self.contrastive_loss == "nce":
            mol_indices = batch["mol_indices"]
            spec_indices = batch["spec_indices"]

            pred_fps = pred_contrast[spec_indices]
            mol_fps = targ_contrast[mol_indices]

            # Compute distance
            # BCE loss is a dist
            # Note for NCE, this must be a probability
            distances = self.dist_fn(pred_fps, mol_fps)
            paired = batch["matched"]

            # Create probabilities
            distances = torch.exp(-distances * self.contrastive_scale)

            # it's already negative
            paired_probs = distances[paired.bool()]
            unpaired_probs = distances[~paired.bool()]

            # Count how many decoys for each exapmle
            repeat_interleave_ct = torch.zeros_like(paired_probs)
            one_vec = torch.ones_like(spec_indices).float()
            repeat_interleave_ct = repeat_interleave_ct.index_add(
                0, spec_indices, one_vec
            )

            # Remove 1 to adjust for the paired examples
            # repeat_interleave_ct -= 1
            # paired_dist_expanded = paired_probs.repeat_interleave(
            #    repeat_interleave_ct.long()
            # )

            # Compute summation
            # Dsetination
            denoms = torch.zeros(len(paired_probs)).to(paired_probs.device)

            # Take all the paired examples of positive - negative and sum them in
            # place
            scatter_index = spec_indices[~paired.bool()]
            denoms.scatter_add_(0, scatter_index, unpaired_probs)
            denoms = denoms + paired_probs
            nums = paired_probs

            # add
            # denoms[denoms == 0] = 1e-8

            # Add epsilon
            # Note: This is mostly equiv to logsumexp for denom beacuse it
            # started as log probability, we exponentiated, summed, then
            # logged. However, we had to add the exponnentiated numerator, so
            # can't use the logsumexp fn
            loss = torch.log(nums) - torch.log(denoms)
            return -loss.mean(-1)

        elif self.contrastive_loss == "clip":
            mol_indices = batch["mol_indices"]
            spec_indices = batch["spec_indices"]

            pred_fps = pred_contrast[spec_indices]
            mol_fps = targ_contrast[mol_indices]

            # Compute distance
            # BCE loss is a dist
            batch_size = pred_fps.shape[0]
            # feat_dim = pred_fps.shape[-1]
            pred_expand = pred_fps[:, None, :].repeat(1, batch_size, 1)
            targ_expand = mol_fps[None, :, :].repeat(batch_size, 1, 1)

            sims = -self.dist_fn(pred_expand, targ_expand)

            # old_shape = pred_expand.shape
            # reshaped_preds = pred_expand.reshape(-1, feat_dim)
            # reshaped_targs = targ_expand.reshape(-1, feat_dim)
            # sims = - self.dist_fn(reshaped_preds, reshaped_targs)
            # sims = sims.reshape(*old_shape[:-1])

            true_labels = torch.arange(batch_size, device=sims.device)

            # Is the closest FP to each prediction its true target?
            spec_loss = F.cross_entropy(sims, true_labels)

            # Is the closest prediction to each FP the true source?
            mol_loss = F.cross_entropy(sims.T, true_labels)

            total_loss = (spec_loss + mol_loss) / 2
            return total_loss
        else:
            raise NotImplementedError()

    def _get_loss_objs(self, batch, batch_idx, train=False):
        pred_fp, aux_outputs = self.encode_spectra(batch)

        # Mol fp's
        target_fp_all, aux_outputs_mol = self.encode_mol(batch)
        target_fp_all = target_fp_all.float()

        # Get the proper matched indices
        mol_inds = batch["mol_indices"]
        norm_inds = mol_inds[batch["matched"]]
        target_fp = target_fp_all[norm_inds]
        loss_dict_main = self.main_model.compute_loss(
            pred_fp,
            target_fp,
            aux_outputs_mol=aux_outputs_mol,
            aux_outputs_spec=aux_outputs,
            fingerprints=batch.get("fingerprints"),
            fingerprint_mask=batch.get("fingerprint_mask"),
            train_step=train,
        )
        total_loss = loss_dict_main["loss"]

        # Contrastive loss
        contrastive_loss = self.compute_contrastive_loss(
            aux_outputs["contrast"], aux_outputs_mol["contrast"], batch
        )
        loss_dict_main["contrastive_loss"] = contrastive_loss
        loss_dict_main["total_loss"] = total_loss
        loss_dict_main["batch_size"] = len(pred_fp)
        return loss_dict_main

    def training_step(self, batch, batch_idx):
        """training_step.

        This is called by lightning trainer.

        Returns loss obj

        """
        loss_dict_main = self._get_loss_objs(batch, batch_idx, train=True)
        contrastive_loss = loss_dict_main.get("contrastive_loss")
        total_loss = loss_dict_main.get("total_loss")
        batch_size = loss_dict_main.get("batch_size")

        self.log(
            "train_contrastive_loss_unweight",
            contrastive_loss.mean().item(),
            # on_epoch=True,
            batch_size=batch_size,
            prog_bar=True,
            logger=True,
        )
        total_loss = (
            total_loss * (1 - self.contrastive_weight)
            + contrastive_loss * self.contrastive_weight
        )
        total_loss = total_loss.sum()
        self.log(
            "train_loss",
            total_loss.mean(),
            # on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            logger=True,
        )
        loss_dict_main["loss"] = total_loss
        return loss_dict_main

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss_dict_main = self._get_loss_objs(batch, batch_idx, train=False)
        contrastive_loss = loss_dict_main.get("contrastive_loss")
        total_loss = loss_dict_main.get("total_loss")
        batch_size = loss_dict_main.get("batch_size")

        self.log(
            "val_contrastive_loss_unweight",
            contrastive_loss.mean().item(),
            batch_size=batch_size,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        total_loss = (
            total_loss * (1 - self.contrastive_weight)
            + contrastive_loss * self.contrastive_weight
        )
        total_loss = total_loss.sum()
        self.log(
            "val_loss_total",
            total_loss.mean(),
            batch_size=batch_size,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_loss",
            contrastive_loss.mean().item(),
            batch_size=batch_size,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        loss_dict_main["loss"] = contrastive_loss.mean()
        return loss_dict_main

    def test_step(self, batch, batch_idx):
        """Test step"""
        loss_dict_main = self._get_loss_objs(batch, batch_idx, train=False)
        contrastive_loss = loss_dict_main.get("contrastive_loss")
        total_loss = loss_dict_main.get("total_loss")
        batch_size = loss_dict_main.get("batch_size")

        self.log(
            "test_contrastive_loss_unweight",
            contrastive_loss.mean().item(),
            batch_size=batch_size,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        total_loss = (
            total_loss * (1 - self.contrastive_weight)
            + contrastive_loss * self.contrastive_weight
        )
        total_loss = total_loss.sum()
        self.log(
            "test_loss_total",
            total_loss.mean(),
            batch_size=batch_size,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_loss",
            contrastive_loss.mean().item(),
            batch_size=batch_size,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        loss_dict_main["loss"] = contrastive_loss.mean()
        return loss_dict_main

    @staticmethod
    def mol_features(mode: Optional[str] = None) -> str:
        return "fingerprint"

    @staticmethod
    def dataset_type(mode: Optional[str] = None) -> str:
        """dataset_type."""
        return "contrastive_hdf"
