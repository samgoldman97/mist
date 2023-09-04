"""ffn_model.py"""
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ForwardFFN(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.0,
        learning_rate: float = 7e-4,
        min_lr: float = 1e-6,
        lr_decay_rate: float = 0.9,
        input_dim: int = 4096,
        output_dim: int = 1000,
        upper_limit: int = 1000,
        use_reverse: bool = True,
        scheduler: bool = False,
        weight_preds: bool = False,
        loss_name: str = "cosine",
        fp_type: str = "csi",
        growing: str = "none",
        growing_weight: str = 0.5,
        growing_layers: str = 3,
        growing_scheme: str = "interleave",
        **kwargs,
    ):
        """__init__.
        Args:
            hidden_size (int): Hidden size
            layers (int): Num layers
            dropout (float): Amount of dropout
            learning_rate (float): Learning rate
            min_lr (float): Min lr
            input_dim (int): Input dim of FP
            output_dim (int): Output dim of FP
            upper_limit (int): Max bin size
            use_reverse (bool): If true, use the reverse scheme
            weight_preds (bool): If true, add weighting on rpeds
            loss_name (str): Name of loss fn
            fp_type (str): Type of fingerprint
            growing (str): Option to grow fingerprint
            growing_weight (float): Growth weight
            growing_layers (int): Num of layers for growing
            growing_scheme (str): "interleave"
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_size = hidden_size

        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.upper_limit = upper_limit
        self.use_reverse = use_reverse
        self.weight_preds = weight_preds
        self.fp_type = fp_type

        self.growing = growing
        self.growing_weight = growing_weight
        self.growing_layers = growing_layers

        # Get bin masses
        self.bin_masses = torch.from_numpy(np.linspace(0, upper_limit, output_dim))
        self.bin_masses = nn.Parameter(self.bin_masses)
        self.bin_masses.requires_grad = False

        self.dropout = dropout

        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.lr_decay_rate = lr_decay_rate

        # Define network
        self.activation = nn.ReLU()

        # Use identity
        self.output_activation = nn.Sigmoid()
        self.init_layer = nn.Linear(self.input_dim, self.hidden_size)
        self.dropout_layer = nn.Dropout(p=self.dropout)

        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = get_clones(middle_layer, self.layers - 1)

        if self.use_reverse:
            # Gates, reverse, forward
            # Define growing modules
            if self.growing == "iterative":
                self.output_layer = GrowingModule(
                    hidden_input_dim=self.hidden_size,
                    final_target_dim=self.output_dim * 3,
                    num_splits=self.growing_layers,
                    scheme=growing_scheme,
                    chunks=3,
                )
            else:
                self.output_layer = nn.Linear(self.hidden_size, self.output_dim * 3)
        else:
            if self.growing == "iterative":
                self.output_layer = GrowingModule(
                    hidden_input_dim=self.hidden_size,
                    final_target_dim=self.output_dim,
                    num_splits=self.growing_layers,
                    scheme=growing_scheme,
                    chunks=1,
                )
            else:
                self.output_layer = nn.Linear(self.hidden_size, self.output_dim)

        if loss_name == "cosine":
            self.loss_fn = self.cosine_loss
        elif loss_name == "bce":
            self.loss_fn = self.bce_loss
        else:
            raise NotImplementedError()
        # self.loss_fn = self.bce_loss

    def cosine_loss(self, pred, targ, weight_preds=False, **kwargs):
        """Loss fn

        Args:
            pred (torch.tensor): Predictions
            targ (torch.tensor): Targets
        """
        if weight_preds:
            weighted_pred = (pred + 1e-22**0.5) * self.bin_masses
            weighted_targ = (targ + 1e-22**0.5) * self.bin_masses
        else:
            weighted_pred = pred
            weighted_targ = targ
        cos_dist = 1 - F.cosine_similarity(weighted_pred, weighted_targ, dim=-1)
        cos_dist_mean = cos_dist.mean()
        return {"loss": cos_dist_mean}

    def bce_loss(self, pred, targ, scalar_mult=4, **kwargs):
        """Loss fn

        Args:
            pred (torch.tensor): Predictions
            targ (torch.tensor): Targets
        """
        binary_targ = targ > 0
        # Weight by intensities but take softmax so zero weights still get some
        # vals
        # Mult by scalar
        weights = torch.softmax(targ * scalar_mult, -1)  # 4 is best so far
        bce_loss = F.binary_cross_entropy(
            pred, binary_targ.float(), weight=weights, reduction="none"
        )
        bce_loss = bce_loss.sum(-1)
        bce_loss_mean = bce_loss.mean()
        return {"loss": bce_loss_mean}

    def forward(self, fps, full_weight=None, training=False):
        """predict spec"""
        fps = fps.float()
        output = self.init_layer(fps)
        output = self.activation(output)
        output = self.dropout_layer(output)

        # Convert full weight into bin index
        # Find first index at which it's true
        full_mass_bin = (full_weight[:, None] < self.bin_masses).int().argmax(-1)
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)

        output = self.output_layer(output)
        if self.growing == "none":

            # Get indices where it makes sense to predict a mass
            full_arange = torch.arange(self.output_dim, device=output.device)
            is_valid = full_arange[None, :] <= full_mass_bin[:, None]

            if self.use_reverse:
                forward_preds, rev_preds, gates = torch.chunk(output, 3, -1)

                # Rejigger reverse preds
                # Set forward preds to 0

                # Get new rowe inds to shift
                new_inds = full_mass_bin[:, None] - full_arange[None, :]

                # Fix neg indices to make them postive
                neg_inds = new_inds[new_inds < 0] + self.output_dim
                new_inds[new_inds < 0] = neg_inds

                # Gather reverse and gates
                rev_preds = torch.gather(rev_preds, dim=-1, index=new_inds)
                dir_gate = torch.sigmoid(gates)
                output = forward_preds * dir_gate + rev_preds * (1 - dir_gate)

            # Mask everything
            output = self.output_activation(output)
            output = output * is_valid.float()
            return output
        elif self.growing == "iterative":
            if not training:
                output = output[-1]
                # Get indices where it makes sense to predict a mass
                full_arange = torch.arange(self.output_dim, device=output.device)
                is_valid = full_arange[None, :] <= full_mass_bin[:, None]
                if self.use_reverse:
                    forward_preds, rev_preds, gates = torch.chunk(output, 3, -1)

                    # Rejigger reverse preds
                    # Set forward preds to 0
                    # Get new rowe inds to shift
                    new_inds = full_mass_bin[:, None] - full_arange[None, :]

                    # Fix neg indices to make them postive
                    neg_inds = new_inds[new_inds < 0] + self.output_dim
                    new_inds[new_inds < 0] = neg_inds

                    # Gather reverse and gates
                    rev_preds = torch.gather(rev_preds, dim=-1, index=new_inds)

                    dir_gate = gates
                    output = forward_preds * dir_gate + rev_preds * (1 - dir_gate)

                # Mask everything
                output = output * is_valid.float()
                return output
            else:
                # For each reverse prediction, rearrange and reshuffle

                outputs = []
                for output_step in output:

                    # Get indices where it makes sense to predict a mass
                    temp_output_shape = output_step.shape[-1]
                    if self.use_reverse:
                        temp_output_shape = output_step.shape[-1] // 3

                    div_factor = int(np.ceil(self.output_dim / temp_output_shape))
                    full_arange = torch.arange(
                        temp_output_shape, device=output_step.device
                    )
                    full_mass_bin_temp = torch.div(
                        full_mass_bin, div_factor, rounding_mode="trunc"
                    )
                    if self.use_reverse:
                        forward_preds, rev_preds, gates = torch.chunk(
                            output_step, 3, -1
                        )

                        # Rejigger reverse preds
                        # Set forward preds to 0
                        # Get new row inds to shift
                        new_inds = full_mass_bin_temp[:, None] - full_arange[None, :]

                        # Fix neg indices to make them postive
                        neg_inds = new_inds[new_inds < 0] + temp_output_shape
                        new_inds[new_inds < 0] = neg_inds

                        # Gather reverse and gates
                        rev_preds = torch.gather(rev_preds, dim=-1, index=new_inds)
                        dir_gate = gates
                        output_step = forward_preds * dir_gate + rev_preds * (
                            1 - dir_gate
                        )

                    # Mask for division
                    is_valid = full_arange[None, :] <= full_mass_bin_temp[:, None]

                    # No mask for train
                    # output_step = output_step * is_valid.float()
                    outputs.append(output_step)
                return outputs

    def training_step(self, batch, batch_idx):
        """training_step.

        Args:
            batch:
            batch_idx:
        """
        # Modified
        pred_spec = self.forward(batch["fps"], batch["full_weight"], training=True)
        if self.growing == "iterative":

            # Compute iterative loss
            int_preds = pred_spec[::-1]
            orig_targ, cur_targ = batch["spectra"], batch["spectra"]
            aux_loss, ret_dict = None, {}
            for int_pred in int_preds[:]:
                targ_shape = int_pred.shape[-1]

                # Find bits where
                batch_ind, oldbit_ind = torch.where(cur_targ)

                # Try new truncation method for pooling??
                # Divide by the reduction factor to pool similar to the convs
                div_factor = np.ceil(cur_targ.shape[-1] / targ_shape)
                bit_ind = torch.div(oldbit_ind, div_factor, rounding_mode="trunc")

                # Develop new target
                new_targ = torch.zeros_like(int_pred).float()
                new_ind = torch.zeros_like(cur_targ).long()
                new_ind[batch_ind, oldbit_ind] = bit_ind.long()
                new_targ.scatter_add_(dim=-1, index=new_ind, src=cur_targ.float())
                # Compute loss
                temp_loss = self.loss_fn(int_pred, new_targ)["loss"].mean(-1)
                ret_dict[f"loss_on_{targ_shape}"] = temp_loss.mean().item()
                if aux_loss is None:
                    aux_loss = temp_loss
                else:
                    aux_loss += temp_loss
                cur_targ = new_targ

            full_loss = ret_dict[f"loss_on_{orig_targ.shape[-1]}"]
            aux_loss -= full_loss
            loss_dict = ret_dict
            loss_dict["full_vec_loss"] = full_loss
            loss_dict["loss"] = (
                full_loss * (1 - self.growing_weight) + aux_loss * self.growing_weight
            )
        else:
            loss_dict = self.loss_fn(pred_spec, batch["spectra"], weight_preds=False)

        for k, v in loss_dict.items():
            self.log(
                f"train_{k}",
                v,
                batch_size=len(pred_spec),
                # on_epoch=True,
                logger=True,
            )
        return loss_dict

    def validation_step(self, batch, batch_idx):
        pred_spec = self.forward(batch["fps"], batch["full_weight"])
        loss_dict = self.loss_fn(pred_spec, batch["spectra"], weight_preds=False)
        self.log(
            "val_loss",
            loss_dict.get("loss"),
            batch_size=len(pred_spec),
        )
        return loss_dict

    def test_step(self, batch, batch_idx):
        pred_spec = self.forward(batch["fps"], batch["full_weight"])
        loss_dict = self.loss_fn(pred_spec, batch["spectra"])

        pred_argsort_inds = torch.argsort(pred_spec, 1, descending=True)
        targ_argsort_inds = torch.argsort(batch["spectra"], 1, descending=True)

        num_peaks = (batch["spectra"] > 0).sum(-1)
        k_list = [1, 5, 10, 20, 50]
        k_ars = [[] for i in k_list]
        for batch_ind in np.arange(pred_spec.shape[0]):
            for k_ind, k in enumerate(k_list):
                pred_k_inds = pred_argsort_inds[batch_ind][:k].cpu().detach().numpy()
                targ_k_inds = targ_argsort_inds[batch_ind][:k].cpu().detach().numpy()
                max_div = num_peaks[batch_ind].cpu().item()
                targ_k_inds = targ_k_inds[:max_div]
                overlap = set(pred_k_inds).intersection(set(targ_k_inds))
                num_overlap = len(overlap) / min(k, max_div)
                k_ars[k_ind].append(num_overlap)

        for k, k_val in zip(k_list, k_ars):
            self.log(
                f"test_top_{k}_peak_overlap",
                np.mean(k_val),
                batch_size=len(pred_spec),
            )

        self.log("test_loss", loss_dict.get("loss"))
        return loss_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=0.0
        )
        decay_rate = self.lr_decay_rate
        start_lr = self.learning_rate
        steps_to_decay = 10000  # 1000 steps
        min_decay_rate = self.min_lr / start_lr
        lr_lambda = lambda epoch: (
            np.maximum(decay_rate ** (epoch // steps_to_decay), min_decay_rate)
        )
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambda
            )
            ret = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "frequency": 1,
                    "interval": "step",
                },
            }
        else:
            ret = {
                "optimizer": optimizer,
            }
        return ret


class GrowingModule(nn.Module):
    """GrowingModule.

    Accept an input hidden dim and progressively grow by powers of 2 s.t.

    We eventually get to the final output size...

    """

    def __init__(
        self,
        hidden_input_dim: int = 256,
        final_target_dim: int = 4096,
        num_splits=4,
        reduce_factor=2,
        chunks=3,
        scheme="ffn_grow",
    ):
        super().__init__()
        self.hidden_input_dim = hidden_input_dim
        self.final_target_dim = final_target_dim
        self.num_splits = num_splits
        self.reduce_factor = reduce_factor
        self.scheme = scheme

        final_output_size = self.final_target_dim

        # Creates an array where we end with final_size and have num_splits + 1
        # different entries in it (e.g., num_splits = 1 with final dim 4096 has
        # [2048, 4096])
        layer_dims = [
            final_output_size // (reduce_factor ** (num_split))
            for num_split in range(num_splits + 1)
        ][::-1]

        # Upscale to enforce strict divisibility of ints by chunks
        layer_dims = [int(np.ceil(i / chunks) * chunks) for i in layer_dims]
        layer_dims[-1] = final_output_size

        # Start by predicting into the very first layer dim (e.g., 256  -> 256)
        self.output_dims = layer_dims

        # Define initial predict module
        self.initial_predict = nn.Sequential(
            nn.Linear(
                hidden_input_dim,
                layer_dims[0],
            )
        )

        predict_bricks = []
        gate_bricks = []
        for layer_dim_ind, layer_dim in enumerate(layer_dims[:-1]):

            out_dim = layer_dims[layer_dim_ind + 1]

            if scheme == "ffn_grow":
                lin_predict = nn.Linear(layer_dim, out_dim)
                predict_brick = nn.Sequential(lin_predict, nn.Sigmoid())
                gate_bricks.append(
                    nn.Sequential(nn.Linear(hidden_input_dim, out_dim), nn.Sigmoid())
                )
                predict_bricks.append(predict_brick)
            elif scheme == "interleave":
                predict_brick = nn.Identity()
                gate_bricks.append(
                    nn.Sequential(
                        nn.Linear(hidden_input_dim, out_dim * 2), nn.Sigmoid()
                    )
                )
                predict_bricks.append(predict_brick)

        self.predict_bricks = nn.ModuleList(predict_bricks)
        self.gate_bricks = nn.ModuleList(gate_bricks)

    def forward(self, hidden):
        """forward.

        Return dict mapping output dim to the

        """

        cur_hidden = self.initial_predict(hidden)
        cur_pred = torch.sigmoid(cur_hidden)
        output_preds = [cur_pred]
        for ind, (_out_dim, predict_brick, gate_brick) in enumerate(
            zip(self.output_dims[1:], self.predict_bricks, self.gate_bricks)
        ):
            if self.scheme == "ffn_grow":
                cur_pred = predict_brick(cur_pred) * gate_brick(hidden)
            elif self.scheme == "interleave":
                predicted_brick = cur_pred.repeat_interleave(3, -1)[:, :_out_dim]
                gate_outs = gate_brick(hidden)
                g1, g2 = torch.chunk(gate_outs, 2, -1)
                cur_pred = predicted_brick * g1 + (1 - g1) * g2

            output_preds.append(cur_pred)
        return output_preds
