"""misc_utils.py"""
from collections import defaultdict
from pathlib import Path
import sys
import copy
import logging
from typing import List, Iterable, Iterator
from itertools import islice

import numpy as np
import torch

import yaml
import pytorch_lightning as pl

from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment


class ConsoleLogger(LightningLoggerBase):
    """Custom console logger class"""

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self):
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        pass

    @property
    @rank_zero_experiment
    def version(self):
        pass

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):

        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status):
        pass


def setup_train(save_dir: Path, kwargs):
    """setup.

    Set seed, define logger, dump args, & update kwargs for debug

    """
    # Seed everything
    pl.utilities.seed.seed_everything(kwargs.get("seed"))

    # Define default root dir
    setup_logger(save_dir, debug=kwargs["debug"])

    # Dump args
    yaml_args = yaml.dump(kwargs, indent=2, default_flow_style=False)
    logging.info(yaml_args)

    # Dump args
    with open(save_dir / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Hard code max_count for debugging!
    if kwargs.get("debug") == "test":
        kwargs["max_epochs"] = 3
        kwargs["max_count"] = 100
    elif kwargs.get("debug") == "test_overfit":
        kwargs["min_epochs"] = 1000
        kwargs["max_epochs"] = None
        kwargs["max_count"] = 100
    else:
        kwargs["max_count"] = None
        pass


def setup_logger(save_dir, log_name="output.log", debug=False):
    """Create output directory"""

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    log_file = save_dir / log_name

    if debug is not False:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


def bin_spectra(
    spectras: List[np.ndarray], num_bins: int = 2000, upper_limit: int = 1000
) -> np.ndarray:
    """bin_spectra.

    Args:
        spectras (List[np.ndarray]): Input list of spectra tuples
            [(header, spec array)]
        num_bins (int): Number of discrete bins from [0, upper_limit)
        upper_limit (int): Max m/z to consider featurizing

    Return:
        np.ndarray of shape [channels, num_bins]
    """
    bins = np.linspace(0, upper_limit, num=num_bins)
    binned_spec = np.zeros((len(spectras), len(bins)))
    for spec_index, spec in enumerate(spectras):

        # Convert to digitized spectra
        digitized_mz = np.digitize(spec[:, 0], bins=bins)

        # Remove all spectral peaks out of range
        in_range = digitized_mz < len(bins)
        digitized_mz, spec = digitized_mz[in_range], spec[in_range, :]

        # Add the current peaks to the spectra
        # Use a loop rather than vectorize because certain bins have conflicts
        # based upon resolution
        for bin_index, spec_val in zip(digitized_mz, spec[:, 1]):
            binned_spec[spec_index, bin_index] += spec_val

    return binned_spec


def merge_norm_spectra(spec_tuples, precision=4) -> np.ndarray:
    """ merge_norm_spectra.

    Take a list of mz, inten tuple arrays and merge them by 4 digit precision
    
    """
    mz_to_inten_pair = {}
    for i in spec_tuples:
        for tup in i:
            mz, inten = tup
            mz_ind = np.round(mz, precision)
            cur_pair = mz_to_inten_pair.get(mz_ind)
            if cur_pair is None:
                mz_to_inten_pair[mz_ind] = tup
            elif inten > cur_pair[1]:
                mz_to_inten_pair[mz_ind] = (mz_ind, inten)
            else:
                pass

    merged_spec = np.vstack([v for k,v  in mz_to_inten_pair.items() ])
    merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()
    return merged_spec


def norm_spectrum(binned_spec: np.ndarray) -> np.ndarray:
    """norm_spectrum.

    Normalizes each spectral channel to have norm 1
    This change is made in place

    Args:
        binned_spec (np.ndarray) : Vector of spectras

    Return:
        np.ndarray where each channel has max(1)
    """

    spec_maxes = binned_spec.max(1)

    non_zero_max = spec_maxes > 0

    spec_maxes = spec_maxes[non_zero_max]
    binned_spec[non_zero_max] = binned_spec[non_zero_max] / spec_maxes.reshape(-1, 1)

    return binned_spec


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = torch.div(index, dim, rounding_mode="trunc")
    return tuple(reversed(out))


def clamped_log_np(x, _min=-100):
    res = np.log(x)
    res[res <= _min] = _min
    return res


def batches(it: Iterable, chunk_size: int) -> Iterator[List]:
    """Consume an iterable in batches of size chunk_size""" ""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


def pad_packed_tensor(input, lengths, value):
    """pad_packed_tensor"""
    old_shape = input.shape
    device = input.device
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths, dtype=torch.int64, device=device)
    else:
        lengths = lengths.to(device)
    max_len = (lengths.max()).item()

    batch_size = len(lengths)
    x = input.new(batch_size * max_len, *old_shape[1:])
    x.fill_(value)

    # Initialize a tensor with an index for every value in the array
    index = torch.ones(len(input), dtype=torch.int64, device=device)

    # Row shifts
    row_shifts = torch.cumsum(max_len - lengths, 0)

    # Calculate shifts for second row, third row... nth row (not the n+1th row)
    # Expand this out to match the shape of all entries after the first row
    row_shifts_expanded = row_shifts[:-1].repeat_interleave(lengths[1:])

    # Add this to the list of inds _after_ the first row
    cumsum_inds = torch.cumsum(index, 0) - 1
    cumsum_inds[lengths[0] :] += row_shifts_expanded
    x[cumsum_inds] = input
    return x.view(batch_size, max_len, *old_shape[1:])


def reverse_packed_tensor(packed_tensor, lengths):
    """reverse_packed_tensor.

    Args:
        packed tensor: Batch x  length x feat_dim
        lengths : Batch
    Return:
        [batch,length] x feat_dim
    """
    device = packed_tensor.device
    batch_size, batch_len, feat_dim = packed_tensor.shape
    max_length = torch.arange(batch_len).to(device)
    indices = max_length.unsqueeze(0).expand(batch_size, batch_len)
    bool_mask = indices < lengths.unsqueeze(1)
    output = packed_tensor[bool_mask]
    return output
