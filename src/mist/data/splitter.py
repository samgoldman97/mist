"""splitter.py"""

from typing import List, Tuple, Iterator
import logging
import pandas as pd
import numpy as np

from mist.data.data import Spectra, Mol

DATASET = List[Tuple[Spectra, Mol]]


def get_splitter(splitter_name: str, **kwargs):
    """get_splitter.

    Args:
        splitter_name (str): splitter_name
        kwargs:
    """
    return {"random": RandomSpectraSplitter, "preset": PresetSpectraSplitter,}[
        splitter_name
    ](**kwargs)


class SpectraSplitter(object):
    """SpectraSplitter."""

    def __init__(
        self,
        split_sizes: List[float],
        num_folds: int = 1,
        reshuffle_val: bool = False,
        **kwargs,
    ):
        """init.

        Args:
            split_sizes (List[float]): List of 3 floats
            num_folds (int): Nummber of folds in the dataset

        """
        self.reshuffle_val = True
        split_sum = sum(split_sizes)
        assert split_sum <= 1
        if (sum(split_sizes)) < 1:
            logging.warning(f"Split sizes add up to {split_sum} not 1")

        assert len(split_sizes) == 3

        # Define splits
        self.train, self.val, self.test = split_sizes

        self.single_split = True
        self.num_folds = num_folds

        if self.single_split:
            self.num_folds = 1

    def split_from_indices(
        self,
        full_dataset: DATASET,
        train_inds: np.ndarray,
        val_inds: np.ndarray,
        test_inds: np.ndarray,
    ) -> Tuple[DATASET]:
        """split_from_indices.

        Split based upon some indices

        Args:
            full_dataset (Datset): Full dataset
            train_inds (np.ndarray): Train indices
            val_inds (np.ndarray): Train indices
            test_inds (np.ndarray): Train indices

        Returns:
            Tuple of train, val, test datasets
        """
        full_dataset = np.array(full_dataset)
        train_sub = full_dataset[train_inds].tolist()
        val_sub = full_dataset[val_inds].tolist()
        test_sub = full_dataset[test_inds].tolist()
        return (train_sub, val_sub, test_sub)

    def _get_split(self, full_dataset: DATASET) -> Tuple[DATASET]:
        """_get_split.

        Get a single split of the data

        Args:
            self:
            full_dataset (SpectraMolDataset): full_dataset

        Returns:
            Tuple[SpectraMolDataset]:
        """
        raise NotImplementedError()


class RandomSpectraSplitter(SpectraSplitter):
    """RandomSpectraSplitter."""

    def __init__(self, **kwargs):
        """RandomSpectraSplitter."""
        super().__init__(**kwargs)

    def _get_split(self, full_dataset: DATASET) -> Tuple[DATASET]:
        """_get_split.

        Get a single split of the data

        Args:
            self:
            full_dataset (DATASET): full_dataset

        Returns:
            Tuple[DATASET]:
        """

        dataset_size = len(full_dataset)
        possible_inds = np.arange(dataset_size)

        # Shuffled
        np.random.shuffle(possible_inds)

        first_ind = int(np.ceil(dataset_size * self.train))
        second_ind = first_ind + int(np.ceil(dataset_size * self.val))
        third_ind = second_ind + int(np.ceil(dataset_size * self.test))

        train_inds = possible_inds[:first_ind]
        val_inds = possible_inds[first_ind:second_ind]
        test_inds = possible_inds[second_ind:third_ind]
        return self.split_from_indices(full_dataset, train_inds, val_inds, test_inds)


class PresetSpectraSplitter(SpectraSplitter):
    """PresetSpectraSplitter."""

    def __init__(self, split_file: str = None, **kwargs):
        """PresetSpectraSplitter."""
        super().__init__(**kwargs)
        if split_file is None:
            raise ValueError("KFold splitter requires split_file arg.")

        self.split_file = split_file

        # Resetting num folds to 10 regardless
        self.split_df = pd.read_csv(self.split_file)
        folds = set(self.split_df.columns)
        folds.remove("name")

        self.num_folds = len(folds)
        self.folds = sorted(list(folds))

    def get_splits(self, full_dataset: DATASET) -> Iterator[Tuple[str, Tuple[DATASET]]]:
        """get_splits.

        Yields all folds from the preset split

        Args:
            full_dataset (DATASET): Full dataset
        Returns:
            Iterator[ Tuple[str, Tuple[DATASET]]]
        """
        # Map name to index
        spec_names = [i.get_spec_name() for i, j in full_dataset]

        names_to_index = dict(zip(spec_names, np.arange(len(full_dataset))))
        val_num = int(len(names_to_index) * self.val)

        for index, fold_name in enumerate(self.folds):
            if index > 0 and self.single_split:
                return

            fold_entries = self.split_df[fold_name]
            train_entries = fold_entries == "train"
            test_entries = fold_entries == "test"
            val_entries = fold_entries == "val"

            test_inds = np.array(
                [
                    names_to_index.get(i)
                    for i in self.split_df["name"][test_entries]
                    if i in names_to_index
                ]
            )

            if np.sum(val_entries) > 0 and not self.reshuffle_val:
                train_inds = np.array(
                    [
                        names_to_index.get(i)
                        for i in self.split_df["name"][train_entries]
                        if i in names_to_index
                    ]
                )
                val_inds = np.array(
                    [
                        names_to_index.get(i)
                        for i in self.split_df["name"][val_entries]
                        if i in names_to_index
                    ]
                )
            else:
                train_entries = np.logical_or(val_entries, train_entries)
                train_inds = np.array(
                    [
                        names_to_index.get(i)
                        for i in self.split_df["name"][train_entries]
                        if i in names_to_index
                    ]
                )
                # Compute val indices
                val_inds = np.random.choice(list(train_inds), val_num, replace=False)
                val_inds = set(val_inds)
                # Remove val inds from train
                train_inds = set(train_inds).difference(val_inds)

            # Convert them all back to lists
            train_inds, val_inds, test_inds = (
                list(train_inds),
                list(val_inds),
                list(test_inds),
            )

            new_split = self.split_from_indices(
                full_dataset, train_inds, val_inds, test_inds
            )
            return (fold_name, new_split)
