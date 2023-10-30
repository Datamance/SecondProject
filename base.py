"""Base classes and data type declarations."""

from abc import ABCMeta, abstractmethod
from typing import NamedTuple, Callable, Sequence
from functools import cached_property, cache
import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
import transformers


DEFAULT_DEVICE = torch.device("mps")


class BaseARDSDataset(Dataset, metaclass=ABCMeta):
    """Base for the ARDS dataset.

    Basically a very light wrapper around the expected format of the pandas
    DF that was provided with our assignment. This is why we include "weights" -
    we *absolutely* should have dataset constructs that account for the imbalance
    intrinsic to this particular dataset.

    TODO: get target length out of here and leave it to the T5 constructs
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerBase,
        prompt_length: int,
        target_length: int,
        weights: Sequence[float],
        device: torch.DeviceObjType = DEFAULT_DEVICE,
    ):
        super().__init__()
        self.df = dataframe
        self.tokenizer = tokenizer
        self.prompt_token_limit = prompt_length
        self.target_token_limit = target_length
        self.weights = weights
        self._device = device

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError(
            "You should implement __getitem__ based on your classifier and training loop implementation."
        )

    @cached_property
    def context(self):
        return self.df.text

    @cached_property
    def answer(self):
        return self.df.label

    @cache
    def _get_processed_context(self, idx) -> str:
        """Get a cleaned up string."""
        unprocessed_text = self.context[idx]
        cruft_removed = unprocessed_text.replace("\n", " ").replace("//", "")
        preprocessed_sentences = [
            sentence
            for sentence in nltk.sent_tokenize(cruft_removed)
            if len(sentence) > 10
        ]

        preprocessed_text = " ".join(preprocessed_sentences)
        return preprocessed_text


class DataManager(NamedTuple):
    dataframe: pd.DataFrame
    train_data: pd.DataFrame
    val_data: pd.DataFrame
    train_sampler: RandomSampler
    val_sampler: RandomSampler
    dataset: BaseARDSDataset
    train_loader: DataLoader
    val_loader: DataLoader
