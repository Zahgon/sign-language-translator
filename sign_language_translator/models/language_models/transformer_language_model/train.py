"""This module contains classes to train transformer language models.

Classes:
    LM_Dataset(torch.utils.data.Dataset): subclass for language model. has a process function that can convert text file into a list of tensors.
    LM_Trainer: Trainer class for language model. runs training loop, prints metrics and makes model checkpoints.
"""

from __future__ import annotations

import os
from glob import glob
from time import time
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from sign_language_translator.models.utils import (
    FullyLambdaLR,
    set_layers_trainability_,
)
from sign_language_translator.text.utils import make_ngrams

if TYPE_CHECKING:
    from sign_language_translator.models.language_models.transformer_language_model.model import (
        TransformerLanguageModel,
    )


class LM_Dataset(torch.utils.data.Dataset):  # type: ignore
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx][..., :-1], self.data[idx][..., 1:])

    @staticmethod
    def prepare(
        file_path: str,
        text_to_token_ids: Callable[[str], List[int]],
        # padding_token_id: int,
        max_sequence_length: int = 32,
        encoding="utf-8",
        dtype=torch.int,
    ) -> List[torch.Tensor]:
        """Process a text file into list of 2d torch tensors of shape (n_examples, n_tokens).

        Args:
            file_path (str): where the input file is stored
            text_to_token_ids (Callable[[str], List[int]]): a function that can process a line from file and convert it into a list of token ids.
            max_sequence_length (int, optional): make n_grams of sequences longer than this of size max_sequence_length. Defaults to 32.
            encoding (str, optional): the encoding used in the text file. Defaults to "utf-8".
            dtype (_type_, optional): the type of returned torch tensors. check the range of values a type can contain and choose the smallest to save space. Defaults to torch.int.

        Returns:
            List[torch.Tensor]: _description_
        """
        pass


class LM_Trainer:
    """class contains functions to train a language model built with pytorch.
    It is not designed to be generic rather it is specific to the TransformerLanguageModel.
    """

    def __init__(
        self,
        model: TransformerLanguageModel,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        epochs: int = 10,
        learning_rate: float = 1e-3,
        lr_lambda: Callable[
            [int, float, float], float
        ] = lambda epoch, base_lr, last_lr: base_lr,
        lr_update_step_count: Optional[int] = None,
        optimizer="adamw",
        seed: int = 0,
        model_output_renderer: Optional[
            Callable[[TransformerLanguageModel], str]
        ] = None,
        epoch_unfreeze_map: Optional[Dict[int, List[str]]] = None,
        class_weights: Optional[torch.Tensor] = None,
        max_gradient_norm: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer = (
            torch.optim.AdamW(model.parameters(), lr=learning_rate)
            if optimizer == "adamw"
            else torch.optim.Adam(model.parameters(), lr=learning_rate)
        )
        self.scheduler = FullyLambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.seed = seed
        self.model_output_renderer = model_output_renderer

        self.lr_update_step_count = lr_update_step_count
        self.epoch_unfreeze_map = epoch_unfreeze_map
        self.class_weights = class_weights
        self.max_gradient_norm = max_gradient_norm

    def train(self, input_sequences, outputs) -> Dict[str, float]:
        """Training loop. calculates loss on non-padding tokens.
        Multiples class weights with the loss to scale it for imbalanced token distributions.
        Clips gradients with norm > max_gradient_norm to avoid exploding gradients.

        Args:
            input_sequences (_type_): batch of sequences
            outputs (_type_): batch of target sequences in which each position contains target token for the input sequence upto that position.

        Returns:
            Dict[str, float]: the metrics tracked e.g. loss
        """
        history = {}
        # inference
        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model.forward(input_sequences)

        # reshape according to loss function's requirement
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = outputs.reshape(B * T)

        # calculate loss
        not_padding_mask = targets != self.model.padding_token_id
        loss = torch.nn.functional.cross_entropy(
            logits[not_padding_mask], targets[not_padding_mask], reduction="none"
        )
        history["loss"] = loss.mean().item()

        # data
        if self.class_weights is not None:
            loss_multiplier = self.class_weights[targets[not_padding_mask]]
            loss_multiplier -= loss_multiplier.mean()
            loss_multiplier /= loss_multiplier.std()
            loss_multiplier -= loss_multiplier.min() - 1
            loss *= loss_multiplier
            history["scaled_loss"] = loss.mean().item()

        # back-propagation
        loss.mean().backward()

        if self.max_gradient_norm:
            torch.nn.utils.clip_grad_norm_(  # type:ignore
                self.model.parameters(), self.max_gradient_norm
            )

        self.optimizer.step()

        return history

    def run(
        self,
        train_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        validation_batches: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        early_stop: bool = False,
        checkpoint_dir: str = "",
        checkpoint_step_count: int = 1000,
        model_output_step_count: int = 100,
        start_epoch_number: int = 0,
    ) -> Dict[str, List[float]]:
        """Run the training/validation loop and generate output & checkpoints.

        Returns:
            Dict[str, List[float]]: the tracked metrics
        """
        pass

    def validate(
        self, input_sequences: torch.Tensor, outputs: torch.Tensor
    ) -> Dict[str, float]:
        """the validation loop. infers the model on validation data without gradients or back propagation and calculates metrics.

        Args:
            input_sequences (torch.Tensor): batch of input tokens
            outputs (torch.Tensor): batch of target sequences

        Returns:
            Dict[str, float]: the tracked metrics
        """
        pass

    def checkpoint(self, checkpoint_dir: str, losses, epoch, steps_fraction) -> None:
        """save metrics in model and save model to disk."""
        pass

    def _average(self, values) -> float:
        pass


__all__ = [
    "LM_Dataset",
    "LM_Trainer",
]


if __name__ == "__main__":
    # TODO: write training script (for now, see notebooks repo)
    # https://github.com/sign-language-translator/notebooks/blob/main/model_training/transformer_lm_training.ipynb
    pass
