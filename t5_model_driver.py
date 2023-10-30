"""Training module."""
from functools import cached_property
from typing import Type, Optional, Union
from pathlib import Path

import pandas as pd
import torch
import transformers
import evaluate
from tqdm import tqdm

from data_tools import get_data_manager, T5ClassifierDataset


DEFAULT_DEVICE = MPS_DEVICE = torch.device("mps")
# We want to do Adam(MODEL.parameters(), lr=0.00001)
DEFAULT_OPTIMIZER_CLS = torch.optim.Adam
DEFAULT_LEARNING_RATE = 0.00001  # conservative, original learning rate
# DEFAULT_LEARNING_RATE = 0.001  # two orders of magnitude higher
# DEFAULT_LEARNING_RATE = 0.000001  # offset to counter the weights


class T5ModelDriver:
    """Primary class for running our classifier model.

    TODO:
    - parameterize epochs?
    - store results of training runs?
    - DRY out the training loop
    - facilitate other optimizer types
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataframe: pd.DataFrame,
        optimizer_cls: Type[torch.optim.Adam] = DEFAULT_OPTIMIZER_CLS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        device: torch.DeviceObjType = DEFAULT_DEVICE,
        **dm_kwargs,
    ):
        """Constructor"""
        self._model = model
        self._tokenizer = tokenizer
        self._data_manager = get_data_manager(
            dataframe, tokenizer, dataset_class=T5ClassifierDataset, **dm_kwargs
        )
        self._optimizer = optimizer_cls(model.parameters(), lr=learning_rate)
        self._device = device

    def save(
        self,
        model_path: Union[Path, str] = "models/ards_bc_model",
        tokenizer_path: Union[Path, str] = "models/ards_bc_tokenizer",
    ):
        self._model.save_pretrained(model_path)
        self._tokenizer.save_pretrained(tokenizer_path)

    @cached_property
    def dataframe(self):
        """Aliasing this for convenience."""
        return self._data_manager.dataframe

    def train(self):
        train_loss = 0
        val_loss = 0
        train_batch_count = 0
        val_batch_count = 0

        for epoch in range(2):
            self._model.train()  # Enter training mode
            for batch in tqdm(self._data_manager.train_loader, desc="Training batches"):
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(
                    self._device
                )
                weights = batch["weight"].to(self._device)  # num elements = batch size!
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )  # Seq2Seq*Output dataclass

                weighted_loss = (outputs.loss * weights).mean()

                # Reset gradients of (optimized) torch tensors and record tensor changes
                self._optimizer.zero_grad()
                # load gradient into tensor
                weighted_loss.backward()
                # Changes the state property of the optimizer
                self._optimizer.step()
                train_loss += weighted_loss.item()
                train_batch_count += 1

            print(f"{epoch + 1}/{2} -> Train loss: {train_loss / train_batch_count}")

            # Evaluation
            self._model.eval()
            for batch in tqdm(self._data_manager.val_loader, desc="Validation batches"):
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)
                decoder_attention_mask = batch["decoder_attention_mask"].to(
                    self._device
                )
                weights = batch["weight"].to(self._device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                )

                weighted_loss = (outputs.loss * weights).mean()

                self._optimizer.zero_grad()
                weighted_loss.backward()
                self._optimizer.step()
                val_loss += weighted_loss.item()
                val_batch_count += 1

            print(f"{epoch + 1}/{2} -> Validation loss: {val_loss / val_batch_count}")

    def predict_answer(self, context: str, ref_answer: Optional[str] = None):
        dataset = self._data_manager.dataset
        inputs = self._tokenizer(
            dataset.prompt_template(text=context),
            max_length=dataset.prompt_token_limit,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
        )

        input_ids = (
            torch.tensor(inputs["input_ids"], dtype=torch.int)
            .to(self._device)
            .unsqueeze(0)
        )
        attention_mask = (
            torch.tensor(inputs["attention_mask"], dtype=torch.int)
            .to(self._device)
            .unsqueeze(0)
        )

        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            min_new_tokens=2,
            max_new_tokens=10,
        )

        predicted_answer = self._tokenizer.decode(
            outputs.flatten(), skip_special_tokens=True
        )

        if ref_answer:
            # Load the Bleu metric
            bleu = evaluate.load("google_bleu")
            score = bleu.compute(
                predictions=[predicted_answer], references=[ref_answer]
            )

            print("Prompt: \n", dataset.prompt_template(text=context)[:100] + "...")
            return {
                "Reference Answer: ": ref_answer,
                "Predicted Answer: ": predicted_answer,
                "BLEU Score: ": score,
            }
        else:
            return predicted_answer
