from typing import Callable, Sequence, Type, Optional, Union

import pandas as pd
import numpy as np
from dataclasses import dataclass
import gc

import torch
import evaluate
import peft
import nltk
from torch.utils import data as torch_data
import transformers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import TrainingArguments, TrainerState, TrainerControl

from base import BaseARDSDataset, DataManager

nltk.download("punkt")

DEFAULT_DEVICE = torch.device("mps")

DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_EPOCH_COUNT = 2
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_PROMPT_LENGTH = 4000  # Question Length, in tokens
DEFAULT_TARGET_LENGTH = 5  # Target Length ("Yes" or "No")
DEFAULT_BATCH_SIZE = 2
DEFAULT_TEST_FRACTION = 0.2

BINARY_CLASSIFIER_PROMPT_TEMPLATE = "Text: {text}\nCategory: [Yes/No]".format


class T5ClassifierDataset(BaseARDSDataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        dataframe: pd.DataFrame,
        prompt_length: int,
        target_length: int,
        prompt_template: Callable[..., str] = BINARY_CLASSIFIER_PROMPT_TEMPLATE,
    ):
        super().__init__(
            dataframe,
            tokenizer,
            prompt_length,
            target_length,
            compute_weights(dataframe),
        )

        self.prompt_template = prompt_template

    def __len__(self):
        """Get the length of the Dataset.

        Use "text" from original dataframe since we infer both question and answer.
        """
        return len(self.context)

    def __getitem__(self, idx):
        # 1. Make the prompt.
        context = self._get_processed_context(idx)
        prompt = self.prompt_template(text=context)
        prompt_tokenized = self.tokenizer(
            prompt,
            max_length=self.prompt_token_limit,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        # 2. Encode the answer as well.
        label = self.answer[idx]
        answer = "Yes" if label else "No"
        answer_tokenized = self.tokenizer(
            answer,
            max_length=self.target_token_limit,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
        )

        labels = torch.tensor(answer_tokenized["input_ids"], dtype=torch.int)
        # the output loss property is initialized with CrossEntropyLoss(ignore_index=-100),
        # See code for T5ForConditionalGeneration:
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1778
        labels[labels == 0] = -100

        # label should be a bool; Python treats False as 0 and True as 1.
        weight = self.weights[label]

        # These are the arguments that are passed to model's __call__ method.
        # Also, remember that everything in here has to be a torch tensor!
        # TODO: cast weight to float16 when you find out what's going on with the mps mismatch error.
        return {
            "input_ids": torch.tensor(prompt_tokenized["input_ids"], dtype=torch.int),
            "attention_mask": torch.tensor(
                prompt_tokenized["attention_mask"], dtype=torch.int
            ),
            "labels": labels,
            "decoder_attention_mask": torch.tensor(
                answer_tokenized["attention_mask"], dtype=torch.int
            ),
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def get_data_manager(
    dataframe: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizerBase,
    prompt_length: int = DEFAULT_PROMPT_LENGTH,
    target_length: int = DEFAULT_TARGET_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    dataset_class: Type[BaseARDSDataset] = T5ClassifierDataset,
) -> DataManager:
    """External constructor for old convenience container class.

    TODO: Deprecate this, as transformers.Trainer does what we need and more.
    """
    train_data, val_data = train_test_split(
        dataframe, test_size=test_fraction, stratify=dataframe.label, random_state=42
    )

    # WeightedRandomSampler would not give us what we want - the contract of samplers is
    # simply to yield integer indices. This would solve the problem by over-sampling the
    # positive class, which would require sampling with replacement. Instead, we can
    # leverage the Dataset class's __getitem__ return value to bestow weights.
    train_sampler = torch_data.RandomSampler(train_data.index)
    val_sampler = torch_data.RandomSampler(val_data.index)

    weights = compute_weights(dataframe)

    qa_dataset = dataset_class(
        dataframe, tokenizer, prompt_length, target_length, weights
    )

    train_loader = torch_data.DataLoader(
        qa_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
    )
    val_loader = torch_data.DataLoader(
        qa_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
    )

    return DataManager(
        dataframe=dataframe,
        train_data=train_data,
        val_data=val_data,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        dataset=qa_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
    )


class LongformerRegressorDataset(BaseARDSDataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: transformers.PreTrainedTokenizerBase,
        prompt_length: int,
        target_length: int,
        weights: Sequence[float],
        device: torch.DeviceObjType = DEFAULT_DEVICE,
    ):
        """Dataset for casting the ARDS prediction task to a regression problem.

        I think this is the best way to do this, because we simultaneously get more information
        about the confidence of the classifier, while also being able to easily force a choice via
        dynamic thresholding.
        """
        super().__init__(
            dataframe=dataframe,
            tokenizer=tokenizer,
            prompt_length=prompt_length,
            target_length=target_length,
            weights=weights,
            device=device,
        )

    def __len__(self):
        """Get the length of the Dataset.

        Use "text" from original dataframe since we infer both question and answer.
        """
        return len(self.context)

    def __getitem__(self, idx):
        # 1. Make the prompt.
        context = self._get_processed_context(idx)
        prompt_tokenized = self.tokenizer(
            context,
            max_length=self.prompt_token_limit,
            padding="max_length",
            truncation=True,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        # Should contain input_ids and attention_mask already (I think as int64)
        # Notice how the keys are pluralized, this is to cue us correctly during the training loop
        # that the tensor shape is predicated on batch size.
        batch_components = {
            k: v.squeeze().to(device=self._device) for k, v in prompt_tokenized.items()
        }

        # answer[idx] will be a boolean.
        # TODO: is there a way for pytorch to keep this as a boolean?
        label = self.answer[idx]
        batch_components["labels"] = torch.tensor(label, dtype=torch.float32).to(
            device=self._device
        )

        # We let the TrainingArguments bring this in now
        # weight = self.weights[label]
        # batch_components["weights"] = torch.tensor(weight, dtype=torch.float32).to(
        #     device=self._device
        # )

        return batch_components


def compute_weights(
    dataframe: pd.DataFrame,
    label_name: str = "label",
    alpha: float = 1.0,
    beta: float = 0.999,
) -> Sequence[float]:
    """This is a highly imbalanced data set, so we need weights."""
    total_count = len(dataframe)
    num_negative = len(dataframe[dataframe[label_name] == 0])
    num_positive = len(dataframe[dataframe[label_name] == 1])

    assert total_count == num_positive + num_negative

    # These suck
    # negative_weight = total_count / (2 * num_negative)
    # positive_weight = total_count / (2 * num_positive)
    # Let's try ISNS (inverse square number of samples) instead
    # negative_weight = 1 / np.sqrt(num_negative)
    # positive_weight = 1 / np.sqrt(num_positive)
    # If that sucks, we'll try ENS (effective number of samples) instead
    # negative_weight = compute_ens_weight(num_negative, alpha=alpha, beta=beta)
    # positive_weight = compute_ens_weight(num_positive, alpha=alpha, beta=beta)
    # We could try the scheme suggested by the torch docs for BCEWithLogitsLoss...
    negative_weight = 1.0
    positive_weight = num_negative / num_positive
    # Last resort, try manually tweaking weights
    # negative_weight = 1
    # positive_weight = 2
    print(
        f"Negative class weight: {negative_weight}\nPositive class weight: {positive_weight}"
    )

    return negative_weight, positive_weight


def compute_ens_weight(
    samples_in_class: int, alpha: float = 1, beta: float = 0.999
) -> float:
    """Compute ENS - Effective Number of Samples.

    Hopefully this is better than ISNS or INS.
    """
    effective_samples = (1 - beta**samples_in_class) / (1 - beta)
    return alpha / effective_samples


@dataclass
class TrainingArgumentsWithClassWeights(transformers.TrainingArguments):
    class_weights: Sequence[float] = (1,)


class TrainerWithWeights(transformers.Trainer):
    """Trainer class with a some changes geared toward our needs."""

    args: TrainingArgumentsWithClassWeights

    def compute_loss(
        self, model: transformers.PreTrainedModel, inputs: dict, return_outputs=False
    ):
        """Use weighted binary cross entropy loss.

        By default, the Longformer transformer models tend to use regular
        cross entropy. That will absolutely not work for our case.
        """
        labels = inputs.get("labels")
        # Pop weights off, since this is not expected as model input.
        # weights = inputs.pop("weights")
        neg_weight, pos_weight = self.args.class_weights
        pos_weight = torch.tensor(pos_weight / neg_weight, dtype=torch.float32)
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss - do I need to cast this to the model.device?
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Labels can't be bools here, which is fine I guess
        loss = loss_fct(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    # Calculate precision, recall, and F1-score
    # precision = precision_score(labels, preds, average="weighted")
    # recall = recall_score(labels, preds, average="weighted")
    f1 = f1_score(labels, preds, average="weighted")

    return {
        # "precision": precision,
        # "recall": recall,
        "f1": f1
    }


class MpsCacheClearCallback(transformers.TrainerCallback):
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        gc.collect()
        torch.mps.empty_cache()
        gc.collect()


def get_trainer(
    dataframe: pd.DataFrame,
    model: Union[transformers.PreTrainedModel, peft.PeftModel],
    tokenizer: transformers.PreTrainedTokenizerBase,
    save_location: str = "models/longformer_classifier_1",
    learning_rate: float = DEFAULT_LEARNING_RATE,
    prompt_length: int = DEFAULT_PROMPT_LENGTH,
    target_length: int = DEFAULT_TARGET_LENGTH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epoch_count: int = DEFAULT_EPOCH_COUNT,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    dataset_class: Type[BaseARDSDataset] = LongformerRegressorDataset,
) -> transformers.Trainer:
    """Construct a trainer."""
    weights = compute_weights(dataframe)
    # 1. Set hyperparameters for training
    training_args = TrainingArgumentsWithClassWeights(
        output_dir=save_location,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch_count,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        weight_decay=weight_decay,
        gradient_accumulation_steps=2,
        # skip_memory_metrics=False,  # breaks with MPS
        # log_level="debug",
        class_weights=weights,
    )

    # 2. Construct data loaders
    train_data, val_data = train_test_split(
        dataframe,
        test_size=test_fraction,
        stratify=dataframe.label,
    )
    # TODO: Find out if we need to modify DataLoaderShard.BatchSampler
    #    it looks like even when we override _get_train_sampler, a SequentialSampler is loaded.
    train_data = train_data.reset_index()
    val_data = val_data.reset_index()

    training_dataset = dataset_class(
        train_data, tokenizer, prompt_length, target_length, weights
    )
    validation_dataset = dataset_class(
        val_data, tokenizer, prompt_length, target_length, weights
    )

    # 3. Construct and return Trainer.
    trainer = TrainerWithWeights(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[MpsCacheClearCallback()],
    )

    return trainer
