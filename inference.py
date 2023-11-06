"""Perform inference with already trained model."""

import pickle
from typing import Union, Optional
from pathlib import Path

import transformers
import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_DEVICE = MPS_DEVICE = torch.device("mps")


BASE_LONGFORMER_MODEL_NAME = "yikuan8/Clinical-Longformer"
BASE_LONGFORMER_TOKENIZER = transformers.AutoTokenizer.from_pretrained(
    BASE_LONGFORMER_MODEL_NAME
)

# Change this checkpoint name to either the HF repo, or local checkpoint that you want to target.
# CHECKPOINT_NAME = "Datamance/Longformer-ARDS-classifier"
# CHECKPOINT_NAME = "models/longformer_classifier/checkpoint-13088"
CHECKPOINT_NAME = "models/longformer_classifier_1/checkpoint-3272"

FINETUNED_LONGFORMER_NAME = CHECKPOINT_NAME
FINETUNED_LONGFORMER_MODEL = (
    transformers.AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT_NAME, device_map="auto"
    )
)


def predict(
    test_df_path: Union[Path, str] = "data/test.csv",
    tokenizer: transformers.PreTrainedTokenizerBase = BASE_LONGFORMER_TOKENIZER,
    model: transformers.PreTrainedModel = FINETUNED_LONGFORMER_MODEL,
    cutoff_probability: Optional[float] = 0.0010,
):
    test_df = pd.read_csv(test_df_path)
    # Assume we have processed this data. If we haven't, this should bork.
    notes_processed = test_df.final_pass_processing
    notes_count = len(notes_processed)

    # Initialize values
    probabilities = []
    decisions = []

    # Set model to evaluate mode
    model.eval()

    with torch.no_grad():
        for idx, note in enumerate(notes_processed):
            # No restrictions on tokenizer here! Preprocessing should have taken care of all of that,
            # so we want noisy warnings here.
            inputs = tokenizer(note, return_tensors="pt").to(device=DEFAULT_DEVICE)
            outputs = model(**inputs)
            probability = torch.sigmoid(outputs.logits).item()
            probabilities.append(probability)
            message = f"Sample #{idx + 1} Probability of ARDS: {100 * probability:.2f}%"
            if cutoff_probability is not None:
                decision = probability >= cutoff_probability
                decisions.append(decision)
                message += f", vote {'Yes' if decision else 'No'} based on threshold of {100 * cutoff_probability:.2f}%"
            message += f". {((idx + 1) / notes_count) * 100:.4f}% done..."

            print(message)

    test_df["ards_probabilities"] = probabilities
    if decisions:
        test_df["decision"] = decisions

    return test_df


if __name__ == "__main__":
    predictions = predict()
    predictions.decision.to_csv("output/test_result_1.csv", index=False, header=False)
