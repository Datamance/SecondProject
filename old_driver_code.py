"""Driver code for the T5 model."""
import os
import pickle
import warnings
from typing import Union
from pathlib import Path

import torch
import transformers

from t5_model_driver import T5ModelDriver

# Just keep this here for now, eventually get rid of it
HF_TOKEN_VALUE = "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"
HF_TOKEN_KEY = "HUGGING_FACE_HUB_TOKEN"
try:
    assert os.getenv(HF_TOKEN_KEY)
except AssertionError as e:
    os.environ[HF_TOKEN_KEY] = HF_TOKEN_VALUE
    # os.environ[HF_TOKEN_KEY] = getpass.getpass("Input Your Huggingface READ Token:")
    assert os.getenv(HF_TOKEN_KEY)

MPS_DEVICE = torch.device("mps")

# Stop being so noisy
warnings.filterwarnings("ignore")


def main() -> T5ModelDriver:
    """Driver code."""
    # 1) Get the training DF and process it
    training_df = get_training_df()

    tokenizer = transformers.T5TokenizerFast.from_pretrained("luqh/ClinicalT5-large")
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        "luqh/ClinicalT5-large",
        from_flax=True,
        return_dict=True,
        # torch_dtype=torch.float16,
    ).to(MPS_DEVICE)

    qa_model_driver = T5ModelDriver(model, tokenizer, training_df)
    qa_model_driver.train()
    qa_model_driver.save()

    return qa_model_driver


def get_training_df(file_path: Union[Path, str] = "data/training.pkl"):
    """Get the training DF.

    No processing for now.
    """
    with open(file_path, "rb") as f:
        training_df = pickle.load(f)

    return training_df


if __name__ == "__main__":
    qa_model_driver = main()
    dataframe = qa_model_driver.dataframe
