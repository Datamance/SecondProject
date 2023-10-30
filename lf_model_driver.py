"""Training module."""
import pickle
import warnings
from pathlib import Path
from typing import Optional, Type, Union

import evaluate
import pandas as pd
import peft  # get_peft_model, LoraConfig, TaskType
import torch
import transformers
from tqdm import tqdm

from data_tools import get_trainer

warnings.filterwarnings("ignore")


DEFAULT_DEVICE = MPS_DEVICE = torch.device("mps")


def get_training_df(file_path: Union[Path, str] = "data/training.pkl"):
    """Get the training DF.

    No processing for now.
    """
    with open(file_path, "rb") as f:
        training_df = pickle.load(f)

    return training_df


if __name__ == "__main__":
    clinical_longformer_model_name = "yikuan8/Clinical-Longformer"
    # dataframe = get_training_df()
    dataframe = pd.read_csv("training_processed_lite.csv").rename(
        columns={"final_pass_processing": "text"}
    )
    print(dataframe.head(5))
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        clinical_longformer_model_name
    )
    # GPTQ does not work on Apple Silicon
    # quantization_config = transformers.GPTQConfig(bits=4, tokenizer=tokenizer)
    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(
        clinical_longformer_model_name,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.float32,
        # quantization_config=quantization_config,
    ).to(device=MPS_DEVICE, dtype=torch.float32)
    # TODO: LoRA when your model is at least scoring well.
    # peft_config = peft.LoraConfig(
    #     target_modules=[
    #         "dense",
    #         "out_proj",
    #         "query",
    #         "key",
    #         "value",
    #         "query_global",
    #         "key_global",
    #         "value_global",
    #     ],
    #     task_type=peft.TaskType.SEQ_CLS,
    #     inference_mode=False,
    #     r=4,
    #     lora_alpha=32,
    #     lora_dropout=0.1,
    # )
    # classifier = peft.get_peft_model(classifier, peft_config)
    # classifier.print_trainable_parameters()
    trainer = get_trainer(dataframe=dataframe, model=classifier, tokenizer=tokenizer)
    # trainer.evaluate()  # See how good the model is with no training
    trainer.train()  # See how much better it gets
