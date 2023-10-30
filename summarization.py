"""Driver code for the model.

I think this will actually just end up being a preprocessing module.
"""
import os
import sys
import yaml
import logging
import getpass
import pickle
import re
import pprint
import time
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
import sklearn as sl
from cleantext import clean
import nltk

import torch
import transformers

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
nltk.download("punkt")

# with open("qlora_config.yaml") as f:
#     QLORA_CONFIG = yaml.safe_load(f)

# For pubmedbert-bio-ext-summ, see
# https://huggingface.co/NotXia/pubmedbert-bio-ext-summ/blob/main/pipeline.py
# NotXia/longformer-bio-ext-summ operates much the same way, with same parameters and outputs
# https://huggingface.co/NotXia/longformer-bio-ext-summ/blob/main/pipeline.py
extractive_summarizer = transformers.pipeline(
    "summarization",
    model="NotXia/pubmedbert-bio-ext-summ",
    tokenizer=transformers.AutoTokenizer.from_pretrained(
        "NotXia/pubmedbert-bio-ext-summ"
    ),
    trust_remote_code=True,
    device=MPS_DEVICE,
)

# abstractive_model_name = "facebook/bart-large-cnn"
# abstractive_model_name = "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract"
# abstractive_model_name = "emilyalsentzer/Bio_ClinicalBert"
# abstractive_summarizer = transformers.pipeline(
#     "summarization", model=abstractive_model_name
# )
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#     "emilyalsentzer/Bio_ClinicalBert"
# )
# model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
#     "emilyalsentzer/Bio_ClinicalBert"
# )
# tokenizer = transformers.T5TokenizerFast.from_pretrained("luqh/ClinicalT5-large")
# # No pytorch_model.bin, but there is a file for flax weights, so we have to pass `from_flax=True`.
# model = transformers.T5ForConditionalGeneration.from_pretrained(
#     "luqh/ClinicalT5-large", from_flax=True
# ).to(MPS_DEVICE)
abstractive_summarizer_name = "Falconsai/medical_summarization"
abstractive_summarizer_tokenizer = transformers.AutoTokenizer.from_pretrained(
    abstractive_summarizer_name
)
abstractive_summarizer_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    abstractive_summarizer_name
).to(MPS_DEVICE)
# model = torch.compile(model, backend="aot_eager")

# I will use this tokenizer to determine which things need to be summarized.
clinical_longformer_model_name = "yikuan8/Clinical-Longformer"
clinical_longformer_tokenizer = transformers.AutoTokenizer.from_pretrained(
    clinical_longformer_model_name
)


def get_training_df(path: str = "data/training.pkl"):
    """Get the training DF and process it."""
    with open(path, "rb") as f:
        training_df = pickle.load(f)

    return training_df


def get_processed_df(
    dataframe: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizerBase = clinical_longformer_tokenizer,
    max_tokens: int = 4096,
):
    first_pass_cleaner = partial(clean, no_line_breaks=True)
    print("starting first pass of cleaning...")
    start = time.perf_counter()
    dataframe["first_pass_processing"] = dataframe.text.apply(first_pass_cleaner)
    end = time.perf_counter()
    print(f"Done first pass clean after {end - start:3f} seconds")
    # If the text is under the token limit, we can use the first pass cleaning.
    dataframe["final_pass_processing"] = dataframe.first_pass_processing
    for i in range(len(dataframe)):
        text = dataframe.loc[i].first_pass_processing
        token_indices = tokenizer.encode(text)
        token_count = len(token_indices)
        if token_count > max_tokens:
            # Maximize tokens + decrease time to summary by depending on ratio
            ratio = round((max_tokens / token_count) * 0.9, 5)
            print(
                f"Too many tokens ({token_count} > {max_tokens}) for record #{i}, "
                f"extracting summary {ratio * 100:.3f}% length of original..."
            )
            start = time.perf_counter()
            summary = get_extractive_notes_summary(
                text, strategy="ratio", strategy_args=ratio
            )
            end = time.perf_counter()
            total_seconds = end - start
            print(f"Took {total_seconds} seconds to produce summary")
            new_token_indices = tokenizer.encode(summary)
            print(f"Summary is {len(new_token_indices)} tokens!")
            print(summary)
            dataframe.loc[i, "final_pass_processing"] = summary

    return dataframe


def get_extractive_notes_summary(
    first_pass_processed_text: str,
    strategy: str = "length",
    strategy_args: Union[float, int] = 3000,
):
    """Get extractive summary for a specific note."""
    sentences = nltk.sent_tokenize(first_pass_processed_text)

    # 4. Run the summarizer
    results = extractive_summarizer(
        {"sentences": sentences}, strategy=strategy, strategy_args=strategy_args
    )

    extracted, indices = results

    return " ".join(extracted)


def get_abstractive_notes_summary(unprocessed_text: str):
    """Get abstractive summary for some text."""
    # 1. Pre-process text
    print("Preprocessing...")
    cruft_removed = unprocessed_text.replace("\n", " ").replace("//", "")
    preprocessed_sentences = [
        sentence for sentence in nltk.sent_tokenize(cruft_removed) if len(sentence) > 10
    ]
    preprocessed_text = " ".join(preprocessed_sentences)

    # 2. Tokenize
    print("Tokenizing...")
    tokens = abstractive_summarizer_tokenizer.encode(
        "summarize: " + preprocessed_text,
        return_tensors="pt",
        padding="longest",
        max_length=4096,
        # truncation=True,
    ).to(MPS_DEVICE)
    # 3. Generate text
    print("Starting generative process...")
    with torch.no_grad():
        model_output = abstractive_summarizer_model.generate(
            tokens,
            num_beams=6,
            no_repeat_ngram_size=2,
            min_length=20,
            max_length=500,
        )
    print("Starting decoding...")
    output = abstractive_summarizer_tokenizer.decode(
        model_output[0], skip_special_tokens=True
    )

    return output


if __name__ == "__main__":
    training_df = get_training_df()
    processed_df = get_processed_df(training_df)
    # test_text = training_df.loc[100].text
    # summary = get_extractive_notes_summary(test_text)
    # for i in range(1000):
    #     start = time.perf_counter()
    #     test_record = training_df.loc[i]
    #     test_text = test_record.text
    #     charlen = test_record.raw_charlen
    #     summary = get_extractive_notes_summary(test_text)
    #     end = time.perf_counter()
    #     print(summary)
    #     print(f"{(end - start) / charlen:.7f} seconds per character")
