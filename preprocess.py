"""Preprocessing module."""

import pickle
import time
from functools import partial
from typing import Union

import nltk
import pandas as pd
import torch
import transformers
from cleantext import clean

DEFAULT_DEVICE = MPS_DEVICE = torch.device("mps")
nltk.download("punkt")


EXTRACTIVE_SUMMARIZER = transformers.pipeline(
    "summarization",
    model="NotXia/pubmedbert-bio-ext-summ",
    tokenizer=transformers.AutoTokenizer.from_pretrained(
        "NotXia/pubmedbert-bio-ext-summ"
    ),
    trust_remote_code=True,
    device=DEFAULT_DEVICE,
)


# I will use this tokenizer to determine which things need to be summarized.
LONGFORMER_MODEL_NAME = "yikuan8/Clinical-Longformer"
LONGFORMER_TOKENIZER = transformers.AutoTokenizer.from_pretrained(LONGFORMER_MODEL_NAME)


def get_dataframe(path: str = "data/training.pkl") -> pd.DataFrame:
    """Get the training DF and process it."""
    with open(path, "rb") as f:
        df = pickle.load(f)

    if type(df) is pd.Series:  # it's the test DF.
        return pd.DataFrame({"text": df})
    else:  # It's the training DF.
        return df


def get_processed_df(
    dataframe: pd.DataFrame,
    tokenizer: transformers.PreTrainedTokenizerBase = LONGFORMER_TOKENIZER,
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
    summarizer_pipeline: transformers.Pipeline = EXTRACTIVE_SUMMARIZER,
    strategy: str = "length",
    strategy_args: Union[float, int] = 3000,
):
    """Get extractive summary for a specific note."""
    sentences = nltk.sent_tokenize(first_pass_processed_text)

    # 4. Run the summarizer
    results = summarizer_pipeline(
        {"sentences": sentences}, strategy=strategy, strategy_args=strategy_args
    )

    extracted, indices = results

    return " ".join(extracted)


if __name__ == "__main__":
    # training_df = get_dataframe()
    # processed_df = get_processed_df(training_df)
    test_df = get_dataframe("data/test.pkl")
    processed_df = get_processed_df(test_df)

    processed_df.to_csv("data/test.csv", index=False)
