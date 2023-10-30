"""Another attempt to cut down the time for preprocessing the dataframe.

TODO: Ask if we can use the Inference API for non-core tasks like summarization!
"""
import httpx
import pickle
import time
import sys
from cleantext import clean
from functools import partial
from tqdm import tqdm

import pandas as pd

HF_TOKEN_VALUE = "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"

AUTHORIZATION_HEADER = {
    "Authorization": f"Bearer {HF_TOKEN_VALUE}",
}

INFERENCE_ENDPOINT = "https://api-inference.huggingface.co/models/Abdulkader/autotrain-medical-reports-summarizer-2484176581"


def get_training_df(path: str = "data/training.pkl"):
    """Get the training DF and process it."""
    with open(path, "rb") as f:
        training_df = pickle.load(f)

    return training_df


def get_processed_df(
    dataframe: pd.DataFrame, inference_endpoint: str = INFERENCE_ENDPOINT
) -> pd.DataFrame:
    first_pass_cleaner = partial(clean, no_line_breaks=True)
    print("starting first pass of cleaning...")
    start = time.perf_counter()
    dataframe["first_pass_processing"] = dataframe.text.apply(first_pass_cleaner)
    end = time.perf_counter()
    print(f"Done first pass clean after {end - start:3f} seconds")
    dataframe["api_summary"] = "N/A"
    for index in tqdm(dataframe.index):
        # Inputs
        input_dict = {
            "inputs": dataframe.loc[index, "first_pass_processing"],
            "wait_for_model": True,
        }
        response = httpx.post(
            inference_endpoint,
            headers=AUTHORIZATION_HEADER,
            json=input_dict,
            timeout=None,
        )
        if response.status_code == 200:
            response_dict = response.json()
            dataframe.loc[index, "api_summary"] = response_dict[0]["summary_text"]
            time.sleep(1)
        else:
            print(
                f"Got {response.status_code}, the model is probably warming up - wait 30 seconds before running again"
            )
            sys.exit(1)

    return dataframe


if __name__ == "__main__":
    dataframe = get_training_df()
    processed = get_processed_df(dataframe)
    processed.to_csv("api_summarized.csv")
