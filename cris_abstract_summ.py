#!/usr/bin/env python3

"""
The following code is an attempt to pre-process the clinical notes before training.
Before running, you should change the HuggingFace Token as well as the training.pkl file location
2023-10-30

"""

# Import dependencies
import os
import re
import pprint
import string

import numpy as np
import pandas as pd
import nltk

import torch
import transformers

import locale; locale.getpreferredencoding = lambda: "UTF-8"

# Get punkt and stopwords
nltk.download("punkt")
nltk.download('stopwords')

# Assert Huggingface
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"
assert os.environ["HUGGING_FACE_HUB_TOKEN"]


# Rico's final text processing function
stopwords = set(nltk.corpus.stopwords.words("english")) # cast to set for O(1) membership check
PUNCT_TRANS_TABLE = str.maketrans(dict.fromkeys(string.punctuation)) # unicode ordinals for punctuation -> None

def get_processed_text(text):
    # Get rid of newlines and double-returns while maintaining spacing
    no_newlines = text.replace("\n\n", " ").replace("\n", " ")
    # lowercase so that we match with nltk stopwords
    no_stopwords = [word for word in nltk.word_tokenize(no_newlines.lower()) if word not in stopwords]
    # finally, kill all the punctuation
    processed = " ".join(no_stopwords).translate(PUNCT_TRANS_TABLE)
    return processed

# Load in data
print("Loading in data from Cris's file location...\n")
df = pd.read_pickle("project2_train.pkl") # Cris's training data location
df = df.fillna("")
total_rows = len(df)
split_0_count = int(total_rows * 0.9)
split_1_count = total_rows - split_0_count
#Create an array with split values based on the counts
split_values = np.concatenate([
    np.zeros(split_0_count),
    np.ones(split_1_count),
])

# Shuffle the array to ensure randomness
np.random.shuffle(split_values)

# Add the 'split' column to the DataFrame
df['split'] = split_values
df['split'] = df['split'].astype(int)

PROMPT = "Using the following medical notes, predict if the patient described has Acute Respiratory Distress Syndrome (ARDS). Predict 'true' if ARDS is likely, or 'false' if it is not."
df['instruction'] = PROMPT
df = df.rename(columns={'text':'notes', 'label':'output'})
df = df.astype({'output':'str'})

#Pre-process text for input
print("Pre-processing notes...\n")
# Use regex to remove '___', '()', and ':'
df['processed_text'] = df['notes'].str.replace(r'[:___/,()*?[0-9]]', '', regex=True)
# Remove subsection headers
remove = ['Note', 'IMPRESSION', 'FINDINGS', 'EXAMINATION', 'INDICATION',
          'TECHNIQUE','COMPARISON','NOTIFICATION', 'RECOMMENDATION'
          ]
for x in remove:
  df['processed_text'] = df['processed_text'].str.replace(x, '',regex=False)

# Perform final text processing for input
df['processed_text'] = df['processed_text'].apply(get_processed_text)

# Check if input length is longer than 10K characters - if it is use abstractive summarization to generate input

# Instantiate model and tokenizer
DEVICE = torch.device("cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("Falconsai/medical_summarization")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    "Falconsai/medical_summarization"
).to(DEVICE)

# Abstract summarizer function
def the_liver_is_the_cocks_comb(text: str):
   # Tokenize
   tokens = tokenizer.encode(
        "summarize: " + text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=4096
    ).to(DEVICE)

   # Generate text
   with torch.no_grad():
        model_output = model.generate(
            tokens,
            num_beams=4,
            no_repeat_ngram_size=2,
            min_length=20,
            max_length=3000,
            early_stopping=True,
        )

   output = tokenizer.decode(model_output[0], skip_special_tokens=True)

   return output

# Check if input character length is greater than 10k
print("Checking token length of pre-processed notes...\n")
# Initialize Longformer tokenizer in order to enforce the training model's token limit
lf_name = "yikuan8/Clinical-Longformer"
lf_tokenizer = transformers.AutoTokenizer.from_pretrained(lf_name)
toke_check = transformers.PreTrainedTokeizerBase = lf_tokenizer
df['input'] = ""
df['token_length'] = ""
limit = int(4096)
print("Abstractively summarizing notes that exceed training model's token limit...\n")
for i in range(len(df['processed_text'])):
    limit = int(4096)
    input_txt = str(df['processed_text'][i])
    toke = toke_check.encode(input_txt)
    toke_count = len(toke)
    # Add token count to data frame
    df.loc[i,'token_length'] = toke_count
    # Check if input text exceeds Longformers token limit. If it does, abstractively summarize
    if toke_count >= limit:
        df.loc[i,'input'] = the_liver_is_the_cocks_comb(input_txt)
        index = i
    else:
        df.loc[i,'input'] = str(df['processed_text'][i])

print(df[df.token_length >= 4096].count())
df.to_csv("cris_preprocess_attempt.csv")
print("Done.\n")
