"""Defines utilities for the tasks."""

import numpy as np
from transformers import T5Tokenizer


def round_stsb_target(label):
    """STSB maps two sentences to a floating point number between 1 and 5
    representing their semantic similarity. Since we are treating all tasks as
    text-to-text tasks we need to convert this floating point number to a string.
    The vast majority of the similarity score labels in STSB are in the set
    [0, 0.2, 0.4, ..., 4.8, 5.0]. So, we first round the number to the closest
    entry in this set, and then we convert the result to a string (literally e.g.
    "3.4"). This converts STSB roughly into a 26-class classification dataset.
    Args:
      label: original label.
    Returns:
      A preprocessed label.
    """
    return np.round((label * 5) / 5, decimals=1)


def compute_task_max_decoding_length(word_list):
    """Computes the max decoding length for the given list of words
    Args:
      word_list: A list of stringss.
    Returns:
      maximum length after tokenization of the inputs.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    max_len = 0
    for word in word_list:
        ids = tokenizer.encode(word)
        max_len = max(max_len, len(ids))
    return max_len
