#!/usr/bin/env python3

"""
Load saved model from disk, and sample data from it.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import random
import sys

from train import (
    load_data,
    build_model,
    load_latest_model,
    sample_prob,
    MAXLEN
)

DIVERSITY = 0.75

def sample_from_model(text, start_index, char_indices, model, indices_char):
    sentence = text[start_index:start_index + MAXLEN]
    for c in sentence:
        yield c

    while True:
        x = np.zeros((1, MAXLEN, len(indices_char)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample_prob(preds, DIVERSITY)
        next_char = indices_char[next_index]
        sentence = sentence[1:] + next_char
        yield next_char


def yield_chars():
    text, char_indices, indices_char = load_data()
    model = build_model(indices_char)
    load_latest_model(model)

    start_index = random.randint(0, len(text) - MAXLEN - 1)
    gen = sample_from_model(text, start_index, char_indices, model, indices_char)

    next_letter_upper = False
    for c in gen:
        if next_letter_upper and c.upper() != c:
            yield c.upper()
            next_letter_upper = False
        else:
            yield c
        if c == '.':
            next_letter_upper = True

if __name__ == '__main__':
    for c in yield_chars():
        sys.stdout.write(c)
        sys.stdout.flush()
