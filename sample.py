#!/usr/bin/env python3

"""
Load saved model from disk, and sample data from it.
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

import random

from train import (
    load_data,
    build_model,
    load_latest_model,
    sample_from_model,
    MAXLEN
)

if __name__ == '__main__':
    text, char_indices, indices_char = load_data()
    model = build_model(indices_char)
    load_latest_model(model)

    for _ in range(10):
        start_index = random.randint(0, len(text) - MAXLEN - 1)
        sample_from_model(text, start_index, char_indices, model, indices_char)
