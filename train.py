#!/usr/bin/env python3

"""
Train on data and save model status to disk
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division


from keras.models import Sequential
from keras.layers.core import (
    Dense,
    Activation,
    Dropout
)
from collections import Counter
from keras.layers.recurrent import LSTM
import codecs
import numpy as np

N_ITER = 60
MAXLEN = 20
BATCH_SIZE = 128

FNAME = 'scrape/4920.txt'

VERSION = 1

def text_crop_at_percentile(text, perc=75):
    """
    Remove all chars from a text which occur within a percentile of less
    than ``perc``.
    """
    c = Counter(text)
    importance = sorted([(a,b) for a, b in c.items()], key=lambda x: x[1])
    i = int((1-(perc/100.)) * len(importance))
    delchars = [x for x, _ in importance[:i]]
    keepchars = [x for x, _ in importance[i:]]
    print('  Keep chars: "%s"\n  Delete chars: ""%s"' % (
        ''.join(reversed(keepchars)).replace('\n', ''),
        ''.join(delchars)
    ))
    return text.translate({ord(x): None for x in delchars})


def load_data():
    with codecs.open(FNAME, 'r', 'utf-8') as f:
        text = f.read().lower()

    print('corpus length:', len(text))
    text = text_crop_at_percentile(text, 50)
    print('Reduced corpus length:', len(text))

    chars = set(text)
    print('total chars:', len(chars))
    indices_char = list(chars)
    char_indices = {c: i for i, c in enumerate(indices_char)}

    return text, char_indices, indices_char

def vectorize(text, char_indices):
    """cut the text in semi-redundant sequences of MAXLEN characters"""
    print('Vectorization...')
    step = 3

    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, step):
        sentences.append(text[i:i + MAXLEN])
        next_chars.append(text[i + MAXLEN])
    print('nb sequences:', len(sentences))

    X = np.zeros((len(sentences), MAXLEN, len(indices_char)), dtype=np.bool)
    y = np.zeros((len(sentences), len(indices_char)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y

def build_model(indices_char):
    # build the model: 2 stacked LSTM
    print('Build and compile model...')
    model = Sequential()
    model.add(LSTM(len(indices_char), 512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(512, 512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(512, len(indices_char)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model

if __name__ == '__main__':
    text, char_indices, indices_char = load_data()
    X, y = vectorize(text, char_indices)
    model = build_model(indices_char)

    # train the model, output generated text after each iteration
    for iteration in range(1, N_ITER):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=1)
