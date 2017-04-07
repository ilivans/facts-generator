#! /usr/bin/env python
from __future__ import print_function
import sys
import random

import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras.optimizers import Adam

data_path = "data/total6.txt"
text = open(data_path).read().lower()
print("Corpus length:", len(text))

chars = sorted(list(set(text)))
print("Total chars:", len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Cut the text in semi-redundant sequences of maxlen characters
seq_len = 40
sentences = []
next_chars = []
for i in range(0, len(text) - seq_len):
    sentences.append(text[i: i + seq_len])
    next_chars.append(text[i + seq_len])
print("nb sequences:", len(sentences))

print("Vectorization...")
X = np.zeros((len(sentences), seq_len, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# Build the model: 2-layer GRU and 2 dense layers
print("Build model...")
model = Sequential()
hidden_size = 256
# model.add(GRU(128, input_shape=(maxlen, len(chars))))  # single RNN
model.add(GRU(hidden_size, input_shape=(seq_len, len(chars)), return_sequences=True))
model.add(GRU(hidden_size, input_shape=(seq_len, hidden_size)))
model.add(Dense(100))
model.add(Dense(len(chars), activation="softmax"))

optimizer = Adam(lr=0.0002)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds)
    return np.argmax(probas)

# Train the model, output generated text after each iteration
for iteration in range(1, 60):
    print()
    print("-" * 50)
    model.fit(X, y,
              batch_size=128,
              epochs=iteration, initial_epoch=iteration - 1)

    start_index = random.randint(0, len(text) - seq_len - 1)

    for diversity in [0.2, 0.5, 1.0]:
        print()
        print("----- diversity:", diversity)

        generated = ""
        sentence = text[start_index: start_index + seq_len]
        generated += sentence
        print("----- Generating with seed: \"" + sentence + "\"")
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, seq_len, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
