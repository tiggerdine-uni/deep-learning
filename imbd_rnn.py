from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras

import numpy as np

imdb = keras.datasets.imdb

#The argument num_words=10000 keeps the top 10,000 most frequently occurring words in the training data.
#The rare words are discarded to keep the size of the data manageable.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

#decode_review(train_data[0])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=500)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=500)

#TODO ^ Above here is common (maybe)

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000
max_words = 500
embedding_size = 32

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, embedding_size, input_length=max_words))
model.add(keras.layers.LSTM(100))
model.add(keras.layers.Dense(1, activation='sigmoid'))

#model.summary()

#Make a new optimizer
sgd = keras.optimizers.SGD(lr=0.4, decay=1e-6, momentum=0.9, nesterov=True)

#TODO Replace adam with sgd
#adam please help us
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

batch_size = 64
num_epochs = 3
X_valid, y_valid = train_data[:batch_size], train_labels[:batch_size]
X_train2, y_train2 = train_data[batch_size:], train_labels[batch_size:]
model.fit(X_train2, y_train2, validation_data=(X_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

results = model.evaluate(test_data, test_labels)

print(results)














































#LICENCE OF TUTORIAL CODE USED:
#
#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.