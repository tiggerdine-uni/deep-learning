from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

import numpy as np

from helpers import make_tmp


def mlp_network(combination, learning_rate, epochs, batches, seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)

    make_tmp()

    imdb = keras.datasets.imdb

    # The argument num_words=10000 keeps the top 10,000 most frequently occurring words in the training data.
    # The rare words are discarded to keep the size of the data manageable.
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # A dictionary mapping words to an integer index
    word_index = imdb.get_word_index()

    # The first indices are reserved
    word_index = {k:(v+3) for k,v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                            value=word_index["<PAD>"],
                                                            padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                           value=word_index["<PAD>"],
                                                           padding='post',
                                                           maxlen=256)

    # input shape is the vocabulary count used for the movie reviews (10,000 words)
    vocab_size = 10000

    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocab_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    # model.summary()

    model.compile(optimizer=keras.optimizers.Adam(lr=learning_rate),  # 0.001
                  loss='binary_crossentropy',
                  metrics=['acc'])

    from datetime import datetime
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    save_string = "imdb-mlp-" + str(combination) + "-" + str(learning_rate) + "-" + str(epochs) + "-" + str(batches) + "-" + str(seed)
    root_logdir = "logs"
    logdir = "{}/{}-{}".format(root_logdir, save_string, now)
    tensorboard = keras.callbacks.TensorBoard(log_dir=logdir)

    x_val = train_data[:10000]
    partial_x_train = train_data[10000:]

    y_val = train_labels[:10000]
    partial_y_train = train_labels[10000:]

    history = model.fit(partial_x_train,
              partial_y_train,
              epochs=epochs,  # 40
              batch_size=batches,  # 512
              validation_data=(x_val, y_val),
              callbacks=[tensorboard],
              verbose=1)

    model.save('tmp/' + save_string + '.ckpt')

    results = model.evaluate(test_data, test_labels)

    train = round(history.history['acc'][-1], 4)
    test = round(results[1], 4)

    print('train: ', train, ' - test: ', test)

    return train, test


def doMagic():
    learning_rates = [0.00025, 0.0005, 0.001, 0.002, 0.004]
    epochs = [10, 20, 40, 80, 160]
    batchess = [128, 256, 512, 1024, 2048]
    if os.path.exists('tmp/magic.csv'):
        os.remove('tmp/magic.csv')
    f = open('tmp/magic.csv', 'w')
    f.write('learning_rate,epoch,batches,train,test\n')
    for learning_rate in learning_rates:
        for epoch in epochs:
            for batches in batchess:
                train, test = mlp_network(0, learning_rate, epoch, batches, 420)
                f.write(str(learning_rate) + ',' +
                        str(epoch) + ',' +
                        str(batches) + ',' +
                        str(train) + ',' +
                        str(test) + '\n')
    f.close()


if __name__ == "__main__":
    # mlp_network(0, 0.00075, 40, 512, 420)
    doMagic()













































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