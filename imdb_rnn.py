import os
import numpy
#uses keras which isnt in venv by default :)
import tensorflow as tf
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rnn_network(learning_rate, epochs, batches, seed):
    # fix random seed for reproducibility
    numpy.random.seed(seed)
    tf.set_random_seed(seed)

    # load the dataset but only keep the top n words, zero the rest
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    from keras.optimizers import Adam
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy']) #lr0.001

    print("'\r{0}".format(model.summary()), end='')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batches)    #epochs 3, batch_size 64

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def main(learning_rate, epochs, batches, seed):
    rnn_network(0.001, 3, 64, 420)


if __name__ == "__main__":
    main(0.001, 3, 64, 420)