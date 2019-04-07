import os
import numpy
#uses keras which isnt in venv by default :)
import tensorflow as tf
from keras import backend
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rnn_network(learning_rate, epochs, batches, seed):
    # fix random seed for reproducibility
    numpy.random.seed(seed)
    tf.set_random_seed(seed)

    # load the dataset but only keep the top n words, zero the rest
    top_words = 10000 #defaulted to 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100, dropout=0.0, recurrent_dropout=0.0))
    model.add(Dense(1, activation='sigmoid'))

    from keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=learning_rate),  # 0.001
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    print("'\r{0}".format(model.summary()), end='')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batches)    #epochs 3, batch_size 64

    combination = 0

    saver = tf.train.Saver()
    save_string = "imdb-rnn-" + str(combination) + "-" + str(learning_rate) + "-" + str(epochs) + "-" + str(
        batches) + "-" + str(seed)
    sess = backend.get_session()
    saver.save(sess, 'tmp/' + save_string + '/' + save_string + '.ckpt')

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def main(learning_rate, epochs, batches, seed):
    rnn_network(learning_rate, epochs, batches, seed)


if __name__ == "__main__":
    rnn_network(0.01, 1, 64, 420)
