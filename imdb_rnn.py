import os
import numpy
#uses keras which isnt in venv by default :)
import tensorflow as tf
from keras import backend
from keras.callbacks import TensorBoard
from keras.datasets import imdb
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# from tensorflow import keras
from helpers import make_tmp

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def rnn_network(combination, learning_rate, epochs, batches, seed):
    # fix random seed for reproducibility
    numpy.random.seed(seed)
    tf.set_random_seed(seed)

    make_tmp()

    # load the dataset but only keep the top n words, zero the rest
    top_words = 10000  # 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # truncate and pad input sequences
    max_review_length = 500
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100,
                   dropout=0.2,  # 0.2
                   recurrent_dropout=0.2))  # 0.2
    model.add(Dense(1, activation='sigmoid'))

    from keras.optimizers import Adam
    model.compile(optimizer=Adam(lr=learning_rate),  # 0.001
                  loss='binary_crossentropy',
                  # loss='mean_squared_error',
                  metrics=['accuracy'])

    from datetime import datetime
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    save_string = "imdb-rnn-" + str(combination) + "-" + str(learning_rate) + "-" + str(epochs) + "-" + str(
        batches) + "-" + str(seed)
    root_logdir = "tf_logs"
    logdir = "{}/{}-{}".format(root_logdir, save_string, now)
    tensorboard = TensorBoard(log_dir=logdir)

    print("'\r{0}".format(model.summary()), end='')
    model.fit(X_train,
              y_train,
              epochs=epochs,  # 3
              batch_size=batches,  # 64
              callbacks=[tensorboard])

    model.save('tmp/' + save_string + '.ckpt')
    #new_model = load_model('tmp/imdb-rnn-0-0.005-0-64-420.ckpt')

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    #scores = new_model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == "__main__":
    rnn_network(0, 0.005, 0, 64, 420)
