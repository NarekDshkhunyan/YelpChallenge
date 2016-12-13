import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import cPickle
import random
from keras.utils import np_utils
from collections import Counter

'''
Sources:
LSTM:
http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
Use existing word2vec embedding:
https://github.com/fchollet/keras/issues/853
How to make a multiclass classifier:
http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
'''
data_file = "./Pickles/train_mat_filtered_big.pkl"
vocab_file = "./Pickles/vocab_filtered_big.pkl"
vocab_inv_file = "./Pickles/vocab_inv_filtered_big.pkl"

# PARAMETERS
LSTM_MEM_UNITS = 100
BATCH_SIZE = 64
NB_EPOCH = 3
DROPOUT_VAL = 0.5

MAX_SAMPLES = 1500000  # maximum number of available samples, do not change
MAX_SAMPLES_TO_USE = 20000  # can be changed
TEST_SET_PERCENTAGE = 0.01
MAX_SAMPLES_PER_RATING = 110000 # can be set to up to 4002, after which default value is 4002
UNIFORMIZE_DATA = True

WORD2VEC_DIM = 300

def print_params():
    print "LSTM_MEM_UNITS = ", LSTM_MEM_UNITS
    print "BATCH_SIZE = ", BATCH_SIZE
    print "NB_EPOCH = ", NB_EPOCH
    print "DROPOUT_VAL = ", DROPOUT_VAL
    print "UNIFORMIZE_DATA = ", UNIFORMIZE_DATA
def get_vocabs():
    with open(vocab_file) as f:
        vocabulary = cPickle.load(f)
    for key in vocabulary:
        vocabulary[key] += 1  # index 0 is reserved for masking

    with open(vocab_inv_file) as f:
        vocabulary_inv = cPickle.load(f)
    vocabulary_inv = ['IGNORE'] + vocabulary_inv  # index 0 is reserved for masking
    return (vocabulary, vocabulary_inv)


def load_data():
    with open(data_file) as f:
        x, y, embedding_matrix = cPickle.load(f)
    zero_row = np.random.uniform(-.25, .25, 300)  # to match 0th index
    embedding_matrix = np.concatenate((np.array([zero_row]), embedding_matrix))  # index 0 is reserved for masking
    # embedding_matrix = np.concatenate((embedding_matrix, np.zeros((1,300))))
    for row in xrange(len(x)):
        for col in xrange(len(x[row])):
            x[row][col] += 1  # increment stuff by 1 to match the vocab
    MAX_SAMPLES = len(x)
    return (x, y, embedding_matrix)


def get_random_samples(labels, num_samples,test_set_percentage,uniform=False):
    if not uniform:
        indicies = random.sample(xrange(MAX_SAMPLES),num_samples)
        train_set_size = int(num_samples*(1-test_set_percentage))
        return indicies[:train_set_size], indicies[train_set_size:]
    else:
        counts = np.bincount(labels)[1:] # omit count of 0th rating
        group_size = int(num_samples/5.)
        max_available = min(min(counts), group_size)
        ratings_arr = [np.where(labels == i) for i in xrange(1, 6)]
        train_set_ind = []
        test_set_ind = []
        for s in ratings_arr:
            chosen = random.sample(xrange(len(s[0])), max_available)
            chosen_idx = [int(s[0][i]) for i in chosen]
            train_set_num = int(len(chosen_idx)*(1-test_set_percentage))
            train_set_ind.extend(chosen_idx[0:train_set_num])
            test_set_ind.extend(chosen_idx[train_set_num:])
        np.random.shuffle(train_set_ind)
        np.random.shuffle(test_set_ind)
        return train_set_ind, test_set_ind


def main():
    vocabulary, vocabulary_inv = get_vocabs()
    data, labels, embedding_matrx = load_data()
    assert (MAX_SAMPLES_TO_USE <= MAX_SAMPLES)
    max_review_length = max([len(i) for i in data])
    train_set_ind, test_set_ind = get_random_samples(labels, MAX_SAMPLES_TO_USE, TEST_SET_PERCENTAGE, UNIFORMIZE_DATA)
    data = np.array(data)
    print len(train_set_ind), " samples ready."
    print "Train label distribution: ", Counter(np.array(labels)[train_set_ind])
    print "Test label distribution: ", Counter(np.array(labels)[test_set_ind])

    print_params()
    max_review_length = max([len(i) for i in data])
    vocab_size = len(vocabulary)
    num_labels = 5
    encoder = LabelEncoder()
    encoder.fit(labels)
    
    X_train, X_test = data[train_set_ind], data[test_set_ind]
    y_train, y_test = np.array(labels[train_set_ind]), np.array(labels[test_set_ind])
    del data
    del labels
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    y_train = encoder.transform(y_train)
    # convert integers to dummy variables (i.e. one hot encoded)
    y_train = np_utils.to_categorical(y_train)

    y_test = encoder.transform(y_test)
    y_test = np_utils.to_categorical(y_test)

    model = Sequential()
    # use existing embedding weights
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=WORD2VEC_DIM, mask_zero=True, weights=[embedding_matrx],
                            input_length=X_train.shape[1]))
    model.add(LSTM(LSTM_MEM_UNITS))
    model.add(Dropout(DROPOUT_VAL))
    model.add(Dense(num_labels, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=NB_EPOCH, batch_size=BATCH_SIZE)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    print(model.metrics_names)


if __name__ == "__main__":
    main()
