from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import random
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import cPickle
import numpy as np
from dataset import Dataset
import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
data_file = "./Pickles/train_mat_filtered_big.pkl"

RUN_BIDIRECTIONAL = False
print("Running Bidirectional LSTM: ", RUN_BIDIRECTIONAL)
USE_UNIFORM_DISTRIBUTION = False
print("Using uniform label distribution ", USE_UNIFORM_DISTRIBUTION)
MAX_SAMPLES = 65323 # will be changed once the data is loaded
MAX_SAMPLES_TO_USE = 20000 # train set + test set
TEST_SET_PERCENTAGE = 0.1

BATCH_SIZE = 32
DISPLAY_STEP = 10

# Neural net parameters
N_HIDDEN = 64 # hidden layer dimension
N_CLASSES = 5 # number of labels
N_INPUT = 300 # dimension of word2vec embedding
LEARNING_RATE = 0.01
TRAINING_ITERS = 1000


def load_data():
    with open(data_file) as f:
        x,y, embedding_matrix = cPickle.load(f)
    MAX_SAMPLES = len(x)
    return (x,y, embedding_matrix)



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



# ==========
#   DATA
# ==========

data,labels,embedding_matrix = load_data()

data = np.array(data)
max_seq_len = max([len(review) for review in data])
print("Loaded ", len(data)," samples.")
encoder = LabelEncoder()
encoder.fit(labels)


train_set_ind,test_set_ind = get_random_samples(labels, MAX_SAMPLES_TO_USE, TEST_SET_PERCENTAGE,USE_UNIFORM_DISTRIBUTION)
train_data, train_labels = data[train_set_ind],labels[train_set_ind]
test_data, test_labels = data[test_set_ind],labels[test_set_ind]

train_labels = encoder.transform(train_labels)
train_labels = np_utils.to_categorical(train_labels)

test_labels = encoder.transform(test_labels)
test_labels = np_utils.to_categorical(test_labels)

trainset = Dataset(train_data, train_labels,embedding_matrix, BATCH_SIZE, max_seq_len)
testset = Dataset(test_data, test_labels, embedding_matrix, BATCH_SIZE, max_seq_len)

print("Train set size ", len(trainset.data))
print("Test set size ", len(testset.data))

# ==========
#   MODEL
# ==========
# # tf Graph input
x = tf.placeholder("float", [None, max_seq_len, N_INPUT])
y = tf.placeholder("float", [None, N_CLASSES])
# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])
# Define weights
if RUN_BIDIRECTIONAL:
    weights = {
        'out': tf.Variable(tf.random_normal([2*N_HIDDEN, N_CLASSES]))
    }
else:
    weights = {
        'out': tf.Variable(tf.random_normal([N_HIDDEN, N_CLASSES]))
    }   
biases = {
    'out': tf.Variable(tf.random_normal([N_CLASSES]))
}

def dynamicRNN(x, seqlen, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, N_INPUT])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, max_seq_len, x)
    if not RUN_BIDIRECTIONAL:
        lstm_cell = rnn_cell.BasicLSTMCell(N_HIDDEN, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)

        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, N_HIDDEN]), index)


        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

    else:

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.BasicLSTMCell(N_HIDDEN, forget_bias=0.8)
        # Backward direction cell
        lstm_bw_cell = rnn_cell.BasicLSTMCell(N_HIDDEN, forget_bias=0.7)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32,sequence_length=seqlen)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

        outputs = tf.pack(outputs[0])
        outputs = tf.transpose(outputs, [1, 0, 2])

        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, 2*N_HIDDEN]), index)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs, weights['out']) + biases['out']



pred = dynamicRNN(x, seqlen, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
#init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    start = time.time()
    step = 1
    # Keep training until reach max iterations
    while step * BATCH_SIZE < TRAINING_ITERS:
        batch_x, batch_y, batch_seqlen = trainset.next()
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
        if step % DISPLAY_STEP == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
            print("Iter " + str(step*BATCH_SIZE) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1

    print("Optimization Finished!")
    print("Training network with batch size = ", BATCH_SIZE, " and number of iterations = ", TRAINING_ITERS)
    print("Total time: ", time.time()-start)
    # Calculate accuracy
    test_data = testset.pad_all()
    test_label = testset.labels
    test_seqlen = testset.seq_len
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))
    print("Evaluation")
    # gets indices, not stars!
    predictions = sess.run(tf.argmax(pred,1),feed_dict={x: test_data,seqlen: test_seqlen})
    gold = np.argmax(test_label, 1) #same here, indices
    precision = precision_score(gold, predictions, pos_label=None, average="weighted")  
    recall = recall_score(gold, predictions, pos_label=None,
                                 average="weighted")  
    f1 = f1_score(gold, predictions, pos_label=None,
                                average="weighted")
    accuracy = accuracy_score(gold, predictions)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

