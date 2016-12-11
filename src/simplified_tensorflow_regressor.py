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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import time
'''
Links and sources:
http://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks
'''
print("Running Regressor!")
#data_file = "./Pickles/train_mat_filtered_big.pkl"
data_file = "./train_mat_filtered.pkl"
RUN_BIDIRECTIONAL = True
print("Running Bidirectional LSTM: ", RUN_BIDIRECTIONAL)
USE_UNIFORM_DISTRIBUTION = False
print("Using uniform label distribution ", USE_UNIFORM_DISTRIBUTION)
MAX_SAMPLES = 65323 # will be changed once the data is loaded
MAX_SAMPLES_TO_USE = 20000
TEST_SET_PERCENTAGE = 0.1

BATCH_SIZE = 64
DISPLAY_STEP = 10

# Neural net parameters
N_HIDDEN = 64 # hidden layer dimension
N_CLASSES = 1 # number of labels
N_INPUT = 300 # dimension of word2vec embedding
LEARNING_RATE = 0.01
TRAINING_ITERS = 30000

NUM_LABELS = 5
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
train_set_ind,test_set_ind = get_random_samples(labels, MAX_SAMPLES_TO_USE, TEST_SET_PERCENTAGE,USE_UNIFORM_DISTRIBUTION)
train_data, train_labels = data[train_set_ind],labels[train_set_ind]
test_data, test_labels = data[test_set_ind],labels[test_set_ind]


trainset = Dataset(train_data, train_labels,embedding_matrix, BATCH_SIZE, max_seq_len)
testset = Dataset(test_data, test_labels, embedding_matrix, BATCH_SIZE, max_seq_len)
print("Train set size ", len(trainset.data))
print("Test set size ", len(testset.data))

# ==========
#   MODEL
# ==========
# # tf Graph input
x = tf.placeholder("float", [None, max_seq_len, N_INPUT],name="data_input")
y = tf.placeholder("float", [None, N_CLASSES],name="labels_input")
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

def dynamicRNN(x, seqlen,weights, biases):

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
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)
   
        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]

        # Start indices for each sample
        index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, N_HIDDEN]), index)
        output_node = tf.matmul(outputs, weights['out']) + biases['out']
        # Use sigmoid activation function
        sigmoid_node = tf.sigmoid(output_node)
        return sigmoid_node

    else:

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.BasicLSTMCell(N_HIDDEN, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn_cell.BasicLSTMCell(N_HIDDEN, forget_bias=1.0)

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
        output_node = tf.matmul(outputs, weights['out']) + biases['out']
        # Use sigmoid activation function
        sigmoid_node = tf.sigmoid(output_node)
        return sigmoid_node



pred = dynamicRNN(x, seqlen, weights, biases) # values in [0, 1]
scaled_pred = tf.scalar_mul(NUM_LABELS,pred) # scale the data to be in [0,5]
rounded_pred = tf.ceil(scaled_pred) # round each prediction to the nearest integer

cost = tf.nn.l2_loss(scaled_pred-y, name="squared_error_cost")
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

# Evaluate model
correct_pred = tf.equal(rounded_pred, y)#tensor of true or false
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    start = time.time()
    step = 1
    # Keep training until reach max iterations
    while step * BATCH_SIZE < TRAINING_ITERS:
        batch_x, batch_y, batch_seqlen = trainset.next()
        batch_y = np.array([[i] for i in batch_y])
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
    test_label = np.array([[i] for i in testset.labels])
    test_seqlen = testset.seq_len
    
    print("Evaluation")
    # gets indices, not stars!
    predictions = sess.run(rounded_pred,feed_dict={x: test_data,seqlen: test_seqlen})
    precision = precision_score(test_label, predictions, pos_label=None, average="weighted")  
    recall = recall_score(test_label, predictions, pos_label=None,
                                 average="weighted")  
    f1 = f1_score(test_label, predictions, pos_label=None,
                                average="weighted")
    accuracy = accuracy_score(test_label, predictions)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    