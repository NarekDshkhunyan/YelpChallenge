from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import random
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import cPickle
import numpy as np

#data_dir = "/Users/danamukusheva/Desktop/6.864/project/6.864_project/src"
data_dir = "./"
data_file = "train_mat_filtered.pkl"
RUN_BIDIRECTIONAL = True

MAX_SAMPLES = 65323 # maximum number of available samples, do not change
TRAIN_SET_SIZE = 30000 # these do not matter if using get_random_samples_strictly_uniform
TEST_SET_SIZE = 3000

BATCH_SIZE = 32
DISPLAY_STEP = 100
N_HIDDEN = 64 # hidden layer dimension
N_CLASSES = 5 # number of labels
N_INPUT = 300 # dimension of word2vec embedding
# Parameters
LEARNING_RATE = 0.01
TRAINING_ITERS = 50000
'''
# 1 epoch = (num_samples/batch_size) iterations
# one epoch = one forward pass and one backward pass of all the training examples
# batch size = the number of training examples in one forward/backward pass
# number of iterations = number of passes, each pass using [batch size] number 
# of examples. To be clear, one pass = one forward pass + one backward pass 
# (we do not count the forward pass and backward pass as two different passes).
#Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.
'''


def test_distriution(labels):
    c = Counter(labels)
    print(c)
def load_data():
    with open(data_dir+"/"+data_file) as f:
        x,y, embedding_matrix = cPickle.load(f)
    return (x,y, embedding_matrix)


def transform_data(data, embedding_matrix):
    '''
    make 3-D array out of 2-D array by substituting word indices 
    with 300-D vectors
    '''
    new_data = []
    for review in data:
        num_words = len(review)
        new_review = np.ndarray((num_words,N_INPUT))
        for i in xrange(num_words):
            word_ind = review[i]
            new_review[i] = embedding_matrix[word_ind]
        new_data.append(new_review)
    
    new_data =  np.array(new_data)
    return new_data


def get_random_samples(train_set_size, test_set_size):
    indicies = random.sample(xrange(MAX_SAMPLES),train_set_size+test_set_size)
    return indicies[:train_set_size], indicies[train_set_size:]

def get_random_samples_strictly_uniform(labels, test_set_percentage):
    counts = np.bincount(labels)[1:] # omit count of 0th rating
    max_available = min(min(counts), 4000)
    #print("Max representatives for each label: ", max_available)
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
    #print('Total data points: ', len(train_set_ind)+len(test_set_ind))
    return train_set_ind, test_set_ind


class Dataset(object):
    """ Generate sequence of data with dynamic length.
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """
    def __init__(self, data, labels, embedding_matrix, batch_size,max_seq_len):
        self.seq_len = [len(review) for review in data]
        self.data = transform_data(data,embedding_matrix) #3D array, (num_samples, num_words_in_sample (varies), 300)
        self.labels = labels #2D array (num_samples, 5)#binarized lables
        self.batch_size = batch_size
        self.batch_id = 0
        self.max_seq_len = max_seq_len

    def next(self):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0 #reset

        batch_data = (self.data[self.batch_id:min(self.batch_id +self.batch_size, len(self.data))])
        
        #start padding
        rand_word  = np.random.uniform(-.25, .25, 300)
        batch_data = list(batch_data)                                         
        for i in xrange(len(batch_data)):
            dif = self.max_seq_len - len(batch_data[i])
            for j in xrange(dif):
                batch_data[i] = list(batch_data[i])
                batch_data[i].append(rand_word)


        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                  self.batch_size, len(self.data))])
        batch_seqlen = (self.seq_len[self.batch_id:min(self.batch_id +
                                                  self.batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + self.batch_size, len(self.data))
        return batch_data, list(batch_labels), batch_seqlen

    def pad_all(self):

        #start padding
        rand_word = np.random.uniform(-.25, .25, 300)
        data = list(self.data)                                         
        for i in xrange(len(data)):
            dif = self.max_seq_len - len(data[i])
            for j in xrange(dif):
                data[i] = list(data[i])
                data[i].append(rand_word)
        return data

# ==========
#   DATA
# ==========

data,labels,embedding_matrix = load_data()
data = np.array(data)
max_seq_len = max([len(review) for review in data])
print("Loaded ", len(data)," samples.")
encoder = LabelEncoder()
encoder.fit(labels)


train_set_ind,test_set_ind = get_random_samples(TRAIN_SET_SIZE, TEST_SET_SIZE)
#train_set_ind,test_set_ind =get_random_samples_strictly_uniform(labels,0.1)
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

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(N_HIDDEN, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32,sequence_length=seqlen)

        print("Before transpose")
        print(len(outputs))
        print(outputs[0].get_shape().as_list())
        print(outputs[10].get_shape().as_list()), '\n'

        outputs = tf.pack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])

        print("After transpose")
        print(outputs.get_shape().as_list())
        print(tf.shape(outputs))
        print(outputs[0].get_shape().as_list())
        print(outputs[10].get_shape().as_list()), '\n'

        #print(len(outputs))

        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        print(batch_size)
        # Start indices for each sample
        index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, N_HIDDEN]), index)

        print("After reshape")
        print(outputs.get_shape().as_list())
        print(outputs[0].get_shape().as_list())
        print(outputs[10].get_shape().as_list()), '\n'

        # Linear activation, using outputs computed above
        return tf.matmul(outputs, weights['out']) + biases['out']

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

        print("Before transpose")
        print(len(outputs))
        #print(outputs[0].get_shape().as_list())
        print(len(outputs[0]))
        print(outputs[0][1].get_shape().as_list())
        print(outputs[0][2].get_shape().as_list())

        outputs = tf.pack(outputs[0])
        outputs = tf.transpose(outputs, [1, 0, 2])

        print("After transpose")
        #print(outputs[0].get_shape().as_list())
        print(outputs[0][1].get_shape().as_list())
        print(outputs[0][2].get_shape().as_list())

        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * max_seq_len + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, 2*N_HIDDEN]), index)

        print("After reshape")
        #print(outputs[0].get_shape().as_list())
        print(outputs[0].get_shape().as_list())
        print(outputs[0].get_shape().as_list())

        # print(outputs.shape)
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
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
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
    # Calculate accuracy
    test_data = testset.pad_all()
    test_label = testset.labels
    test_seqlen = testset.seq_len
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_seqlen}))