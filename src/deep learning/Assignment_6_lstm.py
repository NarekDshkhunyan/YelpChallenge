
# coding: utf-8

# After training a skip-gram model in 5_word2vec.ipynb, the goal of this notebook is to train a LSTM character model over Text8 data

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import os
import numpy as np
import random
import string
import tensorflow as tf
import zipfile
from six.moves import range
from six.moves.urllib.request import urlretrieve


# In[2]:

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)



def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return tf.compat.as_str(f.read(name))
  f.close()
  
text = read_data(filename)
print('Data size %d' % len(text))


# Create training and validation sets

# In[3]:

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print(train_size, train_text[:64])
print(valid_size, valid_text[:64])


# Utility functions to map characters to vocabulary IDs and back.

# In[4]:

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])

def char2id(char):
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char)
    return 0
  
def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '

print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
print(id2char(1), id2char(26), id2char(0))


# Function to generate a training batch for the LSTM model.

# In[5]:

batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))


# Few more useful functions

# In[6]:

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]


# Simple LSTM model

# In[12]:

num_nodes = 64

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Input gate: input, previous output, and bias. Drop the multiplication by four if not following Problem 1 code
  ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes*4], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes*4], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes*4]))
    
  # Forget gate: input, previous output, and bias.
  fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  fb = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Memory cell: input, state and bias.                             
  cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  cb = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Output gate: input, previous output, and bias.
  ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
  om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ob = tf.Variable(tf.zeros([1, num_nodes]))
    
  # Variables saving state across unrollings.
  saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)

  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
  
  # Definition of the cell computation.
  def lstm_cell(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the previous state and the gates."""
    #input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
    #forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
    #update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
    #state = forget_gate * state + input_gate * tf.tanh(update)
    #output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
    #return output_gate * tf.tanh(state), state

    # Problem 1
    mult = tf.matmul(i, ix) + tf.matmul(o, im) + ib
    input_gate_raw, forget_gate_raw, update, output_gate_raw = tf.split(1, 4, mult)
    input_gate = tf.sigmoid(input_gate_raw)
    forget_gate = tf.sigmoid(forget_gate_raw)
    output_gate = tf.sigmoid(output_gate_raw)  
    state = forget_gate * state + input_gate * tf.tanh(update)
    return output_gate * tf.tanh(state), state

  # Input data.
  train_data = list()
  for _ in range(num_unrollings + 1):
    train_data.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  train_inputs = train_data[:num_unrollings]
  train_labels = train_data[1:]                           # labels are inputs shifted by one time step.

  # Unrolled LSTM loop.
  outputs = list()
  output = saved_output
  state = saved_state
  for i in train_inputs:
    output, state = lstm_cell(i, output, state)
    outputs.append(output)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output.assign(output), saved_state.assign(state)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
  saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(saved_sample_output.assign(tf.zeros([1, num_nodes])), 
                                saved_sample_state.assign(tf.zeros([1, num_nodes])))

  sample_output, sample_state = lstm_cell(sample_input, saved_sample_output, saved_sample_state)
  with tf.control_dependencies([saved_sample_output.assign(sample_output), saved_sample_state.assign(sample_state)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


# Now run this model

# In[18]:

num_steps = 7001
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print('Initialized')
  mean_loss = 0

  for step in range(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    for i in range(num_unrollings + 1):
      feed_dict[train_data[i]] = batches[i]
    
    _, l, predictions, lr = session.run([optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % (summary_frequency * 10) == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print('Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print('Minibatch perplexity: %.2f' % float(np.exp(logprob(predictions, labels))))
    
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print('=' * 80)
        for _ in range(5):
          feed = sample(random_distribution())
          sentence = characters(feed)[0]
          reset_sample_state.run()
          for _ in range(79):
            prediction = sample_prediction.eval({sample_input: feed})
            feed = sample(prediction)
            sentence += characters(feed)[0]
          print(sentence)
        print('=' * 80)
        
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in range(valid_size):
        b = valid_batches.next()
        predictions = sample_prediction.eval({sample_input: b[0]})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print('Validation set perplexity: %.2f' % float(np.exp(valid_logprob / valid_size)))


# We want to train a LSTM over bigrams, that is pairs of consecutive characters like 'ab' instead of single characters like 'a'. Since the number of possible bigrams is large, feeding them directly to the LSTM using 1-hot encodings will lead to a very sparse representation that is very wasteful computationally.
# 
# a- Introduce an embedding lookup on the inputs, and feed the embeddings to the LSTM cell instead of the inputs themselves.
# 
# b- Write a bigram-based LSTM, modeled on the character LSTM above.
# 
# c- Introduce Dropout. For best practices on how to use Dropout in LSTMs, refer to this article.

# In[15]:

def bigrams(embeddings):
    """Turn a probability distribution over the bigrams into the most likely
       bigram string representations."""
    bigrams = []
    for i in range(embeddings.shape[0]):
        cosims = np.dot(final_embeddings, embeddings[i]) / la.norm(embeddings[i])
        bigrams.append(reverse_bigram_dict[np.argmax(cosims)])
    return " ".join(bigrams)

batch_size = 64
num_unrollings = 10
vocabulary_size = 128   # since we are now projecting our 729 to 128

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size / (2 * batch_size)
        self._cursor = [ 2 * offset * segment for offset in xrange(batch_size)]
        self._last_batch = self._next_batch()
  
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
        for b in xrange(self._batch_size):
            bigram_idx = bigram_dict[self._text[self._cursor[b]:self._cursor[b]+2]]
            batch[b] = final_embeddings[bigram_idx]
            self._cursor[b] = (self._cursor[b] + 2) % self._text_size
        return batch
  
    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]
        for step in xrange(self._num_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches

def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (mostl likely) character representation."""
    closest_bigrams = []
    for i in range(probabilities.shape[0]):
        p = probabilities[i, :]
        bigram_idx = np.argmax(np.dot(final_embeddings, p) / np.linalg.norm(p), 0)
        closest_bigrams.append(reverse_bigram_dict[bigram_idx])
    return closest_bigrams
        

def batches2string(batches):
    """Convert a sequence of batches back into their (most likely) string
    representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, characters(b))]
    return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
valid_batches = BatchGenerator(valid_text, 1, 1)

print(batches2string(train_batches.next()))
print(batches2string(train_batches.next()))
print(batches2string(valid_batches.next()))
print(batches2string(valid_batches.next()))


# In[ ]:




# In[17]:

num_nodes = 64
embedding_size = 64
num_steps = 24001

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Variables saving state across unrollings.
  saved_output1 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state1 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  
  saved_output2 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state2 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
    
  # Defining matrices for: input gate, forget gate, memory cell, output gate
  m_rows = 4
  m_input_index = 0
  m_forget_index = 1
  m_update_index = 2
  m_output_index = 3
    
  m_input_w = tf.Variable(tf.truncated_normal([m_rows, embedding_size, num_nodes], -0.1, 0.1))
  m_middle = tf.Variable(tf.truncated_normal([m_rows, num_nodes, num_nodes], -0.1, 0.1))
  m_biases = tf.Variable(tf.truncated_normal([m_rows, 1, num_nodes], -0.1, 0.1))
  m_saved_output = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  m_input = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  # Dropout
  keep_prob = tf.placeholder(tf.float32) 

  # Definition of the 2nd LSTM layer
  m_input_w2 = tf.Variable(tf.truncated_normal([m_rows, embedding_size, num_nodes], -0.1, 0.1))
  m_middle_w2 = tf.Variable(tf.truncated_normal([m_rows, num_nodes, num_nodes], -0.1, 0.1))
  m_biases2 = tf.Variable(tf.truncated_normal([m_rows, 1, num_nodes], -0.1, 0.1))
  m_saved_output2 = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  m_input2 = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  
  # Definition of the cell computation.
  def lstm_cell_improved(i, o, state):
    m_input = tf.pack([i for _ in range(m_rows)])
    m_saved_output = tf.pack([o for _ in range(m_rows)])
    
    m_input = tf.nn.dropout(m_input, keep_prob)
    m_all = tf.batch_matmul(m_input, m_input_w) + tf.batch_matmul(m_saved_output, m_middle) + m_biases
    m_all = tf.unpack(m_all)
    
    input_gate = tf.sigmoid(m_all[m_input_index])
    forget_gate = tf.sigmoid(m_all[m_forget_index])
    update = m_all[m_update_index]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(m_all[m_output_index])
    
    return output_gate * tf.tanh(state), state
  
  def lstm_cell_2(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""    
    m_input2 = tf.pack([i for _ in range(m_rows)])
    m_saved_output2 = tf.pack([o for _ in range(m_rows)])
    
    m_input2 = tf.nn.dropout(m_input2, keep_prob)
    m_all = tf.batch_matmul(m_input2, m_input_w2) + tf.batch_matmul(m_saved_output2, m_middle_w2) + m_biases
    m_all = tf.unpack(m_all)
    
    input_gate = tf.sigmoid(m_all[m_input_index])
    forget_gate = tf.sigmoid(m_all[m_forget_index])
    update = m_all[m_update_index]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(m_all[m_output_index])
    
    return output_gate * tf.tanh(state), state
  
  # Input data.
  train_data = list()
  train_labels = list()
  
  for x in xrange(num_unrollings):
    train_data.append(tf.placeholder(tf.int32, shape=[batch_size]))
    train_labels.append(tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  
  encoded_inputs = list()
  for bigram_batch in train_data:
    embed = tf.nn.embedding_lookup(embeddings, bigram_batch)
    encoded_inputs.append(embed)
  
  train_inputs = encoded_inputs

  # Unrolled LSTM loop.
  outputs = list()
  output1 = saved_output1
  output2 = saved_output2
    
  state1 = saved_state1
  state2 = saved_state2
    
  for i in train_inputs:
    output1, state1 = lstm_cell_improved(i, output1, state1)
    output2, state2 = lstm_cell_2(output1, output2, state2)
    outputs.append(output2)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output1.assign(output1),
                                saved_state1.assign(state1),
                                saved_output2.assign(output2),
                                saved_state2.assign(state2)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(10.0, global_step, num_steps / 2, 0.1, staircase=False)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1])
  sample_embed = tf.nn.embedding_lookup(embeddings, sample_input)
  saved_sample_output1 = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state1 = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_output2 = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state2 = tf.Variable(tf.zeros([1, num_nodes]))

  reset_sample_state = tf.group(
    saved_sample_output1.assign(tf.zeros([1, num_nodes])),
    saved_sample_state1.assign(tf.zeros([1, num_nodes])),
    saved_sample_output2.assign(tf.zeros([1, num_nodes])),
    saved_sample_state2.assign(tf.zeros([1, num_nodes])))
    
  sample_output1, sample_state1 = lstm_cell_improved(sample_embed, saved_sample_output1, saved_sample_state1)
  sample_output2, sample_state2 = lstm_cell_2(sample_output1, saved_sample_output2, saved_sample_state2)
    
  with tf.control_dependencies([saved_sample_output1.assign(sample_output1),
                                saved_sample_state1.assign(sample_state1),
                                saved_sample_output2.assign(sample_output2),
                                saved_sample_state2.assign(sample_state2)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output2, w, b))


# In[ ]:



