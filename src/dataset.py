'''
Adapted from:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/dynamic_rnn.py

At the moment, words with word2vec representation only.
'''
import numpy as np

N_INPUT = 300
class Dataset(object):
    """ 
    Generate sequence of data with dynamic length.
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
        """ 
        Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0 #reset

        batch_data = (self.data[self.batch_id:min(self.batch_id +self.batch_size, len(self.data))])
        
        #start padding
        rand_word  = np.random.uniform(-.25, .25, N_INPUT)
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
        '''
        Hack helper for test samples
        '''
        #start padding
        rand_word = np.random.uniform(-.25, .25, N_INPUT)
        data = list(self.data)                                         
        for i in xrange(len(data)):
            dif = self.max_seq_len - len(data[i])
            for j in xrange(dif):
                data[i] = list(data[i])
                data[i].append(rand_word)
        return data

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
