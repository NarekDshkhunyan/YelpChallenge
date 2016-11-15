
import numpy as np
import itertools
from collections import Counter

FILENAME = "../google_data/GoogleNews-vectors-negative300.bin"
#FILENAME = "/Volumes/Dana1TB/xaa"
def word2vec(reviews):
    '''
    Args:
	    filename : string, path to google news word2vec binary
    	    reviews : list of lists of words (preprocessed)
    Returns:
	    mapping from word to feature vector (string --> array of floats)
	'''
    #build vocabulary from reviews
    word_counts = Counter(itertools.chain(*reviews))
    #most_common() returns all elements
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}

    # load google-news word2vecs, file is binary
    word_vecs = {}
    with open(FILENAME, "rb") as f:
        header = f.readline()
        num_words, dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * dim #size in bytes of 1 word
        for _ in xrange(num_words):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocabulary:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    #add unknown words by generating random word vectors
    for word in vocabulary:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)

    # create a mapping between the review index and its corresponding embedded matrix of size (num_words, 300)   
    review_to_embedded_matrix = {}
    for i in xrange(len(reviews)):
        review_to_embedded_matrix[i] = [word_vecs[word] for word in reviews[i]]
    return review_to_embedded_matrix
