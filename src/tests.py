
from data_processor import DataProcessor
from bag_of_words import bag_of_words
from word2vec_model import word2vec

import numpy as np

def test_word_2_vec_simples():
	r1 = ["i", "like", "dogs"]
	r2 = ["i", "love", "cats"]
	r3 = ["i", "hate", "dogs"]

	reviews = [r1,r2,r3]

	embedded_matrix = word2vec(reviews)
	# print embedded_matrix

	print "Similar words should have a smaller distance than opposite meaning words"
	print "Like and love distance"
	print np.sum(np.multiply(embedded_matrix[0][1],embedded_matrix[1][1]))

	print "Like and hate distance"
	print np.sum(np.multiply(embedded_matrix[0][1],embedded_matrix[2][1]))

test_word_2_vec_simples()