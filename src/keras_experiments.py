import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import cPickle
import random
from keras.utils import np_utils

('\n'
 'Sources:\n'
 'LSTM:\n'
 'http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/\n'
 'Use existing word2vec embedding:\n'
 'https://github.com/fchollet/keras/issues/853\n'
 'How to make a multiclass classifier:\n'
 'http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/\n')
data_file = "train_mat_filtered.pkl"
vocab_file = "vocab_filtered.pkl"
vocab_inv_file = "vocab_inv_filtered.pkl"

#PARAMETERS
N_FOLDS = 10 # 10% for test, 90% for train
LSTM_MEM_UNITS = 100
BATCH_SIZE = 64
MAX_SAMPLES = 65323 # maximum number of available samples, do not change
MAX_SAMPLES_TO_USE = 5000 # can be changed
assert(MAX_SAMPLES_TO_USE <= MAX_SAMPLES)

def get_vocabs():
	with open(vocab_file) as f:
		vocabulary = cPickle.load(f)
	# for key in vocabulary:
	# 	vocabulary[key]+=1 # index 0 is reserved for masking

	with open(vocab_inv_file) as f:
		vocabulary_inv = cPickle.load(f)
	vocabulary_inv = ['IGNORE']+vocabulary_inv # index 0 is reserved for masking
	return (vocabulary, vocabulary_inv)

def load_data():
	with open(data_file) as f:
		x,y, embedding_matrix = cPickle.load(f)
	zero_row = np.random.uniform(-.25, .25, 300) #to match 0th index
	embedding_matrix = np.concatenate((np.array([zero_row]),embedding_matrix))# index 0 is reserved for masking
	
	for row in xrange(len(x)):
		for col in xrange(len(x[row])):
			x[row][col]+=1 #increment stuff by 1 to match the vocab
	return (x,y, embedding_matrix)

def get_random_samples(data,labels):
	if MAX_SAMPLES_TO_USE == MAX_SAMPLES:
		return data, labels
	indicies = random.sample(xrange(len(data)), MAX_SAMPLES_TO_USE)
	return data[indicies], labels[indicies]

def main():

	vocabulary, vocabulary_inv = get_vocabs()
	data, labels, embedding_matrx = load_data()
	max_review_length = max([len(i) for i in data])
	data,labels = get_random_samples(np.array(data), np.array(labels))
	print len(data), " samples and ", len(labels), " ready."
	
	max_review_length = max([len(i) for i in data])
	vocab_size = len(vocabulary)
	num_labels = 5
	encoder = LabelEncoder()
	encoder.fit(labels)
	skf = StratifiedKFold(n_splits=N_FOLDS)
	
	for train, test in skf.split(data, labels):
		X_train, X_test = data[train], data[test]
		y_train, y_test = np.array(labels[train]), np.array(labels[test])
		X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
		X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
		# print max_review_length
		# print X_train.shape
		# print X_test.shape

		y_train = encoder.transform(y_train)
		# convert integers to dummy variables (i.e. one hot encoded)
		y_train = np_utils.to_categorical(y_train)

		y_test = encoder.transform(y_test)
		y_test = np_utils.to_categorical(y_test)

		model = Sequential()
		#use existing embedding weights
		model.add(Embedding(input_dim=vocab_size + 1, output_dim=300, mask_zero=True, weights=[embedding_matrx],input_length=X_train.shape[1])) 
		#model.add(Embedding(input_dim=vocab_size, output_dim=300, mask_zero=True, weights=[embedding_matrx],input_length=X_train.shape[1])) 
		model.add(Bidirectional(LSTM(LSTM_MEM_UNITS),merge_mode='concat'))
		#model.add(Dropout(0.5))
		model.add(Dense(num_labels,activation='sigmoid'))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		model.fit(X_train, y_train, nb_epoch=3, batch_size=BATCH_SIZE)
		# Final evaluation of the model
		scores = model.evaluate(X_test, y_test, verbose=0)
		print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ =="__main__":
	main()