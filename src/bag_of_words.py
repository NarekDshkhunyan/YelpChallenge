
from sklearn.feature_extraction.text import CountVectorizer

MAX_VECTOR_DIM = 5000

def bag_of_words(reviews):
	'''
	Args:
		reviews : list of lists of words (preprocessed)

	Returns:
		matrix of integers, a vector for each word in all reviews
	'''
	reviews = [" ".join(review) for review in reviews]

	vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = MAX_VECTOR_DIM) 

	# fit_transform() does two functions: First, it fits the model
	# and learns the vocabulary; second, it transforms our training data
	#into feature vectors. The input to fit_transform should be a list of 
	# strings.
	train_data_features = vectorizer.fit_transform(reviews)

	# create a mapping betwee nthe review index and its corresponding embedded matrix of size (1, vocab_size)   
        review_to_embedded_matrix = {}
	features =  train_data_features.toarray()
	for i in xrange(len(reviews)):
		review_to_embedded_matrix[i] = features[i]
	return review_to_embedded_matrix
