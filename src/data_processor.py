import json 
from nltk.tokenize import RegexpTokenizer
from word2vec_model import word2vec

	

class Preprocessor:
	'''
	Class for conversion from string to the list of words

	Current implementation:
		 - strips out the punctuation and lower case each words

	TODO:
		 - add spell checker
		 - add stemmer and lemmatization
		 - 
	'''
	def __init__(self):
		self.tokenizer = RegexpTokenizer(r'\w+') #strips punctuation

	def run(self, text):
		'''
		API method, do not change

		Args:
			text - string, raw text
		Returns:
			list of strings, separate processed words
		'''
		return [word.lower() for word in self.tokenizer.tokenize(word)]



class DataProcessor:
	def __init__(self, source_file, model_to_use=word2vec):
		self.source_file = source_file # string, filepath
		self.preprocessor = Preprocessor() #has run method taking each review
		self.reviews = [] #list of lists
		self.starts = [] # list of
		self.model = model_to_use # function, either word2vec or bag_of_words

		self.review_to_embedded_matrix = {}


	def get_data(self):
		with open(self.source_file) as f:
			for line in f:
				review = json.loads(line)
				self.reviews.append(self.preprocessor.run(review['text']))
				self.stars.append(review["stars"])

	
	
	def build_embedded_matrices(self):
		'''
		API method, do not change
		'''
		self.get_data()
		self.review_to_embedded_matrix = self.model(self.reviews)

	def get_embedded_matrices(self):
		return self.review_to_embedded_matrix
		
	def get_stars(self):
		return self.stars





