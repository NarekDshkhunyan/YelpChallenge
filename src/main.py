from data_processor import DataProcessor
from bag_of_words_model import bag_of_words
from word2vec_model import word2vec
SOURCE_FILE = "../yelp_data/yelp_academic_dataset_review.json"

def main():
	dp = DataProcessor(SOURCE_FILE,model_to_use=word2vec)
	dp.build_embedded_matrices() 
	#get samples, their feature vectors  and labels (stars)
	#order is the same
	samples = dp.get_embedded_matrices()
	labels = dp.get_stars()





if __name__ == '__main__':
    main()
