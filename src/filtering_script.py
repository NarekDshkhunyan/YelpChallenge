import numpy as np
import itertools
from collections import Counter
import cPickle
import json 
from nltk.tokenize import RegexpTokenizer
import time

tokenizer = RegexpTokenizer(r'\w+')


def tokenize(text):
    #split on white spaces, remove punctuation, lower case each word
    return [word.lower() for word in tokenizer.tokenize(text)]


def filter_data(source_file, target_file, valid_words):
    filtered_reviews = []
    with open(source_file, 'r') as f:
        for line in f:
            review = json.loads(line)
            try:
                filtered_review = {}
                txt = str(review["text"])
                tokenized = tokenize(txt)
                valid = True
                for word in tokenized:
                    if word not in valid_words:
                        valid = False
                        break
                if valid:

                    filtered_review["stars"] = review["stars"]
                    filtered_review["text"] = review["text"]
                    filtered_review["review_id"] = review["review_id"]
                    filtered_reviews.append(filtered_review)
            except:
                continue
    with open(target_file, 'w') as outfile:
        json.dump(filtered_reviews, outfile)
    print len(filtered_reviews)
    return



def extract_words(filename):
    words = set()
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for _ in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            words.add(word.lower())
            f.read(binary_len)
    return words
    



if __name__ == "__main__":
    print "starting..."
    source_file = "../yelp_data/yelp_academic_dataset_review.json"
    # source_file = "../yelp_data/mock_reviews.json"
    target_file = "../yelp_data/yelp_academic_dataset_review_filtered.json"
    googlenews_file = "../google_data/GoogleNews-vectors-negative300.bin"
    # unidentified_words = "./misc/unidentified_words.txt"

    googleWords = extract_words(googlenews_file)
    filter_data(source_file, target_file, googleWords) # sentences - list of list of tokens (words), labels - list of floats
