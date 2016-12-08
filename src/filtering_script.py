import numpy as np
from spellchecker import correction, valid
import json 
from nltk.tokenize import RegexpTokenizer
import time

tokenizer = RegexpTokenizer(r'\w+')


def tokenize(text):
    #split on white spaces, remove punctuation, lower case each word
    return [word.lower() for word in tokenizer.tokenize(text)]


def filter_data(source_file, target_file, google_words):
    filtered_reviews = []
    with open(source_file, 'r') as f:
        not_valid_words = []
        valid_words = []
        for line in f:
            valid_word, not_valid = 0, 0
            review = json.loads(line)
            try:
                filtered_review = {}
                txt = str(review["text"])
                tokenized = tokenize(txt)
                inGoogleNews = True
                #print tokenized
                #print review["text"]
                for word in tokenized:
                    #print word
                    if word not in google_words:
                        inGoogleNews = False
                        break

                    if not valid(word):          # 2.5M words were flagged, out out 18.5M total
                        not_valid += 1
                    elif valid(word):
                        valid_word += 1

                #print "stars:", review["stars"]
                #print not_valid, '\n'
                not_valid_words.append(not_valid)
                valid_words.append(valid_word)

                if inGoogleNews:
                    filtered_review["stars"] = review["stars"]
                    filtered_review["text"] = review["text"]
                    filtered_review["review_id"] = review["review_id"]
                    filtered_reviews.append(filtered_review)
            except:
                continue
        print "Not valid words:", sum(not_valid_words)
        print "Valid words:", sum(valid_words)
    with open(target_file, 'w') as outfile:
        for r in filtered_reviews:
            outfile.write(json.dumps(r)+"\n")#to make sure each json is each line
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
    start = time.time()
    print "starting..."
    source_file = "../yelp_data/yelp_academic_dataset_review.json"
    target_file = "../yelp_data/big_yelp_academic_dataset_review_filtered.json"
    googlenews_file = "../google_data/GoogleNews-vectors-negative300.bin"

    googleWords = extract_words(googlenews_file)
    filter_data(source_file, target_file, googleWords) # sentences - list of list of tokens (words), labels - list of floats
    print time.time() - start
