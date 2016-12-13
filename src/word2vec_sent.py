import numpy as np
import itertools
from collections import Counter
import cPickle
import json 
from nltk.tokenize import RegexpTokenizer
import time
import nltk.data
tokenizer = RegexpTokenizer(r'\w+')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')



def tokenize(text):
    #split on white spaces, remove punctuation, lower case each word
    return [word.lower() for word in tokenizer.tokenize(text)]

def tokenize_sentences(text):
    sentences = sent_detector.tokenize(text.strip())
    return [tokenize(sentence) for sentence in sentences]



def load_data(source_file,tokenize_by_sentence=False):

    txts = []
    stars = []
    review_ids = []
    with open(source_file, 'r') as f:
        for line in f:
            review = json.loads(line) #now each review json is on its own line
            try:
                txt = str(review["text"])
                if not tokenize_by_sentence:
                    txt = tokenize(txt)
                    txts.append(txt)
                    stars.append(review["stars"])
                    review_ids.append(review["review_id"])
                else:
                    sentences = tokenize_sentences(txt)
                    
                    txts.append(sentences)
                    stars.append(review["stars"])#repeat reviews to stars
                    review_ids.append(review["review_id"])#to be able to track back to review
                if len(stars)%100000 == 0 and len(stars)!=0:
                    print "loaded ", len(stars)," reviews."
                if len(stars) >= 500000:
                    break
            except:
                continue
            
    return (txts, stars,review_ids)

def build_vocab(sentences):
    """
    Takes in a list of lists of tokens.
    """
    # Build vocabulary
    #word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    words = set()
    for review in sentences:
        for sent in review:
            for word in sent:
                words.add(word)
    vocabulary_inv = [x[0] for x in words]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    print len(vocabulary), len(vocabulary_inv)
    return [vocabulary, vocabulary_inv]

def vocab_to_word2vec(filename, vocab, k=300):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
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
            word = word.lower()
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    
    print str(len(word_vecs))+" words found in word2vec."

    #============================= review sentence filtering, shuold not be needed

    #add unknown words by generating random word vectors
    #count_missing = 0
   
    #for word in vocab:
    #   if word not in word_vecs:
    #       unidentified_words_file.write(word + "\n")
    #       word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    #       count_missing+=1
    # print str(count_missing)+" words not found, generated by random."

    return word_vecs

def build_word_embedding_mat(word_vecs, vocabulary_inv, k=300):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_mat = np.zeros(shape=(vocab_size, k), dtype='float32')
    for idx in xrange(len(vocabulary_inv)):
        embedding_mat[idx] = word_vecs[vocabulary_inv[idx]]
    print "Embedding matrix of size "+str(np.shape(embedding_mat))
    #initialize the first row,
    #embedding_mat[0]=np.random.uniform(-0.25, 0.25, k)
    return embedding_mat


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    y = []
    for i in xrange(len(sentences)):
        review = sentences[i]
        for sent in review:
            x.append([vocabulary[word] for word in sent])
            y.append(labels[i])
    #x = [[vocabulary[word] for word in sentence] for sentence in sentences]
    y = np.array(labels)
    return [x, y]




if __name__ == "__main__":
    print "starting..."
    start_time = time.time()
    #source_file = "../yelp_data/yelp_academic_dataset_review.json"
    source_file = "../yelp_data/yelp_academic_dataset_review_filtered.json" #pre-filtered reviews #need to resupply this
    googlenews_file = "../google_data/GoogleNews-vectors-negative300.bin"

    sentences, labels,review_ids = load_data(source_file, True) # sentences - list of list of tokens (words), labels - list of floats
    
    print str(len(sentences)) + " sentences read."
    print "Time elapsed: ", time.time()-start_time, " seconds."
    with open("./Pickles/vocab_filtered_big.pkl") as f:
        vocabulary = cPickle.load(f)
    with open("./Pickles/vocab_inv_filtered_big.pkl") as f:
        vocabulary_inv = cPickle.load(f)
    print "Vocabulary size: "+str(len(vocabulary))
    
    start_time = time.time()
    with open("./Pickles/word2vec_big.pkl") as f:
        word2vec = cPickle.load(f)

    embedding_mat = build_word_embedding_mat(word2vec, vocabulary_inv) #i-th row corresponds to i-th word in vocab

    x, y = build_input_data(sentences, labels, vocabulary)      #for each sentences, convert list of tokes to the list of indices in vocab
    print "Time elapsed: ", time.time()-start_time, " seconds."

    start_time = time.time()
    cPickle.dump([x, y, embedding_mat,review_ids], open('Pickles/train_mat_filtered_big_sentences.pkl', 'wb'))
    #cPickle.dump(word2vec, open('Pickles/word2vec_big.pkl', 'wb'))
    #cPickle.dump(vocabulary, open('Pickles/vocab_filtered_big.pkl', 'wb'))
    #cPickle.dump(vocabulary_inv, open('Pickles/vocab_inv_filtered_big.pkl', 'wb'))
    print "Data created"
    print "Time elapsed: ", time.time()-start_time, " seconds."