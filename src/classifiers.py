
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm, tree
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spellchecker import correction

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import string

import cPickle
import numpy as np
import random
from collections import Counter

import time

data_file = "Pickles/train_mat_filtered.pkl"
vocab_file = "Pickles/vocab_filtered.pkl"
vocab_inv_file = "Pickles/vocab_inv_filtered.pkl"

# PARAMETERS
COMPARE_WITH_RANDOM = True
METRICS_CHOICE = 'weighted'             # computes global precision, recall and f1 (not sample or label wise)

# Note: 'micro' causes recall and precision to be the same
# http://stats.stackexchange.com/questions/99694/what-does-it-imply-if-accuracy-and-recall-are-the-same

MAX_SAMPLES = 1594893               # whole Yelp dataset after filtering, only relevant if using _big pcikle files
MAX_SAMPLES_TO_USE = 60000           # can be changed
TEST_SAMPLES_PERCENTAGE = 0.1
MAX_SAMPLES_PER_RATING = 150688     # can be set to up to 4002, after which default value is 4002

assert (MAX_SAMPLES_TO_USE <= MAX_SAMPLES)

#-------------------------------------------------------------------------------------------------------------
def load_data():
    with open(data_file) as f:
        x, y, _ = cPickle.load(f)
    return (x, y)

with open(vocab_file) as f:
    vocabulary = cPickle.load(f)

with open(vocab_inv_file) as f:
    vocabulary_inv = cPickle.load(f)

vocab_size = len(vocabulary)
print "vocabulary size = ", vocab_size

# vocab = {}
# for key in vocabulary.keys():
#     vocab[correction(key)] = vocabulary[key]
# vocabulary = vocab
# print len(vocabulary)

#-------------------------------------------------------------------------------------------------------------
def transform_data(data):
    """ Tranform data into the actual text of the reviews """
    for i, sample in enumerate(data):
        data[i] = [vocabulary_inv[el] for el in data[i]]
        data[i] = " ".join(data[i])
        #print data[i]
    return data


POSITIVE_WORDS = set([line.strip() for line in open('../Stopwords/positive-words.txt', 'r')])
NEGATIVE_WORDS = set([line.strip() for line in open('../Stopwords/negative-words.txt', 'r')])
MORE_STOPWORDS = set([line.strip() for line in open('../Stopwords/more_stopwords.txt', 'r')])

def clean_data(data):
    """ Helper function to remove digits, punctuation, and/or stop words """
    for i, sample in enumerate(data):
        data[i] = data[i].translate(None, string.digits)                          # remove digits
        data[i] = data[i].translate(None, string.punctuation)
        #token_list = word_tokenize(data[i])                                       # tokenize the words
        # token_list = filter(lambda tok: tok not in MORE_STOPWORDS               # remove stopwords
        #                                and tok not in POSITIVE_WORDS
        #                                and tok not in NEGATIVE_WORDS, token_list)
        #STEMMER = PorterStemmer()
        #token_list = [STEMMER.stem(tok.decode('utf-8')) for tok in token_list]  # get the stem of the words
        #data[i] = ' '.join(token_list)
        #print data[i]
    return data


#-------------------------------------------------------------------------------------------------------------
def get_random_samples(data, labels):
    if MAX_SAMPLES_TO_USE == MAX_SAMPLES:
        return data, labels
    indicies = random.sample(xrange(len(data)), MAX_SAMPLES_TO_USE)
    return data[indicies], labels[indicies]

def get_random_samples_strictly_uniform(data,labels):
    counts = np.bincount(labels)
    max_available = min(min(counts[1:]), MAX_SAMPLES_PER_RATING)
    ratings_arr = [np.where(labels == i) for i in xrange(1, 6)]

    relevant_indices = []
    for s in ratings_arr:
        chosen = random.sample(xrange(len(s[0])), max_available)
        chosen_idx = [int(s[0][i]) for i in chosen]
        relevant_indices.extend(chosen_idx)

    np.random.shuffle(relevant_indices)
    return data[relevant_indices], labels[relevant_indices]

#-------------------------------------------------------------------------------------------------------------
def evaluate(y_test, y_predicted, results):
    """ Helper function to print the results """
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average=METRICS_CHOICE)  # true positives / (true positives+false positives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average=METRICS_CHOICE)  # true positives /(true positives + false negatives)
    f1 = f1_score(y_test, y_predicted, pos_label=None, average=METRICS_CHOICE)
    accuracy = accuracy_score(y_test, y_predicted)  # num of correct predictions/ total num of predictions
    print "accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    return results


#-------------------------------------------------------------------------------------------------------------
def classify(method, X_train, y_train, X_test, y_test, features, results):
    """ Runs the classification algorithm of the choice """

    start = time.time()
    if method == "mnb":
        clf = MultinomialNB(alpha=0.5, class_prior=[0.1, 0.1, 0.1, 0.25, 0.45])
    elif method == "svm":
        # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
        # clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = svm.SVC(C=1000.0, kernel="rbf")
    elif method == "knn":
        clf = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    elif method == "dt":
        clf = tree.DecisionTreeClassifier(min_samples_split=30)
    elif method == "rf":
        clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', oob_score=True, n_jobs=-1, random_state=42)
    elif method == "lr":
        clf = LogisticRegression(C=0.1, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1,
                                random_state=42)

    # Fit the model and predict labels
    clf = clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    # Extract the most important features
    if method == "mnb":
        important_indices = [[np.where(clf.feature_log_prob_[i] == feature) for feature in clf.feature_log_prob_[i] if np.exp(feature) > 0.01] for i in [0,1,2,3,4]]
        print [[features[index[i][0][0]] for i in xrange(len(index))] for index in important_indices]
        N = 5
        for i, label in enumerate(sorted(set(y_train))):
            top_n_features_indices = np.argsort(clf.coef_[i])[-N:]
            print "\nThe top %d most informative features for star category %d: \n%s" % (
                N, label, ", ".join([features[index] for index in top_n_features_indices]))

    if method == "rf" or method == "dt":
        important_indices = [np.where(clf.feature_importances_ == feature) for feature in clf.feature_importances_ if feature > 0.02]
        print [features[index[0][0]] for index in important_indices]

    target_names = ['1 star', '2 star', '3 star', '4 star', '5 star']
    print classification_report(y_test, y_predicted, target_names=target_names)
    results = evaluate(y_test, y_predicted, results)
    print "Time elapsed:", time.time() - start, '\n'


#-------------------------------------------------------------------------------------------------------------
def main():

    # Load the data frim pickles and put into NumPy arrays
    data, labels = load_data()
    data = np.array(data)
    labels = np.array(labels)
    print data.shape, labels.shape
    #counter = Counter(labels)

    # Randomly select some portion of the dataset for classification
    #data, labels = get_random_samples_strictly_uniform(data, labels)
    data, labels = get_random_samples(data, labels)
    print "Loaded ", len(data), " samples and ", len(labels), " labels."

    # Transform and clean the data
    # Split the data into train and test samples
    data = clean_data(transform_data(data))
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=TEST_SAMPLES_PERCENTAGE, random_state=42)
    print "Train data shape ", X_train.shape
    print "Test data shape ", X_test.shape
    print "Train labels shape ", y_train.shape
    print "Test labels shape ", y_test.shape, '\n'

    # Transform both train and test samples into BOW or tfidf representation
    vect = CountVectorizer(ngram_range=(1,1), analyzer='word', stop_words='english')
    #vect = TfidfVectorizer(lambda x: x.rsplit(), ngram_range=(3,3), stop_words='english')
    X_train = vect.fit_transform(X_train)
    X_test = vect.transform(X_test)
    features = vect.get_feature_names()
    print "Total # of features:", len(features)
    #print features[0:100]

    del data, labels

    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    results_random = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    # Run the classification algorithm
    classify('rf', X_train, y_train, X_test, y_test, features, results)

    if COMPARE_WITH_RANDOM:
        print " ===== Random ====="
        counter = Counter(y_train)
        prob_weigths = [counter[i] * 1. / len(y_train) for i in xrange(1, 6)]
        y_predicted = [np.random.choice(xrange(1, 6), size=1, replace=False, p=prob_weigths) for i in xrange(len(y_test))]
        results_random = evaluate(y_test, y_predicted, results_random)


#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
