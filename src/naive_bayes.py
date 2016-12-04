from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn import tree

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import CountVectorizer
import cPickle
import numpy as np
import random
from collections import Counter

data_file = "train_mat_filtered.pkl"
vocab_file = "vocab_filtered.pkl"
vocab_inv_file = "vocab_inv_filtered.pkl"

# PARAMETERS
N_FOLDS = 10  # 10% for test, 90% for train
COMPARE_WITH_RANDOM = False
METRICS_CHOICE = 'weighted'  # computes global precision, recall and f1 (not sample or label wise)
# TODO: ask which one is better: 'micro' or 'weighted'

# Note: 'micro' causes recall and precision to be the same
# http://stats.stackexchange.com/questions/99694/what-does-it-imply-if-accuracy-and-recall-are-the-same

MAX_SAMPLES = 65323  # maximum number of available samples, do not change
MAX_SAMPLES_TO_USE = 60000  # can be changed
MAX_SAMPLES_PER_RATING = 4000 # can be set to up to 4002, after which default value is 4002
assert (MAX_SAMPLES_TO_USE <= MAX_SAMPLES)

with open(vocab_file) as f:
    vocabulary = cPickle.load(f)

with open(vocab_inv_file) as f:
    vocabulary_inv = cPickle.load(f)

vocab_size = len(vocabulary)
print "vocabulary size = ", vocab_size


def transform_data(data):
    for i, sample in enumerate(data):
        data[i] = [vocabulary_inv[el] for el in data[i]]
        data[i] = " ".join(data[i])

    vect = CountVectorizer(tokenizer=lambda x: x.rsplit())  # to match the original tokenizer from word2vec.py
    vect = vect.fit(vocabulary.keys())
    data = vect.transform(data)

    return data


def load_data():
    with open(data_file) as f:
        x, y, _ = cPickle.load(f)
    return (x, y)


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


def evaluate(y_test, y_predicted, results):
    precision = precision_score(y_test, y_predicted, pos_label=None,
                                average=METRICS_CHOICE)  # true positives / (true positives+false positives)
    recall = recall_score(y_test, y_predicted, pos_label=None,
                          average=METRICS_CHOICE)  # true positives /(true positives + false negatives)
    f1 = f1_score(y_test, y_predicted, pos_label=None, average=METRICS_CHOICE)
    accuracy = accuracy_score(y_test, y_predicted)  # num of correct predictions/ total num of predictions
    #print "accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1)
    results['accuracy'].append(accuracy)
    results['precision'].append(precision)
    results['recall'].append(recall)
    results['f1'].append(f1)
    return results


def main():
    gnb = MultinomialNB()
    gauss = GaussianNB()

    data, labels = load_data()
    data = np.array(data)
    labels = np.array(labels)
    print data.shape, labels.shape
    #print data[0:10], labels[0:10]
    counter = Counter(labels)

    data, labels = get_random_samples_strictly_uniform(data, labels)
    print "Loaded ", len(data), " samples and ", len(labels), " labels."
    data = transform_data(data)  # get BOW representation
    print data.shape

    skf = StratifiedKFold(n_splits=N_FOLDS)
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    results_random = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)
    print X_train.shape, X_test.shape, y_train.shape, y_test.shape

    gnb = gnb.fit(X_train, y_train)
    y_predicted = gnb.predict(X_test)
    print gnb.score(X_test, y_test)
    results = evaluate(y_test, y_predicted, results)
    print results

    # gauss = gauss.fit(X_train.toarray(), y_train)
    # y_predicted = gauss.predict(X_test.toarray())
    # print gauss.score(X_test.toarray(), y_test)
    # results = evaluate(y_test, y_predicted, results)
    # print results

    #clf = svm.SVC(kernel="linear", gamma=1.0)
    clf = svm.SVC(C=1000.0, kernel="rbf")
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print clf.score(X_test, y_test)
    results = evaluate(y_test, y_pred, results)
    print results

    clf = KNeighborsClassifier(n_neighbors=10, algorithm='auto')
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print clf.score(X_test, y_test)
    results = evaluate(y_test, y_pred, results)
    print results

    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print clf.score(X_test, y_test)
    results = evaluate(y_test, y_pred, results)
    print results

    #print "\n=====================Total average results==================="
    #for key in results:
        #print key, results[key]
        #print key + " = %.3f" % results[key][0]

    # for train, test in skf.split(data, labels):
    #     X_train, X_test = data[train], data[test]
    #     y_train, y_test = labels[train], labels[test]
    #     #print train, test
    #     #print "Data:", X_train.shape, X_test.shape
    #     #print "labels:", y_train.shape, y_test.shape
    #
    #     gnb = gnb.fit(X_train, y_train)
    #     #print gnb.class_count_.shape, gnb.feature_count_.shape
    #     #print gnb.class_count_
    #     y_predicted = gnb.predict(X_test)
    #
    #     results = evaluate(y_test, y_predicted, results)
    #
    #     if COMPARE_WITH_RANDOM:
    #         counter = Counter(y_train)
    #         prob_weigths = [counter[i] * 1. / len(y_train) for i in xrange(1, 6)]
    #         y_predicted = [np.random.choice(xrange(1, 6), size=1, replace=False, p=prob_weigths) for i in xrange(len(y_test))]
    #         #print "-------------Random picker results:------------------"
    #         results_random = evaluate(y_test, y_predicted, results_random)
    #         #print
    #
    # print "\n=====================Total average results==================="
    # for key in results:
    #     print key + " = %.3f," % (sum(results[key]) / N_FOLDS),
    #
    # if COMPARE_WITH_RANDOM:
    #     print
    #     #print "=====================Total average for random ==================="
    #     #for key in results:
    #         #print key + " = %.3f," % (sum(results_random[key]) / N_FOLDS),


if __name__ == "__main__":
    main()
    
#     Current Performance (with get_random_samples)
#     =====================Total average results===================
#     f1 = 0.601, recall = 0.618, precision = 0.597, accuracy = 0.618,
#     =====================Total average for random ===================
#     f1 = 0.317, recall = 0.317, precision = 0.317, accuracy = 0.317,
    
#     Current Performance (with get_random_samples_strictly_uniform)
    # =====================Total average results===================
    # f1 = 0.528, recall = 0.530, precision = 0.527, accuracy = 0.530,
    # =====================Total average for random ===================
    # f1 = 0.203, recall = 0.203, precision = 0.203, accuracy = 0.203,
