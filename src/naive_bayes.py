from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold
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
COMPARE_WITH_RANDOM = True
METRICS_CHOICE = 'weighted'  # computes global precision, recall and f1 (not sample or label wise)
# TODO: ask which one is better: 'micro' or 'weighted'

# Note: 'micro' causes recall and precision to be the same
# http://stats.stackexchange.com/questions/99694/what-does-it-imply-if-accuracy-and-recall-are-the-same

MAX_SAMPLES = 65323  # maximum number of available samples, do not change
MAX_SAMPLES_TO_USE = 60000  # can be changed
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


def evaluate(y_test, y_predicted, results):
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


def main():
    gnb = MultinomialNB()

    data, labels = load_data()
    data = np.array(data)
    labels = np.array(labels)
    counter = Counter(labels)

    data, labels = get_random_samples(data, labels)
    print "Loaded ", len(data), " samples and ", len(labels), " labels."
    data = transform_data(data)  # get BOW representation

    skf = StratifiedKFold(n_splits=N_FOLDS)
    results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    results_random = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for train, test in skf.split(data, labels):
        X_train, X_test = data[train], data[test]
        y_train, y_test = labels[train], labels[test]

        gnb = gnb.fit(X_train, y_train)
        y_predicted = gnb.predict(X_test)

        results = evaluate(y_test, y_predicted, results)

        if COMPARE_WITH_RANDOM:
            counter = Counter(y_train)
            prob_weigths = [counter[i] * 1. / len(y_train) for i in xrange(1, 6)]
            y_predicted = [np.random.choice(xrange(1, 6), size=1, replace=False, p=prob_weigths) for i in
                           xrange(len(y_test))]
            print "-------------Random picker results:------------------"
            results_random = evaluate(y_test, y_predicted, results_random)
            print

    print "\n=====================Total average results==================="
    for key in results:
        print key + " = %.3f," % (sum(results[key]) / N_FOLDS),

    if COMPARE_WITH_RANDOM:
        print
        print "=====================Total average for random ==================="
        for key in results:
            print key + " = %.3f," % (sum(results_random[key]) / N_FOLDS),


if __name__ == "__main__":
    main()
