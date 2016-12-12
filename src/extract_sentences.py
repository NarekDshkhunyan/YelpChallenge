import json
import nltk
from nltk.tokenize import RegexpTokenizer
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

tokenizer = RegexpTokenizer(r'\w+')

#-------------------------------------------------------------------------------------------------------------
UNIGRAMS = {1: ["bad", "don't", "place", "service", "food", "worst", "horrible", "food", "closed", "service"],
            2: ["great", "place", "good", "service", "food", "better", "ok", "good", "service", "food"],
            3: ["ok", "great", "service", "food", "good", "great", "service", "ok", "food", "good"],
            4: ["place", "service", "food", "great", "good"],
            5: ["service", "place", "best", "food", "great", "place", "love", "food", "best", "great"]}

BIGRAMS = {1: ["don't waste", "bad service", "stay away", "ow ow", "customer service", "terrible service", "place closed", "horrible service", "bad service", "customer service"],
           2: ["good food", "just ok", "good service", "customer service", "food good", "good service", "food ok", "customer service", "food good", "just ok"],
           3: ["service good", "good service", "pretty good", "good food", "food good", "food ok", "good service", "pretty good", "food good", "good food"],
           4: ["food great", "great service", "food good", "great food", "good food", "great service", "good service", "food good", "great food", "good food"],
           5: ["highly recommend", "food great", "love place", "great food", "great service", "service great", "food great", "great service", "great food", "love place"]}

TRIGRAMS = {1: ["don't waste money", "worst customer service", "horrible customer service", "don't' waste time", "ow ow ow", \
                "don't waste money", "poor customer service", "worst customer service", "horrible customer service", "don't waste time"],
            2: ["food great service", "food mediocre best", "food good service", "service good food", "food just ok", \
                "food mediocre best", "meh ve experienced", "food good service", "ve experienced better", "food just ok"],
            3: ["food pretty good", "food just ok", "service good food", "good food good", "food good service"],
            4: ["good food great", "good food good", "food good service", "great food great", "food great service"],
            5: ["love love love", "great service great", "great customer service", "food great service", "great food great"]}


#-------------------------------------------------------------------------------------------------------------
# def get_random_samples(data, labels):
#
#     indices = random.sample(xrange(len(data)), MAX_SAMPLES_TO_USE)
#     return data[indices], labels[indices]


def tokenize(text):
    #split on white spaces, remove punctuation, lower case each word
    return [word.lower() for word in tokenizer.tokenize(text)]


def filter_data(source_file):
    with open(source_file, 'r') as f:
        for line in f:
            review = json.loads(line)
            summary_sentences = []
            try:
                text = review["text"]
                label = review["stars"]
                sentences = sent_detector.tokenize(text)
                #sentences = [sentence for sentence in sentences]
                print label, '\n', sentences

                features = UNIGRAMS[label] + BIGRAMS[label] + TRIGRAMS[label]
                for feature in features:
                    for sentence in sentences:
                        if feature in sentence:
                            print "Feature:", feature
                            print "Extracted sentence:", sentence
                    #print [sentence for sentence in sentences]
                    #print filter(feature in sentences, sentences)

                #print filter(lambda x: x.find())
                #print filter(UNIGRAMS[label] in sentences, sentences)
                #print [[[sentence if feature in sentence] for feature in UNIGRAMS[label]] for sentence in sentences]
                # for sentence in sentences:
                #     for feature in UNIGRAMS[label]:
                #         if feature in sentence: print sentence
            except:
                continue

#-------------------------------------------------------------------------------------------------------------
def main():

    source_file = "../yelp_data/yelp_academic_dataset_review.json"
    filter_data(source_file)


# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
