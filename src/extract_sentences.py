import json
import nltk
from nltk.tokenize import RegexpTokenizer
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

tokenizer = RegexpTokenizer(r'\w+')

#-------------------------------------------------------------------------------------------------------------
UNIGRAMS = {1: ["bad", "place", "service", "food", "worst", "horrible", "closed"],
            2: ["great", "place", "good", "service", "food", "better", "ok"],
            3: ["ok", "great", "service", "food", "good", "ok"],
            4: ["place", "service", "food", "great", "good"],
            5: ["service", "place", "best", "food", "great", "love"]}

BIGRAMS = {1: ["don't waste", "bad service", "stay away", "ow ow", "customer service", "terrible service", "place closed", "horrible service"],
           2: ["good food", "just ok", "good service", "customer service", "food good", "food ok"],
           3: ["service good", "good service", "pretty good", "good food", "food good", "food ok"],
           4: ["food great", "great service", "food good", "great food", "good food", "good service"],
           5: ["highly recommend", "food great", "love place", "great food", "great service", "service great"]}

TRIGRAMS = {1: ["don't waste money", "worst customer service", "horrible customer service", "don't' waste time", "ow ow ow", \
                "poor customer service"],
            2: ["food great service", "food mediocre best", "food good service", "service good food", "food just ok", \
                "meh ve experienced", "ve experienced better"],
            3: ["food pretty good", "food just ok", "service good food", "good food good", "food good service"],
            4: ["good food great", "good food good", "food good service", "great food great", "food great service"],
            5: ["love love love", "great service great", "great customer service", "food great service", "great food great"]}

REVIEW_IDS = set()
with open("../annotated_reviews/annotated_reviews_compiled.json", 'r') as f:
    for line in f:
        review = json.loads(line)
        review_id = review["review_id"]
        REVIEW_IDS.add(review_id)

print REVIEW_IDS

#-------------------------------------------------------------------------------------------------------------
def filter_data(source_file):

    summary_sentences = set()
    with open(source_file, 'r') as f:
        for line in f:
            #print line
            review = json.loads(line)
            if review["review_id"] in REVIEW_IDS:
                try:
                    text = review["text"]
                    label = review["stars"]
                    sentences = sent_detector.tokenize(text)
                    print label, '\n', text

                    features = UNIGRAMS[label] + BIGRAMS[label] + TRIGRAMS[label]
                    for sentence in sentences:
                        for feature in features:
                            if feature in sentence:
                                print "Feature:", feature
                                print "Extracted sentence:", sentence
                                summary_sentences.add((review["review_id"], label, sentence))

                except:
                    continue

    print summary_sentences
    return summary_sentences

#-------------------------------------------------------------------------------------------------------------
def write_to_json(summary_sentences, target_file):

    with open(target_file, 'w') as outfile:
        for r in summary_sentences:
            outfile.write(json.dumps(r))
            outfile.write("\n")


#-------------------------------------------------------------------------------------------------------------
def main():

    source_file = "../yelp_data/yelp_academic_dataset_review.json"
    target_file = "../annotated_reviews/extracted_sentences.json"
    summary_sentences = filter_data(source_file)
    #write_to_json(summary_sentences, target_file)


# -------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
