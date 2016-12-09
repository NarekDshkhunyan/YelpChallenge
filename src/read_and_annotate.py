import json
import nltk.data
from nltk.tokenize import RegexpTokenizer
import random
import sys

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer = RegexpTokenizer(r'\w+')

data_dir = "../yelp_data"
data_file = "big_yelp_academic_dataset_review_filtered.json"
out_file = "annotated_random_reviews.json"

NUM_SENTENCES_TO_ANNOTATE = 3

# Note:
#    set NUM_SENTENCES_TO_ANNOTATE to a small number and then merge the annotated reviews using a small script
#  the script only writes the annotations once everything has been annotated, so  you might lose work :(

def process_review(sentences_arr):
#     sentences = re.split(r"\.+\s*", review_text)
#     sentences = sent_detector.tokenize(review_text.strip())
    if len(sentences_arr) <10:
        return None
    
    sentences = [tokenizer.tokenize(s) for s in sentences_arr]
    sentences = [" ".join(s) for s in sentences]
    extracted=[]
    
    ratings = []
    for el in sentences:
        if len(el) >= 2:
            extracted.append(el)
    
    whole_review = ". ".join(sentences)
    print whole_review
    for sentence in extracted:
        print "Sentence: " , sentence
        rating = input("Rating: ")
        ratings.append(rating)
    return ratings
        
def random_review_indices(file_size):
        return set(random.sample(range(file_size), NUM_SENTENCES_TO_ANNOTATE))

if __name__ == '__main__':
    
    file_name = data_dir +'/'+ data_file
    source_json_file = open(file_name, "r")
    
    total_num_reviews = len([1 for line in source_json_file])
#     print num_reviews  #   1594893 in all
    visited = set([])
    annotated_reviews = []
    annotated_review_count = 0
    
    while len(annotated_reviews) < NUM_SENTENCES_TO_ANNOTATE:
        indices_to_check = random_review_indices(total_num_reviews)
        line_number = 0
        
        print "indices: " , indices_to_check
        
        with open(file_name) as source_json_file:
            for line in source_json_file:
                
                if (line_number not in visited) and (line_number in indices_to_check):
                    
                    visited.add(line_number)
                    review = json.loads(line)
                    text = review["text"].encode('utf-8')
                    annotations = process_review(sent_detector.tokenize(text))
                    if annotations:    
                        new_review = {}
                        new_review["line_number"] = line_number
                        new_review["annotations"] = annotations
                        new_review["stars"] = review["stars"]
                        new_review["text"] = review["text"]
                        new_review["review_id"] = review["review_id"]
                        
                        annotated_reviews.append(new_review)
                        annotated_review_count += 1
                        
                if line_number in visited:
                    print line_number, "already in", visited
                        
                if annotated_review_count == 200: break
                line_number += 1
                
        print "re-fetching more reviews. Annotated ", annotated_review_count, "so far."
        
    output_path = data_dir +'/'+ out_file
    with open(output_path, 'w') as outfile:
        for r in annotated_reviews:
            outfile.write(json.dumps(r))
            outfile.write("\n")
            
    print "ALL DONE!!!"
