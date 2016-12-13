from pyrouge import Rouge155
import json
from nltk.tokenize import RegexpTokenizer
import nltk.data
import string
import numpy as np
tokenizer = RegexpTokenizer(r'\w+')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
uppercase = list(string.uppercase)
for i in string.uppercase:
	uppercase.append("A"+i)
NAME_BASE = "s"
system_dir = '../annotated_reviews/rouge_eval/system2'
model_dir = '../annotated_reviews/rouge_eval/reference2'
def write_summaries():

	# store narek's summaries
	summaries = []
	with open("dana_summaries.json") as f:
		for line in f:
			summaries.append(json.loads(line))

	id_to_sum = {}
	for summ in summaries:
		id_to_sum[summ["review_id"]] = summ["sentence"]

	reviews = []
	review_ids =[]
	with open("../annotated_reviews/annotated_reviews_compiled.json") as f:
		for line in f:
			reviews.append(json.loads(line))

	rev_num = 1
	for review in reviews:
		text = review["text"]
		review_id = review["review_id"]
		anns = review["annotations"]
		print sum(anns)
		sentences = sent_detector.tokenize(text.rstrip())
		sentences_new = []
		try:
			for i in xrange(len(sentences)):
				if anns[i] == 1:
					sentences_new.append(sentences[i])
		except:
			print sum(anns), len(sentences)
		sentences=sentences_new
		if review_id not in id_to_sum:
			continue
		task_name = "task"+str(rev_num)
		for i,sent in enumerate(sentences):
			
			f = open(model_dir+"/"+task_name+"_reference"+str(i+1)+".txt","w")
			f.write(sent)
			f.close()
		f = open(system_dir+"/"+task_name+"_system1.txt","w")
		f.write(id_to_sum[review_id])
		f.close()
		rev_num+=1

if __name__ == "__main__":
	write_summaries()
	# r = Rouge155()
	# r.system_dir = '../annotated_reviews/automated_summaries'
	# r.model_dir = '../annotated_reviews/model_summaries'
	# r.system_filename_pattern = NAME_BASE+'.(\d+).txt'
	# r.model_filename_pattern = NAME_BASE+'.[A-Z].#ID#.txt'

	# output = r.convert_and_evaluate()
	# print(output)
	# output_dict = r.output_to_dict(output)