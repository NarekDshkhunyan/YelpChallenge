import json

file_name="../../yelp_data/yelp_academic_dataset_review.json"
file_write = open('stats.text', 'w')

sentences = []
characters = []
words = []
with open(file_name) as file:
	i = 0
	summ = 0
	for line in file:
		# print line
		entry = json.loads(line)
		
		numSent = 0
		for s in entry["text"].split("."):
			if len(s)!= 0:
				numSent += 1
		sentences.append(numSent)
		words.append(len(entry["text"].split(" ")))
		characters.append(len(entry["text"]))

words.sort()
characters.sort()
sentences.sort()

print characters[-1]

file_write.write("characters\n")
file_write.write("\tmean: " + str(sum(characters) / len(characters)) + "\n")
file_write.write ("\t25th percentile: " + str(characters[len(characters) / 4]) + "\n")
file_write.write ("\t50th percentile: " + str(characters[len(characters) / 2]) + "\n")
file_write.write ("\t75th percentile: " + str(characters[len(characters) * 3 / 4]) + "\n")

file_write.write("words\n")
file_write.write ("\tmean: " + str(sum(words) / len(words)) + "\n")
file_write.write ("\t25th percentile:: " + str(words[len(words) / 4]) + "\n")
file_write.write ("\t50th percentile:: " + str(words[len(words) / 2]) + "\n")
file_write.write ("\t75thth percentile:: " + str(words[len(words) * 3 / 4]) + "\n")

file_write.write("sentences\n")
file_write.write ("\tmean: " + str(sum(sentences) / len(sentences)) + "\n")
file_write.write ("\t25th percentile:: " + str(sentences[len(sentences) / 4]) + "\n")
file_write.write ("\t50th percentile:: " + str(sentences[len(sentences) / 2]) + "\n")
file_write.write ("\t75th percentile:: " + str(sentences[len(sentences) * 3 / 4]) + "\n")

file_write.close()
