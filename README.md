Fall 2016


Pickle files description:

word2vec.pkl
=dictionary that maps words from the reviews to 300d vectors

vocab_filtered.pkl
=dictionary that maps words to their index in vocabulary

vocab_inv_filtered.pkl
=array of words, inverse vocabulary, maps index to the word in vocab

train_mat_filtered.pkl
= [x,y,embedding_matrix], where 
x  
=list of lists of indices, each review - element in the outer list, 
each word from the review (or rather its index in vocab) - element in the inner list
y 
=labels (stars,1-5)
embedding_matrix
=2d array with shape (vocab_size, 300), where each row i - word2vec representation 
of word i from vocabulary
