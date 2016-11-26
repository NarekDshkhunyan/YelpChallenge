Fall 2016


Pickle files description:

word2vec.pkl<br />
=dictionary that maps words from the reviews to 300d vectors

vocab_filtered.pkl<br />
=dictionary that maps words to their index in vocabulary

vocab_inv_filtered.pkl<br />
=array of words, inverse vocabulary, maps index to the word in vocab

train_mat_filtered.pkl<br />
= [x,y,embedding_matrix], where <br />
x  <br />
=list of lists of indices, each review - element in the outer list, 
each word from the review (or rather its index in vocab) - element in the inner list;
y <br />
=labels (stars,1-5);
embedding_matrix <br />
=2d array with shape (vocab_size, 300), where each row i - word2vec representation 
of word i from vocabulary;
