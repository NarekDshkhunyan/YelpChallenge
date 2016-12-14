Fall 2016

A Python pipeline for multilabel classification and extractive summarization of Yelp reviews

<b>Preprocessing</b>
-word2vec.py Implements the popular word2vec word embeddings per Mikolov
-filtering_script.py Filters the reviews where all words belong to the Google News dataset
-spellchecker.py Corrects spell checking errors

<b>Classification</b>

-classifer.py Implements multiple popular classifers such as Multinomial Naive Nayes, SVM, Random Forest, etc. Cross-Validation with test sample size at 10%.
-keras_sequential.py RNN LSTM classifier written in Keras
-simplified_tensorflow.py Uni- and Bidirectional Dynamic RNN written in Tensorflow

<b>Sentence Extraction</b> 
-extract_sentences.py Select sentences in reviews which contain the most important features per Naive Bayes
-simplified_tensorflow_regressor.py Predicts a label for each sentence in the training phase,
                                    then in testing phase selects sentences who predicted labels was closes to the actual label of the review

<b>Evauation</b>
-read_and_annotate.py manually annotate each sentence as relevant or not to obtain gold annotations
-rouge.py Implements ROUGE-N metric for evaluation of the summary sentences


Pickle files description:

word2vec.pkl<br />
=dictionary that maps words from the reviews to 300d vectors

word2vec_big.pkl<br />
=the same dictionary as above, but obtained 1.M reviews

vocab_filtered.pkl<br />
=dictionary that maps words to their index in vocabulary

vocab_filtered_big.pkl<br />
=the same as above, but obtained 1.M reviews

vocab_inv_filtered.pkl<br />
=array of words, inverse vocabulary, maps index to the word in vocab

vocab_inv_filtered.pkl<br />
=the same as above, but obtained 1.M reviews

train_mat_filtered.pkl<br />
= [x,y,embedding_matrix], where <br />
x  <br />
=list of lists of indices, each review - element in the outer list, 
each word from the review (or rather its index in vocab) - element in the inner list;<br />
y <br />
=labels (stars,1-5);<br />
embedding_matrix <br />
=2d array with shape (vocab_size, 300), where each row i - word2vec representation 
of word i from vocabulary;

train_mat_filtered_big.pkl<br />
=the same as above, but obtained 1.M reviews
