from typing import List, Dict, Tuple

import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import utils
import numpy as np



def ma_lo_data_set(path: str, n_words: int) -> Tuple[Dict, Dict, Dict[str, int]]:
	'''
	 make, load and return the dataset from a structured data folder
	:param n_words: most important words number
	:type n_words: int
	:param path: path to dataset
	:type path: str
	:return:
	:rtype:
	'''
	word_counter = utils.count_words(path)

	voc = utils.most_rep(word_counter, n_words)

	data = utils.load_data(path)
	np.random.shuffle(data)

	train_set, test_set = data[:-5000], data[-5000:]

	return train_set, test_set, voc


# todo nn con el promedio de los embeddings y con los embeddings directo


def do_counter() -> Tuple:
	'''
	unigram counter matrix with k most repetitive words
	:return: the counter matrix for train set and the counter matrix for test
	 set
	:rtype:
	'''
	train_set, test_set, voc = ma_lo_data_set((r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev'), 10000)

	counter_unigram = CountVectorizer()
	counter_unigram.fit(voc)
	unigram_matrix_train = counter_unigram.transform([' '.join(info['text'])
												for info in train_set])
	unigram_matrix_test = counter_unigram.transform([' '.join(info['text'])
												for info in test_set])

	return unigram_matrix_train, unigram_matrix_test, [info['category']
		   for info in train_set], [info['category'] for info in test_set]


def do_counter_whole():
	'''
	 unigram counter matrix with the whole vocabulary
	:return: the counter matrix for train set and the counter matrix for test
	 set
	:rtype:
	'''
	train_set, test_set, _ = ma_lo_data_set((r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev'), 10000)

	counter_unigram = CountVectorizer()
	unigram_matrix_train = counter_unigram.fit_transform([' '.join(info['text'])
												for info in train_set])
	unigram_matrix_test = counter_unigram.fit_transform([' '.join(info['text'])
												for info in test_set])
	return unigram_matrix_train, unigram_matrix_test, [info['category']
		   for info in train_set], [info['category'] for info in test_set]


def do_bigram_whole():
	'''
	 bigram counter matrix with whole vocabulary
	:return: the counter matrix for train set and the counter matrix for test
	 set
	:rtype:
	'''
	# note the little ocurrence probability of be together the most repetitive
	# words
	train_set, test_set, _ = ma_lo_data_set((r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev'), 10000)

	counter_bigram = CountVectorizer(ngram_range=(2, 2,))
	bigram_matrix_train = counter_bigram.fit_transform([' '.join(info['text'])
												for info in train_set])
	bigram_matrix_test = counter_bigram.fit_transform([' '.join(info['text'])
														for info in test_set])
	return bigram_matrix_train, bigram_matrix_test, [info['category']
		   for info in train_set], [info['category'] for info in test_set]


def do_uni_bigram():
	'''
	 unigram and bigram matrix with k most repetitive words
	:return: the counter matrix for train set and the counter matrix for test
	 set
	:rtype:
	'''
	train_set, test_set, voc = ma_lo_data_set((r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev'),10000)

	counter_uni_bigram = CountVectorizer(ngram_range=(1, 2,))
	counter_uni_bigram.fit(voc)
	uni_bigram_matrix_train = counter_uni_bigram.transform(
		[' '.join(info['text']) for info in train_set])
	uni_bigram_matrix_test = counter_uni_bigram.transform(
		[' '.join(info['text']) for info in test_set])

	return uni_bigram_matrix_train, uni_bigram_matrix_test, [info['category']
		   for info in train_set], [info['category'] for info in test_set]


def do_uni_bi_trigram():
	'''
	 unigram, bigram and trigrma matrix with k most repetitive words
	:return: the counter matrix for train set and the counter matrix for test
	 set
	:rtype:
	'''
	train_set, test_set, voc = ma_lo_data_set((r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev'), 10000)

	counter_uni_bi_trigram = CountVectorizer(ngram_range=(1, 3,))
	counter_uni_bi_trigram.fit(voc)
	uni_bi_trigram_matrix_train = counter_uni_bi_trigram.transform(
		[' '.join(info['text']) for info in train_set])
	uni_bi_trigram_matrix_test = counter_uni_bi_trigram.transform(
		[' '.join(info['text']) for info in test_set])

	return uni_bi_trigram_matrix_train, uni_bi_trigram_matrix_test, \
		   [info['category'] for info in train_set], [info['category']
													  for info in test_set]



# region embedding average representation with the whole vocabulary
# todo duda en como decirle a la capa q solo tenga en cuenta k palabras que no
#  sea quitando las n-k restantes de los docs
# index_words = {index: word for index, word in enumerate(word_counter)}
# max_count_words = len(word_counter) + 1
# max_length_seq = max(data, key=lambda x: len(x['text']))
# embedding_length = 20
#
# samples = [info['text'] for info in data]
# samples = utils.word_seq_transformer(samples, index_words)
#
# model = Sequential()
# model.add(Embedding(max_count_words, embedding_length,
# 					input_length=max_length_seq))
# model.compile()



# endregion


def tfidf_matrix(count_matrix):
	return TfidfTransformer().fit_transform(count_matrix)



if __name__ == '__main__':
	# salvando la data y los objetos necesarios para los clasificadores
	train_set, test_set, voc = ma_lo_data_set((r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev'),10000)

	counter_uni_bigram = CountVectorizer(ngram_range=(1, 2,))
	counter_uni_bigram.fit(voc)

	uni_bigram_matrix_train = counter_uni_bigram.transform(
		[' '.join(info['text']) for info in train_set])
	uni_bigram_matrix_test = counter_uni_bigram.transform(
		[' '.join(info['text']) for info in test_set])

	tfidf_mat = TfidfTransformer()
	tfidf_uni_bigram_train = tfidf_mat.fit_transform(uni_bigram_matrix_train,
										   [' '.join(info['text']) for info in
											train_set])
	tfidf_uni_bigram_test = tfidf_mat.transform(uni_bigram_matrix_test)


	joblib.dump(train_set, 'train_set')
	joblib.dump(test_set, 'test_set')
	joblib.dump(counter_uni_bigram, 'counter')
	joblib.dump(tfidf_mat, 'tfidf_transformer')
	joblib.dump(tfidf_uni_bigram_train, 'tfidf_uni_bigram_train')
	joblib.dump(tfidf_uni_bigram_test, 'tfidf_uni_bigram_test')





