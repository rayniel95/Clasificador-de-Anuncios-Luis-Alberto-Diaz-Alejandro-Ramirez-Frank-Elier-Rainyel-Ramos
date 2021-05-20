import joblib
from utils import preprocessing, to_binary, to_tags
import pickle
from nn import cnn, rnn
from feature_vecs import compute_input, read_output
import numpy as np
from keras_preprocessing.sequence import pad_sequences
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from typing import Dict, List, Tuple, Union

def classify(text: str, estimator: DummyClassifier, counter: CountVectorizer,
			 tfidf_matrix: TfidfTransformer) -> str:
	'''
	 clasifica el texto sucio acorde a las categorias de revolico
	:param text: raw text, not preprocesed
	:type text: str
	:param estimator: clasificador a usar
	:return: una categoria de revolico
	:rtype: str
	'''

	tok_text: List[str] = preprocessing(text)
	count_vec = counter.transform([' '.join(tok_text)])
	tfidf_vec = tfidf_matrix.transform(count_vec)

	return estimator.predict(tfidf_vec)

def classify_cnn(model_path, text, dp, t = 0):
	f = open(dp, 'rb')
	p = pickle.load(f)
	f.close()

	v = list(p['vocab'])
	c = list(p['categories'])

	v.sort()
	c.sort()

	v = { w : i + 1 for i, w in  enumerate(v)}
	c = { w : i for i, w in enumerate(c)}
	ms = p['max_sequence']
	ms = 10 * int(ms / 10) + 10

	model = cnn((ms,), len(v) + 1, ms, pretrained_weights=model_path, t = t)

	x = preprocessing(text)
	x = compute_input(x, v, ms)
	x = np.reshape(x, (1, 5430))


	y = model.predict_on_batch(x)

	return read_output(y, c)

def classify_rnn1(model, text: List[str]) -> str:
	word2index = joblib.load('./word2index')
	tag2index = joblib.load('./tag2index')

	t = preprocessing(text)

	new_sent = [word2index[word] for word in t]
	max_len = 5423
	x = pad_sequences(truncating='post', maxlen=max_len, sequences=[new_sent],
					  padding='post', value=0)

	cats = model.predict(x)

	pred = to_binary(cats)
	pred2 = to_tags(pred, tag2index)

	return pred2

def classify_rnn2(model_path, text, dp):
	f = open(dp, 'rb')
	p = pickle.load(f)
	f.close()

	v = list(p['vocab'])
	c = list(p['categories'])

	v.sort()
	c.sort()

	v = { w : i + 1 for i, w in  enumerate(v)}
	c = { w : i for i, w in enumerate(c)}
	ms = p['max_sequence']
	ms = 10 * int(ms / 10) + 10

	model = rnn((ms,), len(v) + 1, ms, pretrained_weights=model_path)

	x = preprocessing(text)
	x = compute_input(x, v, ms)
	x = np.reshape(x, (1, 5430))


	y = model.predict_on_batch(x)

	return read_output(y, c)