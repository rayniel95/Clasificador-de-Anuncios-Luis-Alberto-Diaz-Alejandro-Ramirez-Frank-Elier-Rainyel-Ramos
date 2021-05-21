import json
import os
import pickle
import re
from functools import singledispatch
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scrapy.http import TextResponse
from scrapy.selector import Selector
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, learning_curve

# todo quitar duplicados en el dataset
# todo poner correctamente los anuncios hay algunos que estan mal clasificados


def preprocessing(text: str):
	text = re.sub(r'\W+|_+|-+|\d+', ' ', text.lower())

	# region se cambian las vocales con tilde por las vocales sin tilde
	text = re.sub('á', 'a', text)
	text = re.sub('é', 'e', text)
	text = re.sub('í', 'i', text)
	text = re.sub('ó', 'o', text)
	text = re.sub('ú', 'u', text)
	text = re.sub('ñ', 'n', text)
	# endregion
	# region se eliminan letras repetidas innecesarias
	text = re.sub('aa+', 'a', text)
	text = re.sub('bb+', 'b', text)
	text = re.sub('ccc+', 'c', text)
	text = re.sub('dd+', 'd', text)
	text = re.sub('ee+', 'e', text)
	text = re.sub('ff+', 'f', text)
	text = re.sub('gg+', 'g', text)
	text = re.sub('hh+', 'h', text)
	text = re.sub('ii+', 'i', text)
	text = re.sub('jj+', 'j', text)
	text = re.sub('kk+', 'k', text)
	text = re.sub('lll+', 'l', text)
	text = re.sub('mm+', 'm', text)
	text = re.sub('nnn+', 'n', text)
	text = re.sub('oo+', 'o', text)
	text = re.sub('pp+', 'p', text)
	text = re.sub('qq+', 'q', text)
	text = re.sub('rrr+', 'r', text)
	text = re.sub('ss+', 's', text)
	text = re.sub('tt+', 't', text)
	text = re.sub('uu+', 'u', text)
	text = re.sub('vv+', 'v', text)
	text = re.sub('ww+', 'w', text)
	text = re.sub('xx+', 'x', text)
	text = re.sub('yy+', 'y', text)
	text = re.sub('zz+', 'z', text)
	# endregion

	tokenizer = RegexpTokenizer(r'\w+')
	tokenized_words = tokenizer.tokenize(text)
	tokenized_words = [word for word in tokenized_words
					   if word not in stopwords.words('spanish')]
	# el lemmatizer rompe algunas palabras
	lemmatizer = WordNetLemmatizer()
	tokenized_words = [lemmatizer.lemmatize(word) for word in tokenized_words]
	tokenized_words = [word for word in tokenized_words if len(word) > 2]

	return tokenized_words


def extract_text_sel(selector: Selector) -> Tuple[str, str]:
	title = ' '.join(sel.get()
					 for sel in selector.css(
		'div[id=adwrap] h1[class=headingText]::text'))

	body = ' '.join(sel.get()
					for sel in selector.css(
		'span[class=showAdText]::text'))

	return title, body


@singledispatch
def extract_text_code(html):
	raise TypeError('type not supported')


@extract_text_code.register
def _(html: str):
	return extract_text_sel(Selector(text=html))


@extract_text_code.register
def _(html: TextResponse):
	return extract_text_sel(Selector(response=html))


def create_index(path_pages: str, path_json: str, category: str):
	errors: List = []
	file: os.DirEntry
	for file in os.scandir(path_pages):
		if file.is_file() and not file.name.startswith('.') and \
				file.name.endswith('.html'):
			with open(file) as page:
				# caso de que halla error de coding
				try:
					info = extract_text_code(page.read())
				except Exception as e:
					errors.append(e)
					continue

			prepro_text = preprocessing(' '.join(info))

			with open(f'{path_json}{os.sep}{file.name}.json', 'w') as jfile:
				json.dump({'category': category, 'text': prepro_text}, jfile)

	print(errors, len(errors))


def create_indexes():
	folder: os.DirEntry
	for folder in os.scandir(r'C:\Users\LsW\Desktop\IA Algorithms\Refer\rev'):
		if folder.is_dir():
			create_index(folder.path,
						 r'C:\Users\LsW\Desktop\IA Algorithms\Refer\jrev',
						 folder.name)


def count_words(json_path: str) -> Dict[str, int]:
	counter: Dict[str, int] = {}
	file: os.DirEntry
	for file in os.scandir(json_path):
		if file.is_file() and not file.name.startswith('.') and \
			file.name.endswith('.json'):

			with open(file) as jfile:
				info: Dict[str, str] = json.load(jfile)
			for word in info['text']:
				try:
					counter[word] += 1
				except KeyError:
					counter[word] = 1

	return counter


def most_rep(words: Dict[str, int], count: int):
	return [tup[0]
			for tup in sorted(list(words.items()), key=lambda x: x[1],
							  reverse=True)[:count]]


def load_data(json_path: str):
	info: List[Dict[str, str]] = []
	file: os.DirEntry
	for file in os.scandir(json_path):
		if file.is_file() and not file.name.startswith('.') and \
				file.name.endswith('.json'):
			with open(file) as jfile:
				info.append(json.load(jfile))
	return info


def word_seq_transformer(samples: List[List[str]],
						 words_index: Dict[str, int]) -> List[List[int]]:
	'''
	Transform sequences of words to integer sequences
	'''
	return [[words_index[word] for word in sample] for sample in samples]


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
	"""
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
	plt.figure()
	plt.title(title)

	if ylim is not None:
		plt.ylim(*ylim)

	plt.xlabel("Training examples")
	plt.ylabel("Score")

	train_sizes, train_scores, test_scores = learning_curve(
		estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	plt.grid()

	plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
					 train_scores_mean + train_scores_std, alpha=0.1,
					 color="r")
	plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
					 test_scores_mean + test_scores_std, alpha=0.1, color="g")
	plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
			 label="Training score")
	plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
			 label="Cross-validation score")

	plt.legend(loc="best")
	return plt


def mean_cross_score(estimator, X, y=None, groups=None, scoring=None,
					 cv='warn', n_jobs=None, verbose=0, fit_params=None,
					 pre_dispatch='2*n_jobs', error_score='raise-deprecating'):

	scores: np.ndarray
	scores = cross_val_score(estimator=estimator, X=X, y=y, groups=groups,
						   scoring=scoring, cv=cv, n_jobs=n_jobs, 
						   verbose=verbose, fit_params=fit_params, 
						   pre_dispatch=pre_dispatch, error_score=error_score)
	return scores.mean()

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

def get_properties(dsp, save_on_disk = True):
	categories = set()
	vocab = set()
	MAX_SEQUENCE = 0

	for f in os.scandir(dsp):
		if f.is_file() and f.name.endswith('.json'):
			with open(f) as fd:
				js = json.load(fd)
			categories.add(js['category'])
			for w in js['text']:
				vocab.add(w)
			MAX_SEQUENCE = max(MAX_SEQUENCE, len(js['text']))

	properties = {'categories': categories, 'vocab': vocab, 'max_sequence': MAX_SEQUENCE}

	if save_on_disk:
		print('saving on disk...')
		f = open('./dataset_properties', 'wb')
		pickle.dump(properties, f)
		f.close()

	return properties
def to_tags(vectors, tags_index: dict):
	vektor = list(vectors)
	tags = []
	index2tags = {value: key for key, value in tags_index.items()}
	for ad in vektor:
		tags.append(index2tags[list(ad).index(1.0)])

	return tags


def to_binary(matrix):
	result = []

	for vec in list(matrix):
		max = vec.max()
		index_max = list(vec).index(max)

		new_vec = np.zeros(len(vec))
		new_vec[index_max] = 1
		result.append(new_vec)

	return result

if __name__ == '__main__':
	get_properties('/home/luis/Desktop/SI/jrev')