from typing import List
import joblib
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.dummy import DummyClassifier
import utils



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

	tok_text: List[str] = utils.preprocessing(text)
	count_vec = counter.transform([' '.join(tok_text)])
	tfidf_vec = tfidf_matrix.transform(count_vec)

	return estimator.predict(tfidf_vec)[0]



if __name__ == '__main__':
	estimator = joblib.load('svm')
	counter = joblib.load('counter')
	tfidf_transformer = joblib.load('tfidf_transformer')

	print(classify('vendo pesas', estimator, counter, tfidf_transformer))