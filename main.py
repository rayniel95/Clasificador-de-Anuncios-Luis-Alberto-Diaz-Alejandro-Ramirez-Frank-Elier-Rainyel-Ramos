

# from typing import List
# from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
# from sklearn.dummy import DummyClassifier
# import utils



# def calssify(text: str, estimator: DummyClassifier, counter: CountVectorizer,
# 			 tfidf_matrix: TfidfTransformer) -> str:
# 	'''
# 	 clasifica el texto sucio acorde a las categorias de revolico
# 	:param text: raw text, not preprocesed
# 	:type text: str
# 	:param estimator: clasificador a usar
# 	:return: una categoria de revolico
# 	:rtype: str
# 	'''

# 	tok_text: List[str] = utils.preprocessing(text)
# 	count_vec = counter.transform([' '.join(tok_text)])
# 	tfidf_vec = tfidf_matrix.transform([count_vec])

# 	return estimator.predict([tfidf_vec])



if __name__ == '__main__':
	pass