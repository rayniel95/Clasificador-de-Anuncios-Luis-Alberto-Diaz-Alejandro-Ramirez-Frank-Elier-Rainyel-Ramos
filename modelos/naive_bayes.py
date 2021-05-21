import joblib
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn.naive_bayes import MultinomialNB
import feature_vecs
import utils



# poner como naive bayes al aumentar el numero de parametros desde los 5000
# palabras mas usadas a las 10000 mejoro casi en un 10 porciento

train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
	feature_vecs.do_uni_bigram()

tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)


# train_cats = [info['category'] for info in joblib.load('train_set')]
# test_cats = [info['category'] for info in joblib.load('test_set')]
# print('b')
#
# tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
# tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')


naive_bayes = MultinomialNB()

naive_bayes.fit(tfidf_uni_bigram_train, train_cats)
pred_nb = naive_bayes.predict(tfidf_uni_bigram_test)

print(metrics.precision_score(test_cats, pred_nb, average='weighted'))

f1_scorer = make_scorer(f1_score, average='weighted')
precision_scorer = make_scorer(metrics.precision_score)

print(utils.mean_cross_score(naive_bayes, tfidf_uni_bigram_train, train_cats,
							 cv=5, n_jobs=-1, scoring=f1_scorer))

# joblib.dump(naive_bayes, 'nb')
# todo plotear learning curve


