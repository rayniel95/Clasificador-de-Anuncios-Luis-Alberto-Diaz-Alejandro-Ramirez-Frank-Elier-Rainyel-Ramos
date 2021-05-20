import joblib

import feature_vecs
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy.stats import expon
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics



# poner como svm a pesar de aumentar el numero de parametros no mejora
train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
	feature_vecs.do_uni_bigram()
print('b')
tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)

#
# train_cats = [info['category'] for info in joblib.load('train_set')]
# test_cats = [info['category'] for info in joblib.load('test_set')]
# print('b')
#
# tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
# tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')

print(tfidf_uni_bigram_train.shape)

# best_svm = joblib.load('svm')
svm_classifier = LinearSVC(max_iter=5000, dual=False)

param_space = {'C': expon(scale=0.1)}
f1_scorer = make_scorer(f1_score, average='weighted')

ran_search = RandomizedSearchCV(svm_classifier,
								param_distributions=param_space, n_iter=200,
								n_jobs=4, cv=5, scoring=f1_scorer)
ran_search.fit(tfidf_uni_bigram_train, train_cats)

best_svm = ran_search.best_estimator_

pred = best_svm.predict(tfidf_uni_bigram_test)
print(pred)
print(metrics.f1_score(test_cats, pred, average='weighted'))

# joblib.dump(best_svm, 'svm')

# todo plotear learning curve



