import joblib
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
import feature_vecs


# hablar de como aumenta la presicion de knn aumentando el numero de vecinos
# y como disminuye si se aumenta demasiado, co 15 76 con 30 80 con 100 solo 71
# de lo interesante que seria hacer una busqueda en el numero de vecinos con
# distribucion normal y variando el tamano del train set


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

knn_classifier = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)

knn_classifier.fit(tfidf_uni_bigram_train, train_cats)
pred = knn_classifier.predict(tfidf_uni_bigram_test)

print(metrics.precision_score(test_cats, pred, average='weighted'))

# joblib.dump(knn_classifier, 'knn')
# todo plotear learning curve