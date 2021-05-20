import joblib
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
import feature_vecs
import utils



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


dt_classifier = DecisionTreeClassifier()

dt_classifier.fit(tfidf_uni_bigram_train, train_cats)
pred = dt_classifier.predict(tfidf_uni_bigram_test)

print(metrics.precision_score(test_cats, pred, average='weighted'))


f1_scorer = make_scorer(f1_score, average='weighted')

print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
							 cv=5, n_jobs=-1, scoring=f1_scorer))

# joblib.dump(dt_classifier, 'dt')

# todo plotear learning curve