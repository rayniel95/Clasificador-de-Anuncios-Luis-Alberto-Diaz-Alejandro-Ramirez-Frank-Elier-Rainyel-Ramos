import joblib
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer, precision_score, \
	accuracy_score, hamming_loss, jaccard_score, matthews_corrcoef, recall_score
from sklearn.neighbors import KNeighborsClassifier
import feature_vecs

# todo no se puede hacer crossvalidation en knn ni learning curve
# hablar de como aumenta la presicion de knn aumentando el numero de vecinos
# y como disminuye si se aumenta demasiado, co 15 76 con 30 80 con 100 solo 71
# de lo interesante que seria hacer una busqueda en el numero de vecinos con
# distribucion normal y variando el tamano del train set


# train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
# 	feature_vecs.do_uni_bigram()
#
# tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
# tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)
import utils

train_cats = [info['category'] for info in joblib.load('train_set')]
test_cats = [info['category'] for info in joblib.load('test_set')]


tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')
#
# # knn_classifier = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
knn = joblib.load('knn')
# # knn_classifier.fit(tfidf_uni_bigram_train, train_cats)
pred_nb = knn.predict(tfidf_uni_bigram_test)
# #
# # print(metrics.precision_score(test_cats, pred, average='weighted'))
# scorer = make_scorer(f1_score, average='weighted')
# # joblib.dump(knn_classifier, 'knn')
print(f1_score(test_cats, pred_nb, average='weighted'))
print(precision_score(test_cats, pred_nb, average='weighted'))
print(accuracy_score(test_cats, pred_nb))
print(hamming_loss(test_cats, pred_nb))
print(jaccard_score(test_cats, pred_nb, average='weighted'))
# val_test_log_loss = log_loss(test_cats, pred_nb)
print(matthews_corrcoef(test_cats, pred_nb))
print(recall_score(test_cats, pred_nb, average='weighted'))



def evaluate():
	train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
		feature_vecs.do_uni_bigram()

	tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
	tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)
	# tfidf_uni_bigram_train, tfidf_uni_bigram_test, train_cats, test_cats = feature_vecs.do_means_embeddings()

	knn_classifier = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)

	knn_classifier.fit(tfidf_uni_bigram_train, train_cats)
	pred_nb = knn_classifier.predict(tfidf_uni_bigram_test)

	val_test_f1 = f1_score(test_cats, pred_nb, average='weighted')
	val_test_precision = precision_score(test_cats, pred_nb, average='weighted')
	val_test_accuracy = accuracy_score(test_cats, pred_nb)
	val_test_hamming_loss = hamming_loss(test_cats, pred_nb)
	val_test_jaccard_score = jaccard_score(test_cats, pred_nb,
										   average='weighted')
	# val_test_log_loss = log_loss(test_cats, pred_nb)
	val_test_matthews_corrcoef = matthews_corrcoef(test_cats, pred_nb)
	val_test_recall_score = recall_score(test_cats, pred_nb, average='weighted')

	return val_test_f1, val_test_precision, val_test_accuracy, \
		   val_test_hamming_loss, val_test_jaccard_score, \
		   val_test_matthews_corrcoef, val_test_recall_score,


if __name__ == '__main__':
	# test_f1 = 0
	# test_precision = 0
	# test_accuracy = 0
	# test_hamming_loss = 0
	# test_jaccard_score = 0
	# test_matthews_corrcoef = 0
	# test_recall_score = 0
	#
	# for time in range(35):
	# 	results = evaluate()
	#
	# 	test_f1 += results[0]
	# 	test_precision += results[1]
	# 	test_accuracy += results[2]
	# 	test_hamming_loss += results[3]
	# 	test_jaccard_score += results[4]
	# 	test_matthews_corrcoef += results[5]
	# 	test_recall_score += results[6]
	#
	# print(f'f1 average: {test_f1 / 35}')
	# print(f'precision average: {test_precision / 35}')
	# print(f'accuracy average: {test_accuracy / 35}')
	# print(f'hamming_loss average: {test_hamming_loss / 35}')
	# print(f'jaccard_score average: {test_jaccard_score / 35}')
	# print(f'matthews_corrcoef average: {test_matthews_corrcoef / 35}')
	# print(f'recall_score average: {test_recall_score / 35}')
	pass