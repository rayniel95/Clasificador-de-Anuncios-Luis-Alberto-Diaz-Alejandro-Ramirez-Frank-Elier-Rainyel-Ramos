import joblib
from sklearn import metrics
from sklearn.metrics import f1_score, make_scorer, precision_score, \
	accuracy_score, hamming_loss, jaccard_score, matthews_corrcoef, recall_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import feature_vecs
import utils



# train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
# 	feature_vecs.do_uni_bigram()
#
# tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
# tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)

train_cats = [info['category'] for info in joblib.load('train_set')]
test_cats = [info['category'] for info in joblib.load('test_set')]
#
tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')

#
# dt_classifier = DecisionTreeClassifier()
dt_classifier = joblib.load('dt')
# dt_classifier.fit(tfidf_uni_bigram_train, train_cats)
# pred_nb = dt_classifier.predict(tfidf_uni_bigram_test)

# #
# # print(metrics.precision_score(test_cats, pred, average='weighted'))
# #
# #
# f1_scorer = make_scorer(f1_score, average='weighted')
# #
# # print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# # 							 cv=5, n_jobs=-1, scoring=f1_scorer))
#
# # joblib.dump(dt_classifier, 'dt')
#
#
# utils.plot_learning_curve(dt_classifier, 'dt',
# 						  tfidf_uni_bigram_train, train_cats, cv=5, n_jobs=-1,
# 						  scorer=f1_scorer)

# muestra falta de datos


# print(f1_score(test_cats, pred_nb, average='weighted'))
# print(precision_score(test_cats, pred_nb, average='weighted'))
# print(accuracy_score(test_cats, pred_nb))
# print(hamming_loss(test_cats, pred_nb))
# print(jaccard_score(test_cats, pred_nb, average='weighted'))
# # val_test_log_loss = log_loss(test_cats, pred_nb)
# print(matthews_corrcoef(test_cats, pred_nb))
# print(recall_score(test_cats, pred_nb, average='weighted'))

f1_scorer = make_scorer(f1_score, average='weighted')
precision_scorer = make_scorer(precision_score, average='weighted')
accuracy_scorer = make_scorer(accuracy_score)
hamming_losser = make_scorer(hamming_loss)
jaccard_scorer = make_scorer(jaccard_score, average='weighted')
matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)
recall_scorer = make_scorer(recall_score, average='weighted')

# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=f1_scorer))
# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=precision_scorer))
# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=accuracy_scorer))
# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=hamming_losser))
# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=jaccard_scorer))
# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=matthews_corrcoef_scorer))
# print(utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=recall_scorer))

# utils.plot_learning_curve(dt_classifier, 'decision tree f1 score',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=f1_scorer)
# utils.plot_learning_curve(dt_classifier, 'decision tree precision score',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=precision_scorer)
utils.plot_learning_curve(dt_classifier, 'decision tree accuracy score',
						  tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1, scorer=accuracy_scorer)
utils.plot_learning_curve(dt_classifier, 'decision tree hamming loss',
						  tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1, scorer=hamming_losser)
utils.plot_learning_curve(dt_classifier, 'decision tree jaccard score',
						  tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1, scorer=jaccard_scorer)
utils.plot_learning_curve(dt_classifier, 'decision tree matthews corr coef scorer', tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1,
						  scorer=matthews_corrcoef_scorer)
utils.plot_learning_curve(dt_classifier, 'decision tree recall score',
						  tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1, scorer=recall_scorer)



def evaluate():
	train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
		feature_vecs.do_uni_bigram()

	tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
	tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)

	# tfidf_uni_bigram_train, tfidf_uni_bigram_test, train_cats, test_cats = feature_vecs.do_means_embeddings()


	dt_classifier = DecisionTreeClassifier()

	dt_classifier.fit(tfidf_uni_bigram_train, train_cats)
	pred_nb = dt_classifier.predict(tfidf_uni_bigram_test)

	# val_test = metrics.f1_score(test_cats, pred, average='weighted')

	# f1_scorer = make_scorer(f1_score, average='weighted')

	# val_cross = utils.mean_cross_score(dt_classifier, tfidf_uni_bigram_train,
	# 							 train_cats, cv=5, n_jobs=-1, scoring=f1_scorer)

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
	# todo hacer el mismo analizis pero para la matriz de frecuencia sola
	print('***************************************')