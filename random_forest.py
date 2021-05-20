import joblib
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, \
	accuracy_score, fbeta_score, hamming_loss, jaccard_score, log_loss, \
	matthews_corrcoef, recall_score

import feature_vecs
import utils




train_cats = [info['category'] for info in joblib.load('train_set')]
test_cats = [info['category'] for info in joblib.load('test_set')]

tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')


# rf = RandomForestClassifier(n_jobs=-1)
rf = joblib.load('rf')
# rf.fit(tfidf_uni_bigram_train, train_cats)
# pred_nb = rf.predict(tfidf_uni_bigram_test)
#
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


# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=f1_scorer))
# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=precision_scorer))
# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=accuracy_scorer))
# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=hamming_losser))
# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=jaccard_scorer))
# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=matthews_corrcoef_scorer))
# print(utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=recall_scorer))


# utils.plot_learning_curve(rf, 'random forest f1 score', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=f1_scorer)
# utils.plot_learning_curve(rf, 'random forest precision score', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=precision_scorer)
# utils.plot_learning_curve(rf, 'random forest accuracy score', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=accuracy_scorer)
# utils.plot_learning_curve(rf, 'random forest hamming loss', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=hamming_losser)
# utils.plot_learning_curve(rf, 'random forest jaccard score', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=jaccard_scorer)
utils.plot_learning_curve(rf, 'random forest matthews corr coef scorer', tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1,
						  scorer=matthews_corrcoef_scorer)
utils.plot_learning_curve(rf, 'random forest recall score', tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1, scorer=recall_scorer)


def evaluate():
	train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
		feature_vecs.do_uni_bigram()

	tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
	tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)

	# tfidf_uni_bigram_train, tfidf_uni_bigram_test, train_cats, test_cats = feature_vecs.do_means_embeddings()

	rf = RandomForestClassifier(n_jobs=-1)

	rf.fit(tfidf_uni_bigram_train, train_cats)
	pred_nb = rf.predict(tfidf_uni_bigram_test)

	val_test_f1 = f1_score(test_cats, pred_nb, average='weighted')
	val_test_precision = precision_score(test_cats, pred_nb, average='weighted')
	val_test_accuracy = accuracy_score(test_cats, pred_nb)
	val_test_hamming_loss = hamming_loss(test_cats, pred_nb)
	val_test_jaccard_score = jaccard_score(test_cats, pred_nb, average='weighted')
	# val_test_log_loss = log_loss(test_cats, pred_nb)
	val_test_matthews_corrcoef = matthews_corrcoef(test_cats, pred_nb)
	val_test_recall_score = recall_score(test_cats, pred_nb, average='weighted')

	# f1_scorer = make_scorer(f1_score, average='weighted')
	# precision_scorer = make_scorer(precision_score, average='weighted')
	# accuracy_scorer = make_scorer(accuracy_score)
	# recall_scorer = make_scorer(recall_score, average='weighted')
	# jaccard_scorer = make_scorer(jaccard_score, average='weighted')
	# hamming_loss_looser = make_scorer(hamming_loss)
	# log_loss_looser = make_scorer(log_loss)
	# matthews_corrcoef_scorer = make_scorer(matthews_corrcoef)
	#
	# val_cross_f1 = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=f1_scorer)
	# val_cross_precision = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=precision_scorer)
	# val_cross_accuracy = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=accuracy_scorer)
	# val_cross_hamming_loss = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=hamming_loss_looser)
	# val_cross_jaccard_score = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=jaccard_scorer)
	# val_cross_log_loss = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=log_loss_looser)
	# val_cross_matthews_corrcoef = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=matthews_corrcoef_scorer)
	# val_cross_recall_score = utils.mean_cross_score(rf, tfidf_uni_bigram_train, train_cats,
	# 							 cv=5, n_jobs=-1, scoring=recall_scorer)

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

	print('***************************************')
# todo intentar hacer pca o manifold para reducir dimensiones y ver si funciona
#  el clustering

# todo poner en el informe que se trato con kmeans pero no dio resultados, poner
	# tambien que se trato con el promedio de los embeddings pero no se llegaron
	# a buenos resultados en ninguno de los clasificadores

	pass