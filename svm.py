import joblib

import feature_vecs
from sklearn.svm import LinearSVC
from sklearn import metrics
from scipy.stats import expon
from sklearn.metrics import make_scorer, f1_score, precision_score, \
	accuracy_score, hamming_loss, jaccard_score, matthews_corrcoef, recall_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics



# poner como svm a pesar de aumentar el numero de parametros no mejora
import utils

# train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
# 	feature_vecs.do_uni_bigram()
# print('b')
# tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
# tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)

#
train_cats = [info['category'] for info in joblib.load('train_set')]
test_cats = [info['category'] for info in joblib.load('test_set')]

tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')

# print(tfidf_uni_bigram_train.shape)

best_svm = joblib.load('svm')
# svm_classifier = LinearSVC(max_iter=5000, dual=False)

# param_space = {'C': expon(scale=0.1)}
# f1_scorer = make_scorer(f1_score, average='weighted')
#
# ran_search = RandomizedSearchCV(svm_classifier,
# 								param_distributions=param_space, n_iter=200,
# 								n_jobs=4, cv=5, scoring=f1_scorer)
# ran_search.fit(tfidf_uni_bigram_train, train_cats)
#
# best_svm = ran_search.best_estimator_
#
# pred_nb = best_svm.predict(tfidf_uni_bigram_test)
# print(pred)
# print(metrics.f1_score(test_cats, pred, average='weighted'))
# scorer = make_scorer(f1_score, average='weighted')
# # joblib.dump(best_svm, 'svm')
#
# utils.plot_learning_curve(best_svm, 'svm', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=scorer)

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

# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=f1_scorer))
# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=precision_scorer))
# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=accuracy_scorer))
# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=hamming_losser))
# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=jaccard_scorer))
# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=matthews_corrcoef_scorer))
# print(utils.mean_cross_score(best_svm, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=recall_scorer))

# utils.plot_learning_curve(best_svm, 'best svm f1 score',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=f1_scorer)
# utils.plot_learning_curve(best_svm, 'best svm precision score',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=precision_scorer)
# utils.plot_learning_curve(best_svm, 'best svm accuracy score',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=accuracy_scorer)
# utils.plot_learning_curve(best_svm, 'best svm hamming loss',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=hamming_losser)
# utils.plot_learning_curve(best_svm, 'best svm jaccard score',
# 						  tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=jaccard_scorer)
# utils.plot_learning_curve(best_svm, 'best svm matthews corr coef scorer', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1,
# 						  scorer=matthews_corrcoef_scorer)
utils.plot_learning_curve(best_svm, 'best svm recall score',
						  tfidf_uni_bigram_train,
						  train_cats, cv=5, n_jobs=-1, scorer=recall_scorer)




def evaluate():
	train_uni_bigram, test_uni_bigram, train_cats, test_cats = \
		feature_vecs.do_uni_bigram()

	tfidf_uni_bigram_train = feature_vecs.tfidf_matrix(train_uni_bigram)
	tfidf_uni_bigram_test = feature_vecs.tfidf_matrix(test_uni_bigram)
	# tfidf_uni_bigram_train, tfidf_uni_bigram_test, train_cats, test_cats = feature_vecs.do_means_embeddings()

	svm_classifier = LinearSVC(max_iter=5000, dual=False)

	svm_classifier.fit(tfidf_uni_bigram_train, train_cats)

	pred_nb = svm_classifier.predict(tfidf_uni_bigram_test)

	# val_test = metrics.f1_score(test_cats, pred, average='weighted')
	#
	# f1_scorer = make_scorer(f1_score, average='weighted')
	#
	# val_cross = utils.mean_cross_score(svm_classifier, tfidf_uni_bigram_train,
	# 								   train_cats, cv=5,
	# 								   scoring=f1_scorer)
	#
	# return val_cross, val_test

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



# todo intentar hacer un adaboost
# todo hacer todo de nuevo para el dataset limpio
# todo si queda tiempo el cross validation para cada una de las 35 iteraciones
	# con las distintas metricas
	pass