from typing import List, Dict
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, make_scorer
import utils



def cluster_name(numb_clus_samples: List[int], cat_samples: List[str],
				 clus_numb: int) -> Dict[int, str]:
	name_set = set(cat_samples)
	times = {}
	for i in range(clus_numb):
		for name in name_set:
			times[i, name] = 0

	for index, name in zip(numb_clus_samples, cat_samples):
		times[index, name] += 1

	clus_names = {
		i: sorted([(name, time) for (numb, name,), time in times.items()
				   if numb == i], reverse=True, key=lambda x: x[1])[0][0]
		for i in range(clus_numb)
	}
	print(clus_names)
	assert len(set(clus_names.values())) == clus_numb

	return clus_names


train_cats = [info['category'] for info in joblib.load('train_set')]
test_cats = [info['category'] for info in joblib.load('test_set')]

tfidf_uni_bigram_train = joblib.load('tfidf_uni_bigram_train')
tfidf_uni_bigram_test = joblib.load('tfidf_uni_bigram_test')


kmeans = KMeans(n_clusters=41, n_init=10, max_iter=500, precompute_distances=True,
				n_jobs=-1)
# rf = joblib.load('rf')
train_clus = kmeans.fit_predict(tfidf_uni_bigram_train, train_cats)
# pred_nb = kmeans.predict(tfidf_uni_bigram_test)

clus_names = cluster_name(train_clus, train_cats, 41)
print(clus_names)
# print(f1_score(test_cats, pred_nb, average='weighted'))
#
f1_scorer = make_scorer(f1_score, average='weighted')
# precision_scorer = make_scorer(metrics.precision_score)
#
# print(utils.mean_cross_score(naive_bayes, tfidf_uni_bigram_train, train_cats,
# 							 cv=5, n_jobs=-1, scoring=f1_scorer))

# joblib.dump(kmeans, 'kmeans')

# utils.plot_learning_curve(kmeans, 'kmeans', tfidf_uni_bigram_train,
# 						  train_cats, cv=5, n_jobs=-1, scorer=f1_scorer)


# todo ver con varias metricas para cada una de las 35 iteraciones
# todo ver la learning curve para el dataset que se tiene
# todo ver el cross validation para cada una de las 30 iteraciones
