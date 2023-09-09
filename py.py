import numpy as np
import sklearn
import zipml08
import sklearn.neural_network
import sklearn.ensemble
import sklearn.cluster
import sklearn.manifold
import sklearn.pipeline
import sklearn.feature_extraction
import sklearn.feature_extraction.text
import sklearn.decomposition
import sklearn.metrics
zp = zipml08.ZPool(zipml08.ZPool.load_reuters(), dedup=True)
zp.train_test_split(0.9)
Xmv = np.load("Xmv_zl08_20230906.npy")
Y = np.load("Y_zl08_20230906.npy")
import collections
xd = collections.OrderedDict()
for i in range(Y.shape[1]):
	tup = np.zeros(Y.shape[1])
	tup[i] = 1
	xd[tuple(tup)] = i
len(xd)
for i in range(Y.shape[0]):
	if tuple(Y[i]) not in xd:
		xd[tuple(Y[i])] = len(xd)
len(xd)
pY = np.zeros((Y.shape[0], len(xd)))
pY.shape
for i in range(Y.shape[0]):
	pY[i, 0:Y.shape[1]] = Y[i]
	pY[i, xd[tuple(Y[i])]] = 1.0
kx = sklearn.manifold.TSNE(n_components=3)
ttt = kx.fit_transform(pY.transpose())
ttt[0]
ttt[5]
ttt[6]
ttt.shape
km = sklearn.cluster.KMeans(n_clusters=20)
km.fit(ttt)
km = sklearn.cluster.KMeans(n_clusters=20, n_init='auto')
km.fit(ttt)
catbag = {}
for i in range(len(zipml08.ZText.CATS)):
	catbag[km.labels_[i]] = catbag.get(km.labels_[i], [])
	catbag[km.labels_[i]].append(list(zipml08.ZText.CATS.keys())[i])
from sklearn.compose import TransformedTargetRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from pprint import pprint
pprint(catbag)
tY = np.array((pY.shape[0], ttt.shape[1]))
Y.shape ; pY.shape ; tY.shape
ttt.shape
tY = np.zeros((pY.shape[0], ttt.shape[1]))
Y.shape ; pY.shape ; tY.shape
for i in range(tY.shape[0]):
	tY[i] = ttt[xd[tuple(Y[i])]]
np.where(Y[0] == 1)
np.where(Y[2] == 1)
np.where(Y[1] == 1)
np.where(Y[3] == 1)
np.where(Y[4] == 1)
np.where(Y[5] == 1)
tY[0]
tY[3]
np.where(Y[6] == 1)
np.where(Y[7] == 1)
np.where(Y[8] == 1)
np.where(Y[9] == 1)
np.where(Y[10] == 1)
np.where(Y[11] == 1)
tY[9]
ttt[2]
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), regressor)
X_train = Xmv[:len(zp.pool), 0:-2]
X_test = Xmv[len(zp.pool):, 0:-2]
tY_train = tY[:len(zp.pool)]
tY_test = tY[len(zp.pool):]
X_train.shape ; X_test.shape ; tY_train.shape ; tY_test.shape
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
np.where(Y[len(zp.pool) + 0] == 1)
pipe.predict(tY_test)[0]
pipe.predict(X_test)[0]
np.argmin(((ttt - pipe.predict(X_test)[0]) ** 2.0).sum(axis=1))
np.argsort(((ttt - pipe.predict(X_test)[0]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - tY_test[0]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - tY_test[1]) ** 2.0).sum(axis=1))[0:5]
np.where(Y[len(zp.pool) + 1] == 1)
np.where(list(xd.keys())[468] == 1)
list(xd.keys())[468]
np.where(np.array(list(xd.keys())[468]) == 1)
np.argsort(((ttt - pipe.predict(X_test)[1]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - tY_test[2]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - pipe.predict(X_test)[2]) ** 2.0).sum(axis=1))[0:5]
regressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(32, 24, 32, 7), max_iter=500)
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), regressor)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
raw = [zt.s for zt in zp.pool+zp.testpool]
import json
jzd = json.load(open("zl08_ZData_20230906.json", 'r'))
jzd.keys()
raw = [ztd['s'] for ztd in jzd['zts_pool_testpool']]
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(analyzer='char', ngram_range=(2, 2))
vX = vectorizer.fit_transform(raw)
vX.shape
vX_train = vX[:len(zp.pool)]
vX_test = vX[len(zp.pool):]
vX.shape ; vX_train.shape ; vX_test.shape
regressor = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(32, 24, 32, 7), max_iter=700)
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.IncrementalPCA(n_components=800), regressor)
pipe.fit(vX_train, tY_train)
pipe.score(vX_train, tY_train)
pipe.score(vX_test, tY_test)
kx = sklearn.manifold.TSNE(n_components=11, method='exact')
ttt = kx.fit_transform(pY.transpose())
tY = np.zeros((pY.shape[0], ttt.shape[1]))
for i in range(tY.shape[0]):
	tY[i] = ttt[xd[tuple(Y[i])]]
ttt[2]
tY_train = tY[:len(zp.pool)]
tY_test = tY[len(zp.pool):]
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), regressor)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
np.argsort(((ttt - pipe.predict(X_test)[2]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - tY_test[2]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - pipe.predict(X_test)[0]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - tY_test[0]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - tY_test[1]) ** 2.0).sum(axis=1))[0:5]
np.argsort(((ttt - pipe.predict(X_test)[1]) ** 2.0).sum(axis=1))[0:5]
predY_ztrain = pipe.predict(X_train)
predY_ztest = pipe.predict(X_te)
predY_ztest = pipe.predict(X_test)
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.IncrementalPCA(n_components=800), regressor)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
predY_ztrain = pipe.predict(X_te)
predY_ztrain = pipe.predict(X_train)
predY_vtest = pipe.predict(X_test)
predY_vtrain = pipe.predict(X_train)
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), regressor)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
predY_ztrain = pipe.predict(X_train)
predY_ztest = pipe.predict(X_test)
corretto1 = 0
correttoany = 0
for i in range(Y_train.shape[0]):
	ccc = np.argsort(((ttt - tY_train[i]) ** 2.0).sum(axis=1))[0]
	ccp = np.argsort(((ttt - predY_ztrain) ** 2.0).sum(axis=1))[0:5]
	if ccc == ccp[0]:
		corretto1 += 1
	if ccc in ccp:
		correttoany += 1
informer = zipml08.RBMTransformer(n_components=500)
Y_train = Y[:len(zp.pool)]
informer.fit(Y_train)
corretto1 = 0
corretto5any = 0
corretto5re = 0
for i in range(predY_ztrain.shape[0]):
	ccc = np.argsort(((ttt - tY_train[i]) ** 2.0).sum(axis=1))[0:5]
	ccp = np.argsort(((ttt - predY_ztrain[i]) ** 2.0).sum(axis=1))[0:5]
	if ccc[0] == ccp[0]:
		correttoany += 1
	if ccc[0] in ccp:
		corretto1 += 1
def corretto(y_true, y_pred):
	corretto1 = 0
	corretto5any = 0
	corretto5re = 0
	for i in range(y_true.shape[0]):
		ccc = np.argsort(((ttt - y_true[i]) ** 2.0).sum(axis=1))[0:5]
		ccp = np.argsort(((ttt - y_pred[i]) ** 2.0).sum(axis=1))[0:5]
		if ccc[0] == ccp[0]:
			corretto1 += 1
		if ccc[0] in ccp:
			corretto5any += 1
		if ccp[0] in ccc:
			corretto5re += 1
	return (corretto1, corretto5any, corretto5re)
corretto(tY_train, predY_ztrain)
tY_train.shape
co_ztrain = corretto(tY_train, predY_ztrain)
co_ztrain ; np.array(co_ztrain) / tY_train.shape[0]
co_ztrain = corretto(tY_test, predY_ztest)
co_ztrain ; np.array(co_ztrain) / tY_test.shape[0]
co_vtrain = corretto(tY_train, predY_vtrain)
co_vtrain ; np.array(co_vtrain) / tY_train.shape[0]
co_vtest = corretto(tY_test, predY_vtest)
co_vtest ; np.array(co_vtest) / tY_test.shape[0]
Y_train = Y[:len(zp.pool)]
Y_test = Y[len(zp.pool):]
len(xd)
uniq, uniq_cnt = np.unique(Y_train, axis=0)
uniq, uniq_cnt = np.unique(Y_train, axis=0, return_counts=True)
len(uniq)
uniq_cnt[:15]
uniq, uniq_cnt = np.unique(Y, axis=0, return_counts=True)
len(uniq)
uniq.shape
uniq.dtype
bool.__doc__
leanY = np.array(Y, dtype=bool)
leanY.dtype
uniq, uniq_cnt = np.unique(leanY[:len(zp.pool)], axis=0, return_counts=True)
len(uniq)
uniq.dtype
uniq.shape
(uniq and leanY[0]).any
(uniq and leanY[0]).any()
uniq and leanY[0]
uniq & leanY[0]
(uniq & leanY[0]).shape
(uniq & leanY[0]).any
(uniq & leanY[0]).any()
(uniq & leanY[0]).any(axis=1)
(uniq & leanY[0]).any(axis=1).shape
np.where((uniq & leanY[0]).any(axis=1))
np.where(leanY[0])
np.where(uniq[304])
((uniq | leanY[0]) == leanY[0]).shape
((uniq | leanY[0]) == leanY[0]).any(axis=1).shape
((uniq | leanY[0]) == leanY[0]).any(axis=1)
((uniq | leanY[0]) == leanY[0]).all(axis=1)
np.where((uniq | leanY[0]) == leanY[0]).all(axis=1))
np.where(((uniq | leanY[0]) == leanY[0]).all(axis=1))
np.where(uniq[910])
np.where(uniq[915])
uniq_cnt[:15]
uniq_cnt[15:]
np.where(((uniq | leanY[0]) == leanY[0]).all(axis=1) & (uniq & leanY[0]).any(axis=1))
pY = np.zeros((Y.shape[0], uniq.shape[0]))
pY.shape
pY.dtype
for i in range(pY.shape[0]):
	pY[i, np.where(((uniq | leanY[0]) == leanY[0]).all(axis=1) & (uniq & leanY[0]).any(axis=1))] = 1.0
pY = np.zeros((Y.shape[0], uniq.shape[0]))
pY.shape ; pY.dtype
for i in range(pY.shape[0]):
	pY[i, np.where(((uniq | leanY[i]) == leanY[i]).all(axis=1) & (uniq & leanY[i]).any(axis=1))] = 1.0
np.where(pY[0] > 0)
kx = sklearn.manifold.TSNE(n_components=11, method='exact')
ttt = kx.fit_transform(pY.transpose())
ttt.shape
uniq.shape
np.where(Y[0] == uniq)
len(np.where(Y[0] == uniq)[0])
np.isin(Y[0], uniq)
len((np.where(Y[0] == uniq).all(axis=1))[0])
len((np.where((Y[0] == uniq).all(axis=1)))[0])
(np.where((Y[0] == uniq).all(axis=1)))
np.where(Y[0] > 0)
np.where(uniq[915] > 0)
np.where(uniq[910] > 0)
tY = np.zeros((pY.shape[0], ttt.shape[1]))
tY.shape ; tY.dtype
for i in range(tY.shape[0]):
np.where((Y[0] == uniq).all(axis=1))
np.where((Y[0] == uniq).all(axis=1))[0]
np.where((Y[0] == uniq).all(axis=1))[0][0]
tY = np.zeros((pY.shape[0], ttt.shape[1]))
Y.shape ; pY.shape ; tY.shape
for i in range(tY.shape[0]):
	tY[i] = uniq[np.where((Y[0] == uniq).all(axis=1))[0][0]]
tY = np.zeros((pY.shape[0], ttt.shape[1]))
Y.shape ; pY.shape ; tY.shape
for i in range(tY.shape[0]):
	tY[i] = ttt[np.where((Y[0] == uniq).all(axis=1))[0][0]]
tY[0]
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
tY_train = tY[:len(zp.pool)]
tY_test = tY[len(zp.pool):]
tY.shape ; tY_train.shape ; tY_test.shape
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), regressor)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
tY[1]
tY = np.zeros((pY.shape[0], ttt.shape[1]))
for i in range(tY.shape[0]):
	tY[i] = ttt[np.where((Y[i] == uniq).all(axis=1))[0][0]]
tY_train = tY[:len(zp.pool)]
tY_test = tY[len(zp.pool):]
Y.shape ; pY.shape ; tY.shape ; tY_train.shape ; tY_test.shape
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), regressor)
pipe.fit(X_train, tY_train)
pipe.score(X_train, tY_train)
pipe.score(X_test, tY_test)
tY[0]
tY[1]
predY_ztrain = pipe.predict(X_train)
predY_ztest = pipe.predict(X_test)
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.IncrementalPCA(n_components=800), regressor)
pipe.fit(vX_train, tY_train)
pipe.score(vX_train, tY_train)
pipe.score(vX_test, tY_test)
predY_vtrain = pipe.predict(vX_train)
predY_vtest = pipe.predict(vX_test)
co_ztrain = corretto(tY_train, predY_ztrain)
co_ztrain ; np.array(co_ztrain) / tY_test.shape[0]
co_ztrain ; np.array(co_ztrain) / tY_train.shape[0]
co_ztrain = corretto(tY_train, predY_ztrain) ; co_ztrain ; np.array(co_ztrain) / tY_train.shape
co_ztrain = corretto(tY_train, predY_ztrain) ; co_ztrain ; np.array(co_ztrain) / tY_train.shape[0]
co_ztest = corretto(tY_test, predY_ztest) ; co_ztest ; np.array(co_ztest) / tY_test.shape[0]
co_vtrain = corretto(tY_train, predY_vtrain) ; co_vtrain ; np.array(co_vtrain) / tY_train.shape[0]
co_vtest = corretto(tY_test, predY_vtest) ; co_vtest ; np.array(co_vtest) / tY_test.shape[0]
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.IncrementalPCA(n_components=800), regressor)
hX = np.concatenate([Xmv, vX.toarray()], axis=1)
hX.shape
hX.shape ; Xmv.shape ; vX.shape
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.compose.ColumnTransformer(["zICA", sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), slice(0, Xmv.shape[1] - 2)), ("mean_var", 'passthrough', slice(Xmv.shape[1] - 2, Xmv.shape[1])), ("tfidf2gram", sklearn.decomposition.IncrementalPCA(n_components=800), slice(Xmv.shape[1], Xmv.shape[1] + vX.shape[1])), regressor)
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.compose.ColumnTransformer(["zICA", sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), slice(0, Xmv.shape[1] - 2)), ("mean_var", 'passthrough', slice(Xmv.shape[1] - 2, Xmv.shape[1])), ("tfidf2gram", sklearn.decomposition.IncrementalPCA(n_components=800), slice(Xmv.shape[1], Xmv.shape[1] + vX.shape[1]))]), regressor)
regressor = sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt')
pipe = sklearn.pipeline.make_pipeline(sklearn.compose.ColumnTransformer([("zICA", sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), slice(0, Xmv.shape[1] - 2)), ("mean_var", 'passthrough', slice(Xmv.shape[1] - 2, Xmv.shape[1])), ("tfidf2gram", sklearn.decomposition.IncrementalPCA(n_components=800), slice(Xmv.shape[1], Xmv.shape[1] + vX.shape[1]))]), regressor)
#pipe.fit(vX_train, tY_train)
hX.shape ; Xmv.shape ; vX.shape
hX_train = hX[:len(zp.pool)]
hX_test = hX[len(zp.pool):]
hX.shape ; hX_train.shape ; hX_test.shape
pipe.fit(hX_train, tY_train)
pipe.score(hX_train, tY_train)
pipe.score(hX_test, tY_test)
predY_htrain = pipe.predict(hX_train)
predY_htest = pipe.predict(hX_test)
co_htrain = corretto(tY_train, predY_htrain) ; co_htrain ; np.array(co_htrain) / tY_train.shape[0]
co_htest = corretto(tY_test, predY_htest) ; co_htest ; np.array(co_htest) / tY_test.shape[0]
regressor = TransformedTargetRegressor(sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt'), transformer=informer, check_inverse=False)
pipe = sklearn.pipeline.make_pipeline(sklearn.compose.ColumnTransformer([("zICA", sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), slice(0, Xmv.shape[1] - 2)), ("mean_var", 'passthrough', slice(Xmv.shape[1] - 2, Xmv.shape[1])), ("tfidf2gram", sklearn.decomposition.IncrementalPCA(n_components=800), slice(Xmv.shape[1], Xmv.shape[1] + vX.shape[1]))]), regressor)
pipe.fit(hX_train, tY_train)
regressor = TransformedTargetRegressor(sklearn.ensemble.ExtraTreesRegressor(n_estimators=80, max_depth=32, min_samples_split=4, max_features='sqrt'), transformer=informer, check_inverse=False)
pipe = sklearn.pipeline.make_pipeline(sklearn.compose.ColumnTransformer([("zICA", sklearn.decomposition.FastICA(n_components=27, whiten='unit-variance', max_iter=500), slice(0, Xmv.shape[1] - 2)), ("mean_var", 'passthrough', slice(Xmv.shape[1] - 2, Xmv.shape[1])), ("tfidf2gram", sklearn.decomposition.IncrementalPCA(n_components=800), slice(Xmv.shape[1], Xmv.shape[1] + vX.shape[1]))]), regressor)
pipe.fit(hX_train, Y_train)
pipe.score(hX_train, Y_train)
pipe.score(hX_test, Y_test)
predY_rtrain = pipe.predict(hX_train)
predY_rtest = pipe.predict(hX_test)
(predY_rtrain == Y_train).shape
hX.shape ; hX_train.shape ; hX_test.shape
(predY_rtrain == Y_train).all(axis=1).shape
(predY_rtrain == Y_train).all(axis=1).sum()
(predY_rtrain == Y_train).all(axis=1).sum() / Y_train.shape[0]
(predY_rtest == Y_test).all(axis=1).sum() / Y_test.shape[0]
#co_ztest and transformed target with rbm, i.e. predY_rtest, yield similar amount, a litte over 50% of completly correct prediction for test data, which is not completely stratified, i.e. just for single labels but noch their combination
sklearn.metrics.f1_score(Y_test, predY_rtest, average='micro')
sklearn.metrics.f1_score(Y_test, predY_rtest, average='macro')
sklearn.metrics.f1_score(Y_test, predY_rtest, average='weighted')
sklearn.metrics.f1_score(Y_test, predY_rtest, average='micro')
pipe.score(hX_test, Y_test)
(predY_rtest == Y_test).all(axis=1).sum() / Y_test.shape[0]
co_ztest = corretto(tY_test, predY_ztest) ; co_ztest ; np.array(co_ztest) / tY_test.shape[0]
