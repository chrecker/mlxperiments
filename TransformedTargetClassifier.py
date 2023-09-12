#!/usr/bin/env python3


import numpy as np

import string
import collections

import sklearn
from sklearn.utils._param_validation import HasMethods

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.manifold import TSNE

class RegressorTSNEtoNHotClassifier(ClassifierMixin, BaseEstimator):
	_parameter_constraints: dict = {
		"regressor": [HasMethods(["fit", "predict"]), ],
	}
	
	def __init__(self, regressor, *, n_components=3, perplexity=30.0):
		self.regressor = regressor
		self.n_components = n_components
		self.perplexity = perplexity
		self.tsne = TSNE(n_components=n_components, method=('exact' if n_components > 3 else 'barnes_hut'), perplexity=perplexity)
		self.emby = None
	
	def transformer_fit(self, y, X=None):
		if self.emby is None:
			# expand N Hot Coded such that each unique combination gets its own column that is in turn hot coded
			leany = np.array(y, dtype=bool)
			uniq, uniq_cnt = np.unique(leany, axis=0, return_counts=True)
			py = np.zeros((y.shape[0], uniq.shape[0]))
			for i in range(py.shape[0]):
				py[i, np.where(((uniq | leany[i]) == leany[i]).all(axis=1) & (uniq & leany[i]).any(axis=1))] = 1.0
			emby = self.tsne.fit_transform(py.transpose())
			emin = emby.min()
			emax = emby.max()
			self.emby = (emby - emin) / (emax - emin)
			self.classes_ = uniq
		else:
			newclasses = []
			newembs = []
			y = np.array(y, dtype=bool)
			for i in range(y.shape[0]):
				idxs = np.where((y[i] == self.classes_).all(axis=1))[0]
				if not idxs:	# no perfect match, add new unique class and embedding
					newclasses.append(y[i:i+1])
					newembs.append(self.transform(newclasses[-1]))
			self.emby = np.concatenate([self.emby,] + newembs, dtype=self.emby.dtype)
			self.classes_ = np.concatenate([self.classes_,] + newclasses, dtype=self.classes_.dtype)
	
	def transform(self, y):
		ty = np.zeros((y.shape[0], self.emby.shape[1]))
		# duct tape approach to associating distances with probalities and hence weights for embedding vectors
		# more sophsticed approach could model the distribution of the points as a stochastic, e.g. gaussian, process and the calculate, e.g. fit, the conditional probability
		dimscale = y.shape[1] ** 0.5
		for i in range(ty.shape[0]):
			idxs = np.where((y[i] == self.classes_).all(axis=1))[0]
			if idxs:	# perfect match
				ty[i] = self.emby[idxs[0]]
			else:
				#w = np.exp2( dimscale * ((self.classes_ - y[i]) ** 2).sum(axis=1))
				if y.dtype == bool:
					dist = np.logical_xor(self.classes_, y[i]).sum(axis=1)
				else:
					dist = np.abs(self.classes_ - y[i]).sum(axis=1)
				dmin = dist.min()
				w = np.exp2( - dimscale * ((dimscale + dmin) / (1 + dmin)) * dist)
				ty[i] = (w[:, None] * self.emby).sum(axis=0) / w.sum()
		return ty
	
	def predict_proba_t(self, ty):
		proba = np.zeros((ty.shape[0], self.classes_.shape[0]), dtype=float)
		# duct tape, see above, now for original vectors, i.e. classes_
		embdimscale = self.classes_.shape[1] ** 0.5
		for i in range(proba.shape[0]):
			#distemb = np.argsort(((self.emby - ty[i]) ** 2.0).sum(axis=1))
			#embw = np.exp2( embdimscale * ((self.emby - ty[i]) ** 2).sum(axis=1))
			#embw = np.exp2( embdimscale * np.abs(self.emby - ty[i]).sum(axis=1))
			embdist = np.abs(self.emby - ty[i]).sum(axis=1)
			dmin = embdist.min()
			embw = np.exp2( - embdimscale * ((embdimscale + dmin) / (1 + dmin)) * embdist)
			proba[i] = embw / embw.sum()
		return proba
	
	def inverse_transform(self, ty):
		yhot = np.zeros((ty.shape[0], self.classes_.shape[1]), dtype=bool)
		for i in range(ty.shape[0]):
			distemb_idxsort = np.argsort(((self.emby - ty[i]) ** 2.0).sum(axis=1))
			yhot[i] = self.classes_[distemb_idxsort[0]]
		return yhot
	
	def fit(self, X, y, **fit_params):
		self.transformer_fit(y, X=None)
		ty = self.transform(y)
		self.regressor.fit(X, ty)
		return self
	
	def predict(self, X, **predict_params):
		ty_pred = self.regressor.predict(X, **predict_params)
		return self.inverse_transform(ty_pred)
	
	def predict_proba(self, X, **predict_params):
		ty_pred = self.regressor.predict(X, **predict_params)
		return self.predict_proba_t(ty_pred)



if __name__ == "__main__":
	pass






