#!/usr/bin/python3

__version__ = '0.0.0x'

import sys
import os
import os.path

import zlib

import numpy as np

from functools import total_ordering
from collections import OrderedDict


WSIZEBITS = 15
WSIZE = 2**WSIZEBITS


@total_ordering
class ZText:
	CATS = OrderedDict()
	@classmethod
	def CATADD(cls, catlabels):
		catids = set()
		for catlabel in catlabels:
			catids.add(cls.CATS.setdefault(catlabel, 1 + len(cls.CATS)))
		return catids
	@classmethod
	def CATINV(cls, catid):
		if isinstance(catid, int) and (1 <= catid <= len(cls.CATS)):
			return list(cls.CATS.keys())[catid - 1]
		else:
			return None
	
	def __init__(self, key, catlabels, s):
		self.key = key
		self.catids = self.CATADD(catlabels)
		self.s = s
		self.b = s.encode(errors='ignore')
		self.z = zlib.compress(self.b, wbits=WSIZEBITS-2)	# reduce large text bias by using smaller window size, hence large text wont compress much better
		self.ratio = len(self.z) / (0.000001 + len(self.b))	# avoid div zero and make div floating point, could also be used against large text bias
	
	def __lt__(self, other):
		return self.ratio < other.ratio
	
	def __eq__(self, other):
		return self.ratio == other.ratio


class ZBin:
	def __init__(self, catid, zt=None, cobj=None):
		self.cobj = zlib.compressobj(wbits=WSIZEBITS) if (cobj is None) else cobj
		if zt:
			self.ztkeys = {zt.key: 0}
			self.cat = catid
			self.blen = len(zt.b)
			self.z = self.cobj.compress(zt.b)
		else:
			self.ztkeys = {}
			self.cat = catid
			self.blen = 0
			self.z = b''
		self.zfllen = len(self.z) + len(self.cobj.copy().flush())
	
	def __call__(self, zt):
		zbnew = ZBin(catid=self.cat, zt=None, cobj=self.cobj.copy())
		zbnew.ztkeys.update(self.ztkeys)
		zbnew.ztkeys[zt.key] = len(self.ztkeys)
		zbnew.blen = self.blen + len(zt.b)
		zbnew.z = self.z + zbnew.cobj.compress(zt.b)
		zbnew.zfllen = len(zbnew.z) + len(zbnew.cobj.copy().flush())
		return zbnew


class ZPool:
	def __init__(self, zts, dedup=True):
		if dedup:
			_pool = []
			s_i_dict = {}
			for i, zt in enumerate(zts):
				dup_i = s_i_dict.get(zt.s, -1)
				if dup_i < 0:
					s_i_dict[zt.s] = len(_pool)
					_pool.append(ZText(zt.key, [ZText.CATINV(catid) for catid in zt.catids], zt.s))
				else:
					for catid in zt.catids:
						if catid not in _pool[dup_i].catids:
							_pool[dup_i].catids.add(catid)
			self.pool = _pool
		else:
			self.pool = [zt for zt in zts]
		lsorted = sorted(self.pool, key=lambda zt: len(zt.b))
		self.median = len(lsorted[len(lsorted) // 2].b)
		self.med2buf = 2**(1 + [m2 for m2 in range(2, 15) if self.median < (2**m2)][0])
		self.pool = list(sorted(self.pool))
		self.testpool = []
		self._bins = []
	
	@staticmethod
	def load_from_basepath(basepath):
		for cat in os.listdir(basepath):
			for fn in os.listdir(os.path.join(basepath, cat)):
				#print(fn)
				try:
					s = open(os.path.join(basepath, cat, fn)).read()
					if s:
						yield ZText(os.path.join(cat, fn), cat, s)
				except UnicodeDecodeError:
					print("Error opening %s" % os.path.join(basepath, cat, fn))
	
	@staticmethod
	def load_reuters():
		import load_reuters as lr
		reuters = lr.stream_reuters_documents()
		for r in reuters:
			if len(r['body']) > 5:
				#for topic in r['topics']:
				#	yield ZText(r['title'], topic, r['body'])
				if r['topics']:
					yield ZText(r['title'], r['topics'], r['body'])
	
	def train_test_split(self, train_size):
		catls = {}
		pool = self.testpool + self.pool
		for i, zt in enumerate(pool):
			catls.setdefault(str(list(sorted(zt.catids))), []).append(i)
		train = []
		test = []
		for catl, idxs in catls.items():
			ridxs = np.array(idxs)
			np.random.shuffle(ridxs)
			train += [pool[i] for i in ridxs[:int(0.5 + len(idxs) * train_size)]]
			test += [pool[i] for i in ridxs[int(0.5 + len(idxs) * train_size):]]
		self.pool = list(sorted(train))
		self.testpool = list(sorted(test))
		self._bins = []
	
	def can_zextend(self, zbin, ztext):
		return (zbin.blen + len(ztext.b)) < (WSIZE - self.med2buf)
	
	@property
	def bins(self):
		if self._bins:
			return self._bins
		bins = []
		for zt in self.pool:
			best_per_cat = []
			for catid in zt.catids:
				bestidx = -1
				bestbin = ZBin(catid=catid, zt=zt)
				for idx, zbin in enumerate(bins):
					if (zbin.cat == catid) and self.can_zextend(zbin, zt):
						zbnew = zbin(zt)
						if (1.0 * zbnew.zfllen / zbnew.blen) <= (1.0 * bestbin.zfllen / bestbin.blen):
							bestidx = idx
							bestbin = zbnew
				if bestidx == -1:	# no (available) bin in this cat, add one
					bins.append(bestbin)
				else:
					#best_per_cat.append( (1.0 * bestbin.zfllen / bestbin.blen, bestidx, bestbin) )
					bins[bestidx] = bestbin
					#if bestidx == 40:
					#	print(bestbin.cat)
					#	print(bins[bestidx].cat)
			#if len(best_per_cat) == 1:
			#	bins[best_per_cat[0][1]] = best_per_cat[0][2]
			#elif len(best_per_cat) == 2:
			#	bins[best_per_cat[0][1]] = best_per_cat[0][2]
			#	bins[best_per_cat[1][1]] = best_per_cat[1][2]
			#elif len(best_per_cat) > 2:	# take best bins for best and worst cat
			#	#print(list(sorted([rrr[1] for rrr in best_per_cat])))
			#	best_worst = list(sorted(best_per_cat))
			#	bins[best_worst[0][1]] = best_worst[0][2]
			#	bins[best_worst[-1][1]] = best_worst[-1][2]
			#yield bins
		self._bins = list(sorted(bins, key=lambda zbin: zbin.cat))
		return self._bins
	
	def cbX(self, zts=None, include_mean_var=False):
		if zts is None:
			zts = self.pool + self.testpool
		meanvar_cols = 2 if include_mean_var else 0
		X = np.zeros((len(zts), len(self.bins) + meanvar_cols), dtype=int)
		enu_zts = list(enumerate(zts))
		for i, zbin in enumerate(self.bins):
			zbin_zfllen = zbin.zfllen
			zbin_len_z = len(zbin.z)
			zbin_cobj = zbin.cobj
			for j, zt in enu_zts:
				cobj = zbin_cobj.copy()
				zfllen = zbin_len_z + len(cobj.compress(zt.b)) + len(cobj.flush())
				X[j, i] = zfllen - zbin_zfllen
			print("ZBin (%5i) of %5i" % (i+1, len(self.bins)))
		Xnorm = np.array(X, dtype=float)
		if meanvar_cols == 0:
			for i in range(X.shape[0]):
				Xnorm[i] -= Xnorm[i].mean()
				Xnorm[i] /= (Xnorm[i].var()) ** 0.5
		else:
			for i in range(X.shape[0]):
				Xnorm[i, -2] = Xnorm[i, :-meanvar_cols].mean()
				Xnorm[i, :-meanvar_cols] -= Xnorm[i, -2]
				Xnorm[i, -1] = Xnorm[i, :-meanvar_cols].var()
				Xnorm[i, :-meanvar_cols] /= Xnorm[i, -1] ** 0.5
		return Xnorm
	
	def cbY(self, zts=None):
		if zts is None:
			zts = self.pool + self.testpool
		Y = np.zeros((len(zts), len(ZText.CATS)), dtype=float)
		for i, zt in enumerate(zts):
			for cati in zt.catids:
				Y[i, cati - 1] = 1.0
		return Y






def ztbin(zt, zbins):
	relratios_idxs = [(zbin.relratio_zextend(zt), idx) for idx, zbin in enumerate(zbins)]
	best_idxs = [idx for relr, idx in sorted(relratios_idxs)[:6]]
	return best_idxs





def fitter(zp, X_all, Y_all):
	import sklearn
	import sklearn.decomposition
	from sklearn.compose import TransformedTargetRegressor
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.ensemble import ExtraTreesRegressor
	from sklearn.random_projection import GaussianRandomProjection
	from matplotlib import pyplot as plt
	regressor = TransformedTargetRegressor(ExtraTreesRegressor(max_depth=17), transformer=sklearn.decomposition.IncrementalPCA(n_components=3), check_inverse=False)
	pipe = sklearn.pipeline.make_pipeline(sklearn.decomposition.FastICA(n_components=7, whiten='unit-variance'), regressor)
	X_train = X_all[0:len(zp.pool)]	# , 0:-2]
	X_test = X_all[len(zp.pool):]	# , 0:-2]
	Y_train = Y[:len(zp.pool)]
	Y_test = Y[len(zp.pool):]
	pipe.fit(X_train, Y_train)
	print(pipe.score(X_train, Y_train))
	print(pipe.score(X_test, Y_test))
	return [pipe, X_train, Y_train, X_test, Y_test]

def RBMTransformer(n_components = 500, n_iter=30):
	import sklearn.preprocessing
	import sklearn.neural_network
	rbm = sklearn.neural_network.BernoulliRBM(n_components = n_components, n_iter=n_iter)
	def ft_tfunc(tX):
		return np.matmul(tX, rbm.components_.transpose())
	def ft_ifunc(tY):
		rbinv = np.linalg.pinv(rbm.components_)
		mulr = np.matmul(tY, rbinv.transpose())
		reX = np.zeros(mulr.shape, dtype=float)
		for i in range(reX.shape[0]):
			reX[i, np.where(mulr[i] > min(0.5, 0.4 * max(mulr[i])))] = 1.0
		return reX
	transformer = sklearn.preprocessing.FunctionTransformer(ft_tfunc, ft_ifunc, check_inverse=False)
	transformer.fit = rbm.fit
	return transformer


	








