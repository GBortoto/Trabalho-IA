# -*- coding: utf-8 -*-

# Ativar qd rodar localmente
# import ProcessTexts as preprocessor
# import Matrix as mtx
# import KMeans as kmeans
# import KMeansDefault as kmeans_default

import numpy as np
# from . import dotmap, histogram, hitmap, mapview, umatrix
import os
import string
import tensorflow as tf
import matplotlib.pyplot as plt
from math import inf as positive_infinite
from sklearn.cluster import KMeans as KMeansDefault
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from sklearn.decomposition import PCA

if __name__ == "__main__":
	env = 'kaggle'
	# env = 'local'

	if env == 'kaggle':
		preprocessor = ProcessTexts(texts=['bbc_kaggle'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		dados = matrix.get_matrix(type='tf-idf')

		# ---------------------
		# K-means
		print('----- Iniciando Processamento K-means -----')
		kmeans = KMeans(dados)

		# ---------------------
		# SOM
		print('----- Iniciando Processamento SOM -----')

		mapsize = [25,25]
		som = SOMFactory.build(dados, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='random', neighborhood='gaussian', training='batch')
		som.train(n_job=3, verbose='info')

		# ---------------------
		# Plots
		v = View2DPacked(25, 25, 'SOM Plots',text_size=8)
		# v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
		v.show(som, what='codebook', which_dim=[0,1,2,3,4,5], cmap=None, col_sz=6) #which_dim='all' default
		v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6) #which_dim='all' default
		v.save('2d_packed_test2')
		# som.component_names = ['1','2']
		# v = View2DPacked(2, 2, 'test',text_size=8)
		# cl = som.cluster(n_clusters=10)
		# getattr(som, 'cluster_labels')
		# h = HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
		# h.show(som)
		# h.save('2d_packed_test3')
		# u = UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
		# UMAT  = u.build_u_matrix(som, distance=1, row_normalized=False)
		# UMAT = u.show(som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)
		# u.save('2d_packed_test4')
