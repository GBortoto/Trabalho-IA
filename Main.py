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
		# map_dim = 20
		# som = MiniSom(map_dim, map_dim, dados.shape[1], sigma=1.0, random_seed=1, learning_rate=0.5)
		# som.random_weights_init(dados)
		# som.train_batch(dados, 10000)
		# print(som.activation_response(dados))
		# print(som.quantization_error(dados))
		# print(som.win_map(dados))
		# print(som.distance_map(dados))
		# som.plot2(dados)

		mapsize = [50,50]
		som = SOMFactory.build(dados, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch')
		som.train(n_job=5, verbose='info')  # verbose='debug' will print more, and verbose=None wont print anything
		v = View2DPacked(50, 50, 'test',text_size=8)
		# could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
		v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
		v.save('2d_packed_test')


	else:
		preprocessor = ProcessTexts(texts=['bbc_local'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		data = matrix.get_matrix(type='tf-idf')
