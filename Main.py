# -*- coding: utf-8 -*-

# Ativar qd rodar localmente
# import ProcessTexts as preprocessor
# import Matrix as mtx
# import KMeans as kmeans
# import KMeansDefault as kmeans_default

import numpy as np
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
		map_dim = 20
		som = MiniSom(map_dim, map_dim, dados.shape[1], sigma=1.0, random_seed=1, learning_rate=0.5)
		som.random_weights_init(dados)
		som.train_batch(dados, 10000)
		# print(som.activation_response(dados))
		# print(som.quantization_error(dados))
		# print(som.win_map(dados))
		# print(som.distance_map(dados))
		som.plot2(dados)

	else:
		preprocessor = ProcessTexts(texts=['bbc_local'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		data = matrix.get_matrix(type='tf-idf')
