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

# Ativar para rodar SOM local
# import sompy as sompy

from sklearn.cluster import KMeans as KMeansDefault
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer


if __name__ == "__main__":
	env = 'kaggle'
	# env = 'local'

	if env == 'kaggle':
		preprocessor = ProcessTexts(texts=['bbc_kaggle'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		dados = matrix.get_matrix(type='tf-idf')


		# kmeans = kmeans_default.KMeansDefault(matrix.get_matrix(type='tf-n'))
		# kmeans = KMeans(dados)
		# kmeans.plots()
		# kmeans.roda_kmeans(5)
		# kmeans.plots(type='movement')
		print('----- Iniciando Processamento SOM -----')
		# Implementação usando MiniSOM + kaggle
		map_dim = 16
		# som = MiniSom(map_dim, map_dim, 50, sigma=1.0, random_seed=1)
		som = MiniSom(map_dim, map_dim, 33752, sigma=1.0, random_seed=1)
		#som.random_weights_init(W)
		som.train_batch(dados, len(dados)*500)

		# som = SOM(20, 30, 3, 33752)
		# som.train(dados)

		#Get output grid
		# image_grid = som.get_centroids()

		#Map colours to their closest neurons
		# mapped = som.map_vects(colors)

	else:
		preprocessor = ProcessTexts(texts=['bbc_local'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		data = matrix.get_matrix(type='tf-idf')
		# kmeans = kmeans_default.KMeansDefault(matrix.get_matrix(type='tf-n'))
		kmeans = KMeans(data)
		# kmeans.plots()
		# kmeans.roda_kmeans(5)
		# kmeans.plots(type='movement')

		# Implementação local do SOM
		# som = SomDefault(data)


	"""
		[X] - Ler todos os textos
		[X] - Fazer data clean dos dados
		[X] - Roda Bag of Words para transformar lista de textos em vetor bidimensional de frequencia de palavra por texto
		[X] - Criar 3 outputs do Bag of Words
			[X] - Matrix binaria
			[X] - Matrix tf
			[X] - Matrix tf_idf
			[] - Ngrams
		[] - Rodar K-means para cada matrix
		[] - Rodar SOM para cada matrix
		[] - Pos-processamento
	"""
