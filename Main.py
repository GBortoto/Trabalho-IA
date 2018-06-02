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


		# kmeans = kmeans_default.KMeansDefault(matrix.get_matrix(type='tf-n'))
		kmeans = KMeans(dados)
		# kmeans.plots()
		# kmeans.roda_kmeans(5)
		# kmeans.plots(type='movement')
		# kmeans.plots(type='movement2')
		# kmeans.plots(type='movement3')
		# kmeans.plots(type='points')

		print('----- Iniciando Processamento SOM -----')
		# Implementação usando MiniSOM + kaggle
		map_dim = 20
		som = MiniSom(map_dim, map_dim, 50, sigma=1.0, random_seed=1)
		# print('Shape' + str(dados.shape))
		som = MiniSom(map_dim, map_dim, dados.shape[1], sigma=1.0, random_seed=1, learning_rate=0.5)
		som.random_weights_init(dados)
		som.train_batch(dados, 10000)
		print('-- Activation Response --')
		print(som.activation_response(dados))
		print('-- Quantization Error --')
		print(som.quantization_error(dados))
		print('-- Win Map --')
		print(som.win_map(dados))
		som.plot2(dados)
		som.plot3()
		# som.plot4()
		som.plot5()

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
	[X] - Ngrams
	[X] - Rodar K-means para cada matrix
	[X] - Rodar SOM para cada matrix
	[] - Pos-processamento
"""
