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


if __name__ == "__main__":
	env = 'kaggle'

	if env == 'kaggle':
		preprocessor = ProcessTexts()
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		dados = matrix.get_matrix(type='tf-n')
		# kmeans = kmeans_default.KMeansDefault(matrix.get_matrix(type='tf-n'))
		kmeans = KMeans(dados)
		# kmeans.plots()
		kmeans.roda_kmeans(5)
		kmeans.plots(type='movement')

		som = SOM(20, 30, 3, 33752)
		som.train(dados)

		#Get output grid
		# image_grid = som.get_centroids()

		#Map colours to their closest neurons
		# mapped = som.map_vects(colors)

	else:
		preprocessor = preprocessor.ProcessTexts()
		print('----- Transformando Tokens em Matriz -----')
		matrix = mtx.TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')

		# kmeans = kmeans_default.KMeansDefault(matrix.get_matrix(type='tf-n'))
		kmeans = kmeans.KMeans(matrix.get_matrix(type='tf-n'))
		# kmeans.plots()
		kmeans.roda_kmeans(5)
		kmeans.plots(type='movement')


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
