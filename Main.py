import numpy as np
# from . import dotmap, histogram, hitmap, mapview, umatrix
import os
import string
import tensorflow as tf
import matplotlib.pyplot as plt
from math import inf as positive_infinite
from scipy.spatial import distance
from sklearn.cluster import KMeans as KMeansDefault
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from sklearn.decomposition import PCA
import pylab as pl

if __name__ == "__main__":
	env = 'kaggle'
	# env = 'local'

	if env == 'kaggle':
		preprocessor = ProcessTexts(texts=['eua_kaggle'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		dados = matrix.get_matrix(type='tf-idf')

		# ---------------------
		# K-means
		# print('----- Iniciando Processamento K-means -----')
		# kmeans = KMeans(dados)
		# kmeans.roda_kmeans(3)

		# kmeans = KMeans(dados, type_of_kmeans='kmeans++')
		# kmeans.roda_kmeans(3)

		# kmeans = KMeans(dados, type_of_kmeans='kmeans++', distance_type='cosine_similarity')
		# kmeans.roda_kmeans(3)

		# ---------------------
		# SOM
		print('----- Iniciando Processamento SOM -----')

		map_dim = 16
		som = MiniSom(map_dim, map_dim, dados.shape[1], sigma=1.0, random_seed=1)
		som.random_weights_init(dados)
		som.train_batch(dados, 100)

		plt.figure(figsize=(14, 14))
		for i, vec in enumerate(W):
			winnin_position = som.winner(vec)
			plt.text(winnin_position[0], winnin_position[1]+np.random.rand()*.9)
		plt.xticks(range(map_dim))
		plt.yticks(range(map_dim))
		plt.grid()
		plt.xlim([0, map_dim])
		plt.ylim([0, map_dim])
		plt.plot()
		# ---------------------
		# Plots
		# v = View2DPacked(25, 25, 'SOM Plots',text_size=8)
		# v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
		# v.show(som, what='codebook', which_dim=[0,1,2,3,4,5], cmap=None, col_sz=6) #which_dim='all' default
		# v.show(som, what='codebook', which_dim='all', cmap='jet', col_sz=6) #which_dim='all' default
		# v.save('2d_packed_test2')
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
