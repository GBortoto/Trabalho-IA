# -*- coding: utf-8 -*-
"""
# Dada uma matriz pura, essa classe é responsável por retornar qualquer tipo de matrix customizável
# corpus[texto0[palavraA, palavraB, palavraC], texto1[palavraX, palavraY, palavraZ]]
# Matrix pura: corpus[n_textos][m_palavras_fixas]
# a = [["A","B","C"],["X","Y","Z"],["L","M","N"]]
# x = [["John likes to watch movies. Mary likes movies too.",
"John also likes to watch football games."],["X","Y","Z"],["L","M","N"]]
# y = ["¿Plata o plomo?  La cárcel em Estados Unidos es peor que la muerte. Bueno ... en mi opinión, una cárcel en Estados Unidos es peor que la muerte. ", "Ma tem ou não tem o celular do milhãouamm? Ma! Ao adquirir o carnê do Baú, você estará concorrendo a um prêmio de cem mil reaisam. Ma quem quer dinheiroam? Ma não existem mulher feiam, existem mulher que não conhece os produtos Jequitiamm. Qual é a musicamm? Vem pra lá, mah você vai pra cá. Agora vai, agora vem pra láamm. Patríciaaammmm... Luiz Ricardouaaammmmmm. Ma vejam só, vejam só. Ma! Ao adquirir o carnê do Baú, você estará concorrendo a um prêmio de cem mil reaisam. Ma não existem mulher feiam, existem mulher que não conhece os produtos Jequitiamm. Estamos em ritmo de festamm."]

Fonte:
http://scikit-learn.org/stable/modules/feature_extraction.html

"""

class TransformMatrix():
	def __init__(self, matrix):
		# Guarda matrix de lista de palavras por texto
		self.matrix = matrix

		# Cria matrix
		self._matrix_creation()

	def _matrix_creation(self):
		# Iremos criar uma "vetorizacao" baseado em frequencia (count)
		vectorizer = CountVectorizer()

		#Retorna array TF de cada palavra
		self.bag_of_words = (vectorizer.fit_transform(self.matrix)).toarray()

		# Retorna array com as palavras (labels)
		self.feature_names = vectorizer.get_feature_names()

	# Matrix binaria será sempre a matrix TF para os casos em que a frequencia é diferente de 0
	def matrix_binaria(self):
		# Método sign identifica se numero != 0
		return (sp.sign(self.bag_of_words))

	# Matrix TF somente com frequencia da palavra, independente da frequencia relativa do corpus
	def matrix_tf(self):
		return self.bag_of_words

	# Matrix TF normalizada com frequencia indo de [0, 1)
	def matrix_tf_normalizada(self):
		listas = [np.sum(lista, axis=0) for lista in self.bag_of_words]
		result = sum(listas)
		return self.bag_of_words / result

	# Matrix TF_IDF que utiliza inverse document
	def matrix_tfidf(self):
		tfidf_vectorize = TfidfTransformer(smooth_idf=False)
		return tfidf_vectorize.fit_transform(self.bag_of_words).toarray()
"""Module to process texts."""

import string                                                   # Lista de caracteres de pontuação
import os                                                       # Miscellaneous operating system interfaces
import re                                                       # Regular Expressions
import random                                                   # Python Random Library
import scipy as sp
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize                         # Tokenizer
from nltk.corpus import stopwords                               # Stop Words
from nltk.stem.porter import *                                  # Stemmer - Porter
from nltk.stem.snowball import SnowballStemmer                  # Stemmer - Snowball
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from typing import List                                         # Anotação de help quando uma função é escrita
from pympler import asizeof


class ProcessaTextos():
    """Class that process multiple datasets to provide a input for KMeans and SOM."""

    def __init__(self):
        """Read texts and read them."""
        texts = []  # list of text samples
        labels_index = {}  # dictionary mapping label name to numeric id
        labels = []  # list of label ids
        for name in sorted(os.listdir('../input/')):
            path = os.path.join('../input/', name)
            label_id = len(labels_index)
            labels_index[name] = label_id
            f = open(path, encoding='latin-1')
            t = f.read()
            i = t.find('\n\n')  # skip header
            if 0 < i:
                t = t[i:]
            texts.append(t)
            f.close()
            labels.append(label_id)

        print('Found %s texts.' % len(texts))
        tokens = []
        for text in texts:
            tokens.append(word_tokenize(text))
class Kmeans():

    def __init__(self, type_of_kmeans, points):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.

        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
        self.points = points

    def see_points(self):
        # plt.scatter(points[:,0], points[:,1])
        ax = plt.gca()

    def inicia_centroides(self, k_centroids):
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        distancias = np.sqrt(((self.points - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distancias, axis=0)

    def roda_kmeans(self):
        self.inicia_centroides(4)
        self.movimenta_centroides(self.busca_centroid_mais_proximo())

    def movimenta_centroides(self, closest):
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])
# -*- coding: utf-8 -*-


class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    # To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        m x n -> dimensao do SOM
        n_interations -> #epocas que será treinado a rede
        alpha -> taxa de aprendizagem. Default 0.3
        sigma -> taxa de vizinhança. Define o raio que o BMU afeta. Default max(m, n)
        """

        # Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        ##INITIALIZE GRAPH - TF Graphs é o confunto de operações que serão realizadas + operadores
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))

            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training

            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training

            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.sub(self._weightage_vects, tf.pack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)

            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.sub(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.mul(alpha, learning_rate_op)
            _sigma_op = tf.mul(sigma, learning_rate_op)

            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.sub(
                self._location_vects, tf.pack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.mul(_alpha_op, neighbourhood_func)

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.pack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.mul(
                learning_rate_multiplier,
                tf.sub(tf.pack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return
# -*- coding: utf-8 -*-

if __name__ == "__main__":
	# preprocessor = Preprocessor()
	# texts = preprocessor.readAllTextsFromDatabase()
	# #texts contêm todos os textos que serão utilizados de forma que cada index do array tem uma notícia. As notícias não estão tratadas , são o texto puro , retirado apenas os e-mails e em ordem aleatória.
	# texts = preprocessor.processTexts(texts)
	# textos = []
	# for txt in texts:
	# 	textos.append(' '.join(txt))
	# texts = [item for sublist in textos for item in sublist]
	#
	#
	# # Devemos remover o vetor de vetores e somente deixar um vetor com várias palavras por indices
	# transformador = TransformMatrix(texts)
	# mtx_binaria = transformador.matrix_binaria()
	# print(mtx_binaria)

	preprocessor = ProcessaTextos()


	"""
		[X] - Ler todos os textos
		[X] - Fazer data clean dos dados
		[] - Roda Bag of Words para transformar lista de textos em vetor bidimensional de frequencia de palavra por texto
		[] - Criar 3 outputs do Bag of Words
			[] - Matrix binaria
			[] - Matrix tf
			[] - Matrix tf_idf
		[] - Rodar K-means para cada matrix
		[] - Rodar SOM para cada matrix
		[] - Pos-processamento
	"""

	""" Teste KMeans
	n = 10000
    dimentions = 10
    NGroups = 5
    iterations = 200

    data = ListOfPoints(n, dimentions)
    data.points = [[random()*1000 for j in range(dimentions)] for i in range(n)]
    kmeans = KMeans(data, NGroups)

    print('Inicializando execução')
    print(str(n) + ' elementos de dados')
    print(str(dimentions) + ' dimenções')
    print(str(NGroups) + ' grupos')
    print(str(iterations) + ' iterações')
    print('')
    print('Iteração\tNúmero de elementos em cada grupo')

    for i in range(iterations):
        print(str(i+1) + '/' + str(iterations) + '\t\t', end='')
        kmeans.run()

    input()
	"""
