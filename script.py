from math import sqrt

from numpy import (array, unravel_index, nditer, linalg, random, subtract,
                   power, exp, pi, zeros, arange, outer, meshgrid, dot)
from collections import defaultdict
from warnings import warn

# for unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from numpy.testing import assert_array_equal
import unittest

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    return sqrt(dot(x, x.T))


class MiniSom(object):
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=None, neighborhood_function='gaussian',
                 random_seed=None):
        """Initializes a Self Organizing Maps.
        Parameters
        ----------
        decision_tree : decision tree
        The decision tree to be exported.
        x : int
            x dimension of the SOM
        y : int
            y dimension of the SOM
        input_len : int
            Number of the elements of the vectors in input.
        sigma : float, optional (default=1.0)
            Spread of the neighborhood function, needs to be adequate
            to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T)
            where T is #num_iteration/2)
            learning_rate, initial learning rate
            (at the iteration t we have
            learning_rate(t) = learning_rate / (1 + t/T)
            where T is #num_iteration/2)
        decay_function : function (default=None)
            Function that reduces learning_rate and sigma at each iteration
            default function:
            lambda x, current_iteration, max_iter :
                        x/(1+current_iteration/max_iter)
        neighborhood_function : function, optional (default='gaussian')
            Function that weights the neighborhood of a position in the map
            possible values: 'gaussian', 'mexican_hat'
        random_seed : int, optiona (default=None)
            Random seed to use.
        """
        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        if random_seed:
            self._random_generator = random.RandomState(random_seed)
        else:
            self._random_generator = random.RandomState(random_seed)
        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, max_iter: x/(1+t/max_iter)
        self._learning_rate = learning_rate
        self._sigma = sigma
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        for i in range(x):
            for j in range(y):
                # normalization
                norm = fast_norm(self._weights[i, j])
                self._weights[i, j] = self._weights[i, j] / norm
        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function
        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat}
        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))
        self.neighborhood = neig_functions[neighborhood_function]

    def get_weights(self):
        """Returns the weights of the neural network"""
        return self._weights

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
           the element i,j is the response of the neuron i,j to x"""
        s = subtract(x, self._weights)  # x - w
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            self._activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

    def activate(self, x):
        """Returns the activation map to x"""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Returns a Gaussian centered in c"""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c"""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x"""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t):
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        """
        eta = self._decay_function(self._learning_rate, t, self.T)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, self.T)
        # improves the performances
        g = self.neighborhood(win, sig)*eta
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            x_w = (x - self._weights[it.multi_index])
            self._weights[it.multi_index] += g[it.multi_index] * x_w
            # normalization
            norm = fast_norm(self._weights[it.multi_index])
            self._weights[it.multi_index] = self._weights[it.multi_index]/norm
            it.iternext()

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data."""
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self._weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """Initializes the weights of the SOM
        picking random samples from data"""
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            norm = fast_norm(self._weights[it.multi_index])
            self._weights[it.multi_index] = self._weights[it.multi_index]/norm
            it.iternext()

    def train_random(self, data, num_iteration):
        """Trains the SOM picking samples at random from data"""
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            # pick a random sample
            rand_i = self._random_generator.randint(len(data))
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

    def train_batch(self, data, num_iteration):
        """Trains using all the vectors in data sequentially"""
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        """Initializes the parameter T needed to adjust the learning rate"""
        # keeps the learning rate nearly constant
        # for the last half of the iterations
        self.T = num_iteration/2

    def distance_map(self):
        """Returns the distance map of the weights.
        Each cell is the normalised sum of the distances between
        a neuron and its neighbours."""
        um = zeros((self._weights.shape[0], self._weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1, it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1, it.multi_index[1]+2):
                    if (ii >= 0 and ii < self._weights.shape[0] and
                            jj >= 0 and jj < self._weights.shape[1]):
                        w_1 = self._weights[ii, jj, :]
                        w_2 = self._weights[it.multi_index]
                        um[it.multi_index] += fast_norm(w_1-w_2)
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self, data):
        """
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = zeros((self._weights.shape[0], self._weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        error = 0
        for x in data:
            error += fast_norm(x-self._weights[self.winner(x)])
        return error/len(data)

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap


class TestMinisom(unittest.TestCase):
    def setup_method(self, method):
        self.som = MiniSom(5, 5, 1)
        for i in range(5):
            for j in range(5):
                # checking weights normalization
                assert_almost_equal(1.0, linalg.norm(self.som._weights[i, j]))
        self.som._weights = zeros((5, 5))  # fake weights
        self.som._weights[2, 3] = 5.0
        self.som._weights[1, 1] = 2.0

    def test_decay_function(self):
        assert self.som._decay_function(1., 2., 3.) == 1./(1.+2./3.)

    def test_fast_norm(self):
        assert fast_norm(array([1, 3])) == sqrt(1+9)

    def test_unavailable_neigh_function(self):
        with self.assertRaises(ValueError):
            MiniSom(5, 5, 1, neighborhood_function='boooom')

    def test_gaussian(self):
        bell = self.som._gaussian((2, 2), 1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_win_map(self):
        winners = self.som.win_map([5.0, 2.0])
        assert winners[(2, 3)][0] == 5.0
        assert winners[(1, 1)][0] == 2.0

    def test_activation_reponse(self):
        response = self.som.activation_response([5.0, 2.0])
        assert response[2, 3] == 1
        assert response[1, 1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)

    def test_quantization_error(self):
        self.som.quantization_error([5, 2]) == 0.0
        self.som.quantization_error([4, 1]) == 0.5

    def test_quantization(self):
        q = self.som.quantization(array([4, 2]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        # same initialization
        assert_array_almost_equal(som1._weights, som2._weights)
        data = random.rand(100, 2)
        som1 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som1.train_random(data, 10)
        som2 = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        som2.train_random(data, 10)
        # same state after training
        assert_array_almost_equal(som1._weights, som2._weights)

    def test_train_batch(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_batch(data, 10)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5, 5, 2, sigma=1.0, learning_rate=0.5, random_seed=1)
        data = array([[4, 2], [3, 1]])
        q1 = som.quantization_error(data)
        som.train_random(data, 10)
        assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = MiniSom(2, 2, 2, random_seed=1)
        som.random_weights_init(array([[1.0, .0]]))
        for w in som._weights:
            assert_array_equal(w[0], array([1.0, .0]))
"""."""
# -*- coding: utf-8 -*-

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# import numpy as np

class TransformMatrix():
	"""."""
	def __init__(self, matrix):
		# Guarda matrix de lista de palavras por texto
		self.matrix = matrix

		# Cria matrix
		self._matrix_creation()

	def _matrix_creation(self):
		# Iremos criar uma "vetorizacao" baseado em frequencia (count)
		# vectorizer = CountVectorizer(max_df=0.9, min_df=0.05)
		vectorizer = CountVectorizer()

		#Retorna array TF de cada palavra
		self.bag_of_words = (vectorizer.fit_transform(self.matrix)).toarray()

		# Retorna array com as palavras (labels)
		self.feature_names = vectorizer.get_feature_names()

		# del self.matrix

	def get_matrix(self, type='tf-n'):
		# Matrix binaria será sempre a matrix TF para os casos em que a frequencia é diferente de 0
		# Método sign identifica se numero != 0
		# print(type)
		# print(type == 'tf-n')
		if type is 'binary':
			print('----- Processando Matriz Binaria -----')
			return (sp.sign(self.bag_of_words))
		# Matrix TF somente com frequencia da palavra, independente da frequencia relativa do corpus
		if type == 'tf':
			print('----- Processando Matriz TF -----')
			return self.bag_of_words
		# Matrix TF normalizada com frequencia indo de [0, 1)
		if type == 'tf-n':
			print('----- Processando Matriz TF-Normalizada -----')
			listas = [np.sum(lista, axis=0) for lista in self.bag_of_words]
			result = sum(listas)
			return self.bag_of_words / result
		# Matrix TF_IDF que utiliza inverse document
		if type == 'tf-idf':
			print('----- Processando Matriz TF-IDF -----')
			tfidf_vectorize = TfidfTransformer(smooth_idf=False)
			return tfidf_vectorize.fit_transform(self.bag_of_words).toarray()
"""Class to process all texts."""

# import os
# import string
#
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer


class ProcessTexts():
    """."""

    def __init__(self, texts):
        """."""
        self._read_text(texts)
        self._process_text()

    def _read_text(self, texts):
        self._texts = []  # list of text samples

        if 'bbc_local' in texts:
            for directory in sorted(os.listdir('./database/bbc_news')):
                for file in sorted(os.listdir('./database/bbc_news/' + directory)):
                    path = './database/bbc_news/' + directory + "/" + file
                    f = open(path, encoding='latin-1')
                    t = f.read()
                    self._texts.append(t)
                    f.close()
        if 'bbc_kaggle' in texts:
            for directory_type in sorted(os.listdir('../input/bbc news summary/BBC News Summary/')):
                for directory in sorted(os.listdir('../input/bbc news summary/BBC News Summary/' + directory_type)):
                    for file in sorted(os.listdir('../input/bbc news summary/BBC News Summary/' + directory_type + "/" + directory)):
                        f = open('../input/bbc news summary/BBC News Summary/' + directory_type + "/" + directory + "/" + file, encoding='latin-1')
                        t = f.read()
                        self._texts.append(t)
                        f.close()

    def _process_text(self, type='Porter'):
        print("----- Tokenizando Sentencas e Palavras -----")
        table = str.maketrans('', '', string.punctuation)
        stop_words = set(stopwords.words('english'))
        self.tokens = []

        # Para cada texto
        for index, text in enumerate(self._texts):
            # Tokenize por sentenca
            sentences = sent_tokenize(text)
            tokens_of_sentence = []
            # Para cada sentenca
            for sentence in sentences:
                # Tokenize por palavras, elimine stop words, pontuação e de lower
                stripped = [word.translate(table).lower() for word in word_tokenize(sentence) if not word in stop_words]
                stemmerized = self._normalize_text(tokens=stripped, type=type)
                tokens_of_sentence = tokens_of_sentence + stemmerized
            self.tokens.append(tokens_of_sentence)
        del self._texts
        self._join_words()

    def _normalize_text(self, tokens, type):
        if type is 'Porter':
            porter = PorterStemmer()
            return [porter.stem(t) for t in tokens]
        if type is 'Lancaster':
            lancaster = LancasterStemmer()
            return [lancaster.stem(t) for t in tokens]
        if type is 'Snowball':
            snowball = SnowballStemmer('english')
            return [snowball.stem(t) for t in tokens]
        if type is 'Lemma':
            lemma = WordNetLemmatizer()
            return [lemma.lemmatize(t) for t in tokens]

    def _join_words(self):
        new_tokens = []
        for token in self.tokens:
            # new_tokens.append((' '.join(token)).replace('  ', ' '))
            new_tokens.append(' '.join(token))
        self.tokens = new_tokens
"""https://flothesof.github.io/k-means-numpy.html ."""

# import numpy as np
# import matplotlib.pyplot as plt


class KMeans():
    """."""

    def __init__(self, points, type_of_kmeans='default'):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.

        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
        self.points = points

    def plots(self, type='points', save=True):
        """."""
        if type == 'points':
            plt.scatter(self.points[:, 0], self.points[:, 1])
            ax = plt.gca()
            ax.add_artist(plt.Circle(np.array([1, 0]), 0.75/2, fill=False, lw=3))
            ax.add_artist(plt.Circle(np.array([-0.5, 0.5]), 0.25/2, fill=False, lw=3))
            ax.add_artist(plt.Circle(np.array([-0.5, -0.5]), 0.5/2, fill=False, lw=3))

        if type == 'centroids':
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', s=100)

        if type == 'movement':
            plt.subplot(121)
            plt.scatter(self.points[:, 0], self.points[:, 1])
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', s=100)

            plt.subplot(122)
            plt.scatter(self.points[:, 0], self.points[:, 1])
            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', s=100)

        if save is False:
            plt.show()
        else:
            print('Salvando resultados...')
            plt.savefig('result_' + type + '.png')

    def inicia_centroides(self, k_centroids):
        """."""
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        """."""
        distancias = np.sqrt(((self.points - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distancias, axis=0)

    def roda_kmeans(self, k_centroids):
        """."""
        self.inicia_centroides(k_centroids)
        self.movimenta_centroides(self.busca_centroides_mais_proximo())

    def movimenta_centroides(self, closest):
        """."""
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])
"""."""
# Ativar para rodar SOM local
# import sompy as sompy


class SomDefault():
    """."""

    def __init__(self, data):
        """."""
        som = sompy.SOMFactory.build(data, [50, 50], mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
        som.train(n_job=3, verbose='info')
        v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
        # could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
        v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
        # v.save('2d_packed_test')
# -*- coding: utf-8 -*-

# import tensforflow as tf

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
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
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
            learning_rate_op = tf.subtract(1.0, tf.divide(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)

            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.divide(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
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
            print('-- Iteracao nº' + str(iter_no) + ' --')
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
	# env = 'kaggle'
	env = 'local'

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

		# Implementação usando MiniSOM + kaggle
		map_dim = 16
		som = MiniSom(map_dim, map_dim, 50, sigma=1.0, random_seed=1)
		#som.random_weights_init(W)
		som.train_batch(data, len(data)*500)

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
