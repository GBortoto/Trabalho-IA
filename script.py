# -*- coding: utf-8 -*-

"""Implementação minimalista do self-organizing maps.

Fonte: https://github.com/JustGlowing/minisom
"""

from math import sqrt
from numpy import (array, unravel_index, nditer, linalg, random, subtract, power, exp, pi, zeros, arange, outer, meshgrid, dot)
from collections import defaultdict
from warnings import warn

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def fast_norm(x):
    """Return norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    # Retorna a normalização da matrix. Ref: https://en.wiktionary.org/wiki/two-norm
    return sqrt(dot(x, x.T))


class MiniSom(object):
    """."""

    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=None, neighborhood_function='gaussian',
                 random_seed=None):
        """Initialize a Self Organizing Maps.

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
        """Return the weights of the neural network."""
        return self._weights

    def _activate(self, x):
        """Update matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x."""
        s = subtract(x, self._weights)  # x - w
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            # || x - w ||
            self._activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

    def activate(self, x):
        """Return the activation map to x."""
        self._activate(x)
        return self._activation_map

    def _gaussian(self, c, sigma):
        """Return a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    def winner(self, x):
        """Compute the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(),
                             self._activation_map.shape)

    def update(self, x, win, t):
        """Update the weights of the neurons.

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
        """Assign a code book (weights vector of the winning neuron) to each sample in data."""
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self._weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """Initialize the weights of the SOM picking random samples from data."""
        it = nditer(self._activation_map, flags=['multi_index'])
        while not it.finished:
            rand_i = self._random_generator.randint(len(data))
            self._weights[it.multi_index] = data[rand_i]
            norm = fast_norm(self._weights[it.multi_index])
            self._weights[it.multi_index] = self._weights[it.multi_index]/norm
            it.iternext()
        self.starting_weights = self.get_weights().copy()

    def train_random(self, data, num_iteration):
        """Trains the SOM picking samples at random from data"""
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            # print("[Treinando SOM: " + str(iteration/num_iteration) + "% COMPLETO]")
            # pick a random sample
            rand_i = self._random_generator.randint(len(data))
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

    def train_batch(self, data, num_iteration):
        """Trains using all the vectors in data sequentially"""
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            # print("[Treinando SOM: " + str(iteration/num_iteration) + "% COMPLETO]")
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

    def plot(self, dados):
        author_to_color = {0: 'chocolate', 1: 'steelblue', 2: 'dimgray'}
        color = [author_to_color[yy] for yy in y]
        plt.figure(figsize=(14, 14))
        for i, (t, c, vec) in enumerate(zip(titles, color, dados)):
            winnin_position = winner(vec)
            plt.text(winnin_position[0],
                     winnin_position[1]+np.random.rand()*.9,
                     t,
                     color=c)

        plt.xticks(range(map_dim))
        plt.yticks(range(map_dim))
        plt.grid()
        plt.xlim([0, map_dim])
        plt.ylim([0, map_dim])
        plt.plot()

    def plot2(self, dados):
        # Plotting the response for each pattern in the iris dataset
        plt.bone()
        plt.pcolor(self.distance_map().T)  # plotting the distance map as background
        plt.colorbar()
        t = np.zeros(len(dados), dtype=int)
        markers = ['o', 's', 'D']
        colors = ['r', 'g', 'b']
        for cnt, xx in enumerate(dados):
            w = self.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None', markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
        plt.axis([0, 7, 0, 7])
        # plt.show()
        plt.savefig('temp.png')

    def plot4(self, dados):
        plt.figure(figsize=(7, 7))
        wmap = {}
        im = 0
        for x, t in zip(dados, num):  # scatterplot
            w = self.winner(x)
            wmap[w] = im
            plt. text(w[0]+.5,  w[1]+.5,  str(t),
                      color=plt.cm.Dark2(t / 4.), fontdict={'weight': 'bold',  'size': 11})
            im = im + 1
        plt.axis([0, self.get_weights().shape[0], 0,  self.get_weights().shape[1]])
        plt.show()

        plt.figure(figsize=(10, 10), facecolor='white')
        cnt = 0
        for j in reversed(range(20)):  # images mosaic
            for i in range(20):
                plt.subplot(20, 20, cnt+1, frameon=False,  xticks=[],  yticks=[])
                if (i, j) in wmap:
                    plt.imshow(digits.images[wmap[(i, j)]],
                               cmap='Greys', interpolation='nearest')
                else:
                    plt.imshow(np.zeros((8, 8)),  cmap='Greys')
                cnt = cnt + 1

        plt.tight_layout()
        plt.show()

    def plot6(self):
        # Find frauds
        mappings = self.win_map(X)
        mappings
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

	def _matrix_creation(self, n_gram=(1, 1)):
		# Iremos criar uma "vetorizacao" baseado em frequencia (count)
		stop_words = set(stopwords.words('english'))
		vectorizer = CountVectorizer(max_df=0.9, min_df=0.05, stop_words=stop_words, ngram_range=n_gram)
		# vectorizer = CountVectorizer()

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

class KMeans():
    """."""

    def __init__(self, points, type_of_kmeans='default'):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.
        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
        self.points = points
        self.MediaDistAtual = 100000000000000000000.0
        self.erro = 0.1
        self.labels = []
        self.lista_centroid_mais_proximos = None

    def plots(self, type='points', save=True):
        """."""
        if type == 'points':
            plt.scatter(self.points[:, 0], self.points[:, 1])
            ax = plt.gca()
            pca = PCA(n_components=2).fit(self.points)
            dados2d = pca.transform(self.points)
            print(str(len(dados2d)))
            plt.scatter(dados2d[:,0], dados2d[:,1])
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

        if type == 'movement2':
            plt.subplot(121)
            plt.scatter(self.points[:, 1], self.points[:, 2])
            plt.scatter(self.centroids[:, 1], self.centroids[:, 2], c='r', s=100)

            plt.subplot(122)
            plt.scatter(self.points[:, 1], self.points[:, 2])
            plt.scatter(self.centroids[:, 1], self.centroids[:, 2], c='r', s=100)

        if type == 'movement3':
            plt.subplot(121)
            plt.scatter(self.points[:, 2], self.points[:, 3])
            plt.scatter(self.centroids[:, 2], self.centroids[:, 3], c='r', s=100)

            plt.subplot(122)
            plt.scatter(self.points[:, 2], self.points[:, 3])
            plt.scatter(self.centroids[:, 2], self.centroids[:, 3], c='r', s=100)

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
        #É adicionado uma nova dimensão aos centroids de forma que seja possivel 
        #calcular as diferenças entre as coordenadas para todos os centroids de uma vez
        #Ex: 
        #Antes de ser redimensionado  :  centroids                = [[1,2,3][4,5,6][7,8,9]]
        #Depois de ser redimensionado :  centroids_redimensionado = [[[1,2,3],[4,5,6],[7,8,9]]]
        #dessa forma podemos obter as diferenças entre as cordenadas em uma unica operação:
        #supondo p1 seja um ponto : [0,1,2]
        #temos: centroids_redimensionado - p1 = [
        #                     [1,1,1], -> diferença das cordenadas do ponto 1 para o centroid 1
        #                     [4,4,4], -> diferença das cordenadas do ponto 1 para o centroid 2
        #                     [7,7,7]  -> diferença das cordenadas do ponto 1 para o centroid 3 
        #                   ]
        #
        centroids_redimensionado = self.centroids[:, np.newaxis , :]
        #eleva-se a diferença ao quadrado 
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        #soma as diferenças e faz a raiz delas, obtendo as distancias euclidianas de todos os pontos para todos os centroids
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        #identifica o centroid mais próximo de cada ponto
        centroid_mais_proximo = np.argmin(distancias, axis=0)
    
        return centroid_mais_proximo

    def roda_kmeans(self, k_centroids):
        """."""
        self.inicia_centroides(k_centroids)
        MediaDistAnterior = 0.0
        nIteracoes = 0 
        while(abs(MediaDistAnterior- self.MediaDistAtual) > self.erro):
            #Só executa se a lista de centroids não tiver sido determinada na ultima iteração
            nIteracoes += 1
            print("quantidade de iterações igual à " +str(nIteracoes))
            if(self.lista_centroid_mais_proximos is None):
                self.labels = self.busca_centroides_mais_proximo()
                self.centroids = self.movimenta_centroides(self.labels)
            else:
                #movimenta os centroids  a partir da lista adquirida na ultima iteração
                self.centroids = self.movimenta_centroides(self.lista_centroid_mais_proximos)
            MediaDistAnterior = self.MediaDistAtual
            #atualiza lista de centroids mais proximos e calcula a média da distancia entre os pontos e
            #os centroids mais proximos
            self.MediaDistAtual = self.calculaMediaDistancias(self.lista_centroid_mais_proximos)
            
            
    def movimenta_centroides(self, closest):
        """."""
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])

    def calculaMediaDistancias(self , centroid_mais_proximo):
        
        centroids_redimensionado = self.centroids[:, np.newaxis , :]
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        centroid_mais_proximo = np.argmin(distancias, axis=0)
        
        listaDistancias = [0.0]*len(self.centroids)
        indexlista = 0
        #soma todas as distâncias entre os pontos e os centroids mais próximos
        for centroid in centroid_mais_proximo:
            listaDistancias[centroid] += distancias[centroid][indexlista]
            indexlista += 1
        #tira a média da distância entre os pontos e os centroids 
        for indice in range(0 , len(listaDistancias)):
            listaDistancias[indice] = listaDistancias[indice]/sum(centroid_mais_proximo == indice)
        return sum(listaDistancias)# -*- coding: utf-8 -*-

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
