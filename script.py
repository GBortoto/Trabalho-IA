class KMeansPlotter():

    def __init__(self ):
    #limitado a 24 markers
        self.markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        np.random.shuffle(self.markers)
    def plots(self, kmeans,  type='points', save=True):
        """."""
        if type == 'points':
            plt = self.createPlotPoints(kmeans)

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
            plt.savefig('result_' + type + str(time()) + '.png')
        plt.clf()


    def createPlotPoints(self, kmeans):
         ##algumas variaveis de plotagem
        areaPoints = 10
        areaCentroid = 150
        
        ##lista com todos os pontos do gráfico
        
        all_points = kmeans.points.tolist()
        all_points.extend(kmeans.centroids.tolist())

        #lista com os centroids de cada ponto (serve como label de cada cluster)
        #centroids são demarcados com -1 para serem identificados na plotagem
        centroid_labels = [-1] * len(kmeans.centroids)
        clusters = kmeans.lista_centroid_mais_proximos.tolist()
        clusters.extend(centroid_labels)

        pca = PCA(n_components=2).fit(all_points)
        dados2d = pca.transform(all_points)
        
        for point in range(0 ,dados2d.shape[0]):
            centroid = clusters[point]
            if(centroid >= 0):
                pl.scatter(dados2d[point,0],dados2d[point,1], s = areaPoints, c= self.color[centroid%8],marker=self.markers[centroid%24])
            #centroids
            elif(centroid < 0):
                pl.scatter(dados2d[point,0],dados2d[point,1] , s = areaCentroid , c ='k' , marker = 'o')
                
    
        return pl
    
'''
Índice Silhouette

    Sendo i um dado,
        Axioma 1) Isil(i) = ( b(i) - a(i) ) / max{a(i), b(i)}
    representa o índice silhouette para o dado i

    onde:
        a(i) é a distância média do dado i a todos os demais dados do seu grupo
        b(i) é a distância média do dado i a todos os dados do grupo mais próximo ao seu
            (aka a menor distância média entre o dado e os dados dos outros grupos)

    Axioma 2) Isil(Grupo) = média(Isil(dado 1), Isil(dado 2), ..., Isil(dado N)  )

    Axioma 3) Isil(Agrupamento) = média(Isil(Grupo 1), Isil(Grupo 2), ..., Isil(Grupo N)  )

    O índice varia entre [-1, 1]
        1   = Ponto bem agrupado
        0   = a(i) ~= b(i) --> Não está claro se i deve pertencer ao grupo A ou ao grupo B
        -1  = Ponto mal agrupado


Referência: "Técnica de Agrupamento (Clustering)"
            Sarajane M. Peres e Clodoaldo A. M. Lima
            Slides 127-132
'''

class Silhouette():

    def __init__(self):
        pass

    # ------------------ Funções de conversão ----------------------------------

    def getAllGroups(self, points, labels):
        return [getGroupFromLabel(points, labels, i) for i in range(len(set(labels)))]

    def getGroupFromLabel(self, points, labels, label):
        return [a[i] for i in range(len(points)) if labels[i] == label]

    def getGroupOfPoint(self, point, points, labels):
        group = getGroupFromLabel(points, labels, labels[points.index(point)])

    # ------------------ Funções de distância ----------------------------------

    def distanceBetweenPointAndPoint(self, pointA, pointB, typeOfDistance):
        """Retorna a distância entre os pontos A e B, dado um tipo de cálculo de distância"""

        if(len(pointA) != len(pointB)):
            raise ValueError("Silhouette - distanceBetweenPointAndPoint(): number of dimentions of PointA != number of dimentions of PointB")
        numberOfDimentions = len(pointA)

        distance = -1
        if(typeOfDistance == 'euclidean'):
            distance = sqrt(sum([(pointA[i] - pointB[i])**2 for i in range(numberOfDimentions)]))
        elif(typeOfDistance == 'cosine similarity'):
            distance = sum([pointA[i] * pointB[i] for i in range(numberOfDimentions)])/ (
                              sqrt(sum([pointA[i]**2 for i in range(numberOfDimentions)]))
                            * sqrt(sum([pointB[i]**2 for i in range(numberOfDimentions)]))
                        )
        # Caso o tipo de cálculo de distância seja inválido, o método jogará o erro
        if(distance == -1):
            raise ValueError('Silhouette - distanceBetweenPointAndPoint(): Invalid type of distance: "' + typeOfDistance + '"' )
        return distance

    def meanDistanceBetweenPointAndGroup(self, point, group, typeOfDistance):
        """Retorna a média das distâncias entre o ponto recebido e todos os pontos do grupo recebido"""
        groupSize = len(group)
        if(group.index(point) != -1):
            groupSize -= 1
        return sum(
            [distanceBetweenPointAndPoint(point, group[i], typeOfDistance)
             for i in range(len(group)) if point != group[i]]
            ) / groupSize

    def findNearestGroup(self, point, groupOfPoint, groups, typeOfDistance):
        """Encontra o grupo mais próximo do ponto recebido"""

        # Cria uma lista de todos os grupos exceto o grupo atual do ponto
        # Calcula as médias de distância entre o ponto e todos os grupos da lista criada acima
        # Retorna o grupo que possui a menor distância média em relação ao ponto

        otherGroups = groups.remove(groupOfPoint)
        means = [meanDistanceBetweenPointAndGroup(point, otherGroups[i], typeOfDistance)
                 for i in range(len(otherGroups))]
        return otherGroups[means.index(min(means))]


    # ------------------- Funções Públicas -------------------------------------

    def pointSilhouette(self, point, points, labels, typeOfDistance='euclidean'):
        """Calcula o índice Silhouette para um dado"""
        groupOfPoint = getGroupOfPoint(point, points, labels)
        groups = getAllGroups(points, labels)

        # Retorna o cálculo do índice Silhouette para o ponto (Axioma 1)
        Ai = meanDistanceBetweenPointAndGroup(point,
                                              groupOfPoint,
                                              typeOfDistance)

        Bi = meanDistanceBetweenPointAndGroup(point,
                                              findNearestGroup(point, groupOfPoint, groups),
                                              typeOfDistance)
        return (Bi - Ai) / max(Ai, Bi)

    def groupSilhouette(self, label, points, labels, typeOfDistance='euclidean'):
        """Calcula o índice Silhouette para um grupo de dados"""

        group = getGroupFromLabel(points, labels, label)
        groups = getAllGroups(points, labels)

        # Retorna a média dos silhouetes dos dados no grupo (Axioma 2)
        return sum(
            [pointSilhouette(group[i], points, labels, typeOfDistance)
             for i in range(len(group))]
            )/len(group)

    def allGroupsSilhouette(self, points, labels, typeOfDistance='euclidean'):
        """Calcula o índice Silhouette para um grupo de grupos de dados"""

        groups = getAllGroups(points, labels)

        # Retorna a média dos silhouetes dos grupos de dados (Axioma 3)
        return sum(
            [groupSilhouette(groups[i], groups, typeOfDistance)
             for i in range(len(groups))]
            )/len(groups)
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
# -*- coding: utf-8 -*-
# Fonte: https://github.com/JustGlowing/minisom

# https://github.com/JustGlowing/minisom/blob/master/minisom.py
# https://github.com/JustGlowing/minisom/blob/master/examples/PoemsAnalysis.ipynb
# https://github.com/JustGlowing/minisom/blob/master/examples/examples.ipynb
# https://glowingpython.blogspot.com/2013/09/self-organizing-maps.html

u"""Implementação minimalista do self-organizing maps."""

from math import sqrt
from numpy import (array, unravel_index, nditer, linalg, random, subtract, power, exp, pi, zeros, arange, outer, meshgrid, dot)
from collections import defaultdict
from warnings import warn

# Gera um norm de uma matrix -> Norm é o produto vetorial da uma matrix X por sua transporta, tirando a raiz do resultado
def fast_norm(x):
    """Return norm-2 of a 1-D numpy array.

    * faster than linalg.norm in case of 1-D arrays (numpy 1.9.2rc1).
    """
    # Retorna a normalização da matrix. Ref: https://en.wiktionary.org/wiki/two-norm
    return sqrt(dot(x, x.T))


class MiniSom(object):
    """."""

    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, decay_function=None, neighborhood_function='gaussian', random_seed=None):
        """Inicializa o SOM.

        x: dimensao do lattice do SOM
        y: dimensao do lattice do SOM
        input_len : numero de dimensões do vetor de input
        sigma : taxa de espalhamento da função de vizinhança. Ajusta conforme iteração baseado na formula: (at the iteration t we have sigma(t) = sigma / (1 + t/T) onde T é #num_iteration/2)
        learning_rate: taxa de aprendizado. Ajusta conforme iteração baseado na formula (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) onde T é #num_iteration/2)
        decay_function : função de decaimento para sigma e alfa. Por padrão: lambda x, current_iteration, max_iter : x/(1+current_iteration/max_iter)
        neighborhood_function : função de vizinhança para calcular efeito do BMU. Padrão é gaussiana
        random_seed : see aleatória
        """

        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        if random_seed:
            self._random_generator = random.RandomState(random_seed)
        else:
            self._random_generator = random.RandomState(random_seed)

        # Define função de decaimento default
        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, max_iter: x/(1+t/max_iter)

        self._learning_rate = learning_rate
        self._sigma = sigma
        # Inicializa pesos aleatóriamente
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        for i in range(x):
            for j in range(y):
                # Normaliza os pesos
                norm = fast_norm(self._weights[i, j])
                self._weights[i, j] = self._weights[i, j] / norm

        # Cria mapa de ativação do tamanho do lattice
        self._activation_map = zeros((x, y))
        self._neigx = arange(x)
        self._neigy = arange(y)  # used to evaluate the neighborhood function

        # Define função de vizinhança
        neig_functions = {'gaussian': self._gaussian, 'mexican_hat': self._mexican_hat}
        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function, ', '.join(neig_functions.keys())))
        self.neighborhood = neig_functions[neighborhood_function]

    # Inicia pesos aleatoriamente
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

    def get_weights(self):
        """Return the weights of the neural network."""
        return self._weights

    # Parametro T que será usado para ajustar o sigma e alfa
    def _init_T(self, num_iteration):
        """Initializes the parameter T needed to adjust the learning rate"""
        # keeps the learning rate nearly constant
        # for the last half of the iterations
        self.T = num_iteration/2

    # Treinamento usando batch seguindo sequencialmente os dados de input
    def train_batch(self, data, num_iteration):
        """Trains using all the vectors in data sequentially"""
        self._init_T(len(data)*num_iteration)
        iteration = 0
        calculate_error = 10
        while iteration < num_iteration:
            if calculate_error == 10:
                erro_quantizacao = self.quantization_error(data)
                print('Iteracao: ' + iteration + ' erro quantizacao: ' + erro_quantizacao)
                calculate_error = 0
            idx = iteration % (len(data)-1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1
            calculate_error += 1

    # Define função gaussiana
    def _gaussian(self, c, sigma):
        """Return a Gaussian centered in c."""
        d = 2*pi*sigma*sigma
        ax = exp(-power(self._neigx-c[0], 2)/d)
        ay = exp(-power(self._neigy-c[1], 2)/d)
        return outer(ax, ay)  # the external product gives a matrix

    def winner(self, x):
        """Compute the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return unravel_index(self._activation_map.argmin(), self._activation_map.shape)

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
        # Ajusta decaimento do alfa e do sigma
        eta = self._decay_function(self._learning_rate, t, self.T)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, self.T)
        # improves the performances

        # Calcula nova vizinhança baseada na função de decaimento do neuronio vencedor e o novo sigma * a função de aprendizagem
        g = self.neighborhood(win, sig)*eta
        it = nditer(g, flags=['multi_index'])

        # Aplica a moviamentação para todos os pesos que foram afetados pela rede de vizinhança
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            x_w = (x - self._weights[it.multi_index])
            self._weights[it.multi_index] += g[it.multi_index] * x_w
            # normalization
            norm = fast_norm(self._weights[it.multi_index])
            self._weights[it.multi_index] = self._weights[it.multi_index]/norm
            it.iternext()

    # Calcula a quantização para cada neuronio vencedor
    def quantization(self, data):
        """Assign a code book (weights vector of the winning neuron) to each sample in data."""
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self._weights[self.winner(x)]
        return q

    # Não é utilizado
    def _mexican_hat(self, c, sigma):
        """Mexican hat centered in c."""
        xx, yy = meshgrid(self._neigx, self._neigy)
        p = power(xx-c[0], 2) + power(yy-c[1], 2)
        d = 2*pi*sigma*sigma
        return exp(-p/d)*(1-2/d*p)

    # Não utilizamos
    def train_random(self, data, num_iteration):
        """Trains the SOM picking samples at random from data"""
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            # print("[Treinando SOM: " + str(iteration/num_iteration) + "% COMPLETO]")
            # pick a random sample
            # Pega um input aleatorio
            rand_i = self._random_generator.randint(len(data))
            # Atualiza o lattice através do calculo do vendecor para aquele input de dado
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

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

    # Erro de quantização: calcula média da distancia entre cada input e seu respectivo BMU
    def quantization_error(self, data):
        """Returns the quantization error computed as the average distance between each input sample and its best matching unit."""
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
        table2 = str.maketrans('', '', string.digits)
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
                stripped = [word.translate(table).translate(table2).lower() for word in word_tokenize(sentence) if not word in stop_words]
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
from numpy import dot
from numpy.linalg import norm

class KMeans():
    """."""

    def __init__(self, points, type_of_kmeans='default', distance_type='euclidian'):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.
        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
        self.distance_type = distance_type
        self.points = points
        self.labels = []
	## uma lista contendo os centroids mais proximos de cada ponto
        self.lista_centroid_mais_proximos = None
        self.plotter = KMeansPlotter()

    def inicia_centroides(self, k_centroids):
        """."""
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        """."""
        centroids_redimensionado = self.centroids[:, np.newaxis , :]
        if self.distance_type == 'euclidian':
            #eleva-se a diferença ao quadrado
            diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
            #soma as diferenças e faz a raiz delas, obtendo as distancias euclidianas de todos os pontos para todos os centroids
            distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
            #identifica o centroid mais próximo de cada ponto
            centroid_mais_proximo = np.argmin(distancias, axis=0)
            print('Shape distancia euclidiana')
            print(distancias.shape)


        if self.distance_type == 'cosine_similarity':
            cos_sim = dot(self.points, centroids_redimensionado)/(norm(self.points)*norm(centroids_redimensionado))
            centroid_mais_proximo = np.argmin(1-cos_sim, axis=0)
            print('Shape distancia cosseno')
            print(cos_sim.shape)

        return centroid_mais_proximo

    def roda_kmeans(self, k_centroids, n_iteracoes_limite = 1000, erro = 0.001, centroid_aleatorio = None):
        """."""
        if centroid_aleatorio is None:
            if self.type_of_kmeans == 'kmeans++':
                self.inicia_centroides(1)
                self.inicia_kmeanspp(k_centroids-1)
            else:
                self.inicia_centroides(k_centroids)
        else:
            self.centroids = centroid_aleatorio


        MediaDistAnterior = 0.0
        MediaDistAtual = positive_infinite

        nIteracoes = 0
        while((nIteracoes < n_iteracoes_limite) and abs(MediaDistAnterior - MediaDistAtual) > erro):
            # Só executa se a lista de centroids não tiver sido determinada na ultima iteração
            nIteracoes += 1
            print("quantidade de iterações igual à " + str(nIteracoes))
            print(str(abs(MediaDistAnterior - MediaDistAtual)))
            if(self.lista_centroid_mais_proximos is None):
                self.lista_centroid_mais_proximos = self.busca_centroides_mais_proximo()
            #movimenta os centroids  a partir da lista adquirida na ultima iteração
            self.centroids = self.movimenta_centroides(self.lista_centroid_mais_proximos)
            MediaDistAnterior = MediaDistAtual
            #atualiza lista de centroids mais proximos e calcula a média da distancia entre os pontos e
            #os centroids mais proximos
            MediaDistAtual = self.calculaMediaDistancias()
            self.plotter.plots(self)

    def movimenta_centroides(self, closest):
        """."""
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])

    def calculaMediaDistancias(self ):

        centroids_redimensionado = self.centroids[:, np.newaxis , :]
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        self.lista_centroid_mais_proximos = np.argmin(distancias, axis=0)

        listaDistancias = [0.0]*len(self.centroids)
        indexlista = 0
        #soma todas as distâncias entre os pontos e os centroids mais próximos
        for centroid in self.lista_centroid_mais_proximos:
            listaDistancias[centroid] += distancias[centroid][indexlista]
            indexlista += 1
        #tira a média da distância entre os pontos e os centroids
        return sum(listaDistancias)

    def inicia_kmeanspp(self, centroids_pedidos):
        """."""
        # Gera uma lista de probabilidade para cada ponto
        lista_distancias_normalizadas = np.zeros(self.points.shape[0])
        # print(self.points.shape)
        # Rodamos um loop o numero de vezes que queremos de centroids
        for novo_centroid in range(0, centroids_pedidos):
            print("-- Escolhendo centroide " + str(novo_centroid+2))
            # Redimensionamos o array com os centroids
            centroids_redimensionado = self.centroids[:, np.newaxis, :]
            print(centroids_redimensionado.shape)
            # Elevamos a diferença de todos os pontos, a um centroi especifico, ao quadrado
            diffCordenadasAoQuadrado = (self.points - centroids_redimensionado[novo_centroid]) ** 2
            # Calculamos a soma da raiz para as diferenças em cada dimensão de um ponto
            distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=1))
            # print('distancias')
            # print(distancias.shape)
            # print(distancias)
            # Calculamos a distancia total
            distancia_total = np.sum(distancias, axis=0)
            # Normalizamos as distancias
            lista_distancias_normalizadas = np.zeros(self.points.shape[0])
            lista_distancias_normalizadas = lista_distancias_normalizadas + (distancias / distancia_total)
            # print('lista distancias')
            # print(lista_distancias_normalizadas.shape)
            # print(np.sum(lista_distancias_normalizadas, axis=0))
            # Rodamos a probabilidade
            # print('Escolha probabilistica!')
            rand = random.choice(np.array(self.points.shape[0]), p=lista_distancias_normalizadas)
            # print(rand)
            # print(np.random.choice(lista_distancias_normalizadas, p=lista_distancias_normalizadas))
            # Adicionamos um novo centroid com base no ponto que foi selecionado
            ponto_escolhido = self.points[rand]
            ponto_escolhido = ponto_escolhido[np.newaxis, :]
            # print('Comparacao')
            # print(self.centroids.shape)
            # print(ponto_escolhido.shape)
            self.centroids = np.append(self.centroids, ponto_escolhido, axis=0)
            # print('Resultado')
            # print(self.centroids.shape)

class Xmeans():

    def __init__(self, points):
        """."""
        self.points = points
        self.labels = []

    def roda_xmeans(self, n_iteracoes = 100, trim_percentage = 0.9):
        """."""
        num_centroids = 2

        # Instância para controle do K-Means global
        global_kmeans = KMeans(self.points)
        global_kmeans.roda_kmeans(num_centroids, n_iteracoes)

        # Instância utilizada para K-Means locais
        local_kmeans = KMeans(self.points)
        for iter in range(n_iteracoes):

            # Pais que não vale a pena dividir em filhos
            ultimate_fathers = []

            # Pais cujos filhos são melhores que ele
            fathers_to_pop = []

            for i in range(num_centroids):

                # Como não vale a pena dividir, vai pro próximo pai
                if i in ultimate_fathers:
                    continue
                    
                points_centroid_father = get_centroid_points(i, global_kmeans.points, global_kmeans.labels)
                father_labels = [i for j in range(len(points_centroid_father))]
                father_centroid = global_kmeans.centroids[i]

                # BIC dos centróide pai
                bic_father = self.compute_bic(points_centroid_father, father_centroid, father_labels, 1)

                # O número representa quanto do range dos pontos será utilizado
                new_centroids = self.get_two_new_centroids(trim_percentage, father_centroid, global_kmeans.points)

                # Executa K-Means local para dois filhos
                local_kmeans.lista_centroid_mais_proximos = None
                local_kmeans.points = points_centroid_father
                local_kmeans.roda_kmeans(2, new_centroids)

                # BIC dos centróides filhos
                bic_children = self.compute_bic(local_kmeans.points, local_kmeans.centroids, local_kmeans.labels, 2)

                # Se bic_children melhor que bic_pai, guarda índice do pai para ser removido
                # e coloca as crianças na lista de centroids.
                # Caso contrário guarda índice do pai no array de pais que não serão mais avaliados
                if bic_children > bic_pai:
                    fathers_to_pop.append(i)
                    global_kmeans.centroids.extend(local_kmeans.centroids)
                else:
                    ultimate_fathers.append(i)

            # Se nenhum pai virou dois filhos, os centróides são as melhores representações dos dados
            if not fathers_to_pop:
                return
            
            # Remove os pais, com índices guardados no father_to_pop, e atualiza número de centróides
            for i in range(len(fathers_to_pop)):
                global_kmeans.centroids.pop(fathers_to_pop[i])
            num_centroids = len(global_kmeans.centroids)


    # Divisão de centróides em 2

    def get_centroid_points(self, i, param_points = None, param_labels = None):
        if param_points is None:
            param_points = self.points
        if param_labels is None:
            param_labels = self.labels

        points_per_centroid = []

        for k in range(len(param_labels)):
            if (param_labels[k] == i):
                points_per_centroid.append(param_points[k])

        return points_per_centroid

    def get_two_new_centroids(self, trim_percentage, father_centroid, param_points = None):
        if param_points is None:
            param_points = self.points

        one_centroid = []
        two_centroid = []

        for i in range(len(param_points)):
            range_dimension = max(param_points[i]) - min(param_points[i])
            range_dimension = range_dimension * trim_percentage
            range_divided = range_dimension / 2

            one_centroid.append(father_centroid[i] - range_divided)
            two_centroid.append(father_centroid[i] + range_divided)

        return [one_centroid, two_centroid]

    # Método para avaliação dos modelos de centróides

    def compute_bic(X, centers, labels, K):
        """Computes the BIC metric for a given clusters

        Parameters:
        -----------------------------------------
        kmeans:  List of clustering object from scikit learn

        X     :  multidimension np array of data points

        Returns:
        -----------------------------------------
        BIC value"""

        # número de pontos contidos em cada centróide
        n = np.bincount(labels)

        # size of data set
        R, d = X.shape

        # compute variance for all clusters beforehand
        cl_var = (1.0 / (R - K) / d) * sum([sum(distance.cdist(X[np.where(labels == i)], [centers[0][i]], 'euclidean')**2) for i in range(K)])

        const_term = 0.5 * K * np.log(R) * (d + 1)

        BIC = np.sum([
                    n[i] * np.log(n[i]) -
                    n[i] * np.log(R) -
                    ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                    ((n[i] - 1) * d / 2) for i in range(K)
                    ]) - const_term

        return(BIC)

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
		preprocessor = ProcessTexts(texts=['bbc_kaggle'])
		print('----- Transformando Tokens em Matriz -----')
		matrix = TransformMatrix(preprocessor.tokens)
		print('----- Resultados do bag of words -----')
		dados = matrix.get_matrix(type='tf-idf')

		# ---------------------
		# K-means
		# print('----- Iniciando Processamento K-means -----')
		# kmeans = KMeans(dados)
		# kmeans.roda_kmeans(3/)

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
