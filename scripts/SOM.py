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
