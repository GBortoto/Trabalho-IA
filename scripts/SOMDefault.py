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

        markers = ['o', 's', 'D']
        colors = ['r', 'g', 'b']
        for cnt, xx in enumerate(dados):
            w = self.winner(xx)  # getting the winner
            # palce a marker on the winning position for the sample xx
            plt.plot(w[0]+.5, w[1]+.5, markers[t[cnt]], markerfacecolor='None',
                     markeredgecolor=colors[t[cnt]], markersize=12, markeredgewidth=2)
        plt.axis([0, 7, 0, 7])
        plt.show()

    def plot3(self):
        starting_weights = self.get_weights().copy()  # saving the starting weights
        print('quantization...')
        qnt = self.quantization(pixels)  # quantize each pixels of the image
        print('building new image...')
        clustered = np.zeros(img.shape)
        for i, q in enumerate(qnt):  # place the quantized values into a new image
            clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q
        print('done.')

        # show the result
        plt.figure(1)
        plt.subplot(221)
        plt.title('original')
        plt.imshow(img)
        plt.subplot(222)
        plt.title('result')
        plt.imshow(clustered)

        plt.subplot(223)
        plt.title('initial colors')
        plt.imshow(starting_weights, interpolation='none')
        plt.subplot(224)
        plt.title('learned colors')
        plt.imshow(self.get_weights(), interpolation='none')

        plt.tight_layout()
        plt.show()

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

    def plot5(self):
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
