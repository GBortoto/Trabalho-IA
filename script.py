

class View(object):
    def __init__(self, width, height, title, show_axis=True, packed=True,
                 text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        self.width = width
        self.height = height
        self.title = title
        self.show_axis = show_axis
        self.packed = packed
        self.text_size = text_size
        self.show_text = show_text
        self.col_size = col_size

    def prepare(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, filename):
        raise NotImplementedError()

    def show(self, *args, **kwrags):
        raise NotImplementedError()


class MatplotView(View):

    def __init__(self, width, height, title, show_axis=True, packed=True,
                 text_size=2.8, show_text=True, col_size=6, *args, **kwargs):
        super(MatplotView, self).__init__(width, height, title, show_axis,
                                          packed, text_size, show_text,
                                          col_size, *args, **kwargs)
        self._fig = None

    def __del__(self):
        self._close_fig()

    def _close_fig(self):
        if self._fig:
            plt.close(self._fig)

    def prepare(self, *args, **kwargs):
        self._close_fig()
        self._fig = plt.figure(figsize=(self.width, self.height))
        plt.title(self.title)
        plt.axis('off')
        plt.rc('font', **{'size': self.text_size})

    def save(self, filename, transparent=False, bbox_inches='tight', dpi=400):
        self._fig.savefig(filename, transparent=transparent, dpi=dpi,
                          bbox_inches=bbox_inches)

    def show(self, *args, **kwrags):
        raise NotImplementedError()

import logging
from functools import wraps
from time import time


def timeit(log_level=logging.INFO, alternative_title=None):
    def wrap(f):
        @wraps(f)  # keeps the f.__name__ outside the wrapper
        def wrapped_f(*args, **kwargs):
            t0 = time()
            result = f(*args, **kwargs)
            ts = round(time() - t0, 3)

            title = alternative_title or f.__name__
            logging.getLogger().log(
                log_level, " %s took: %f seconds" % (title, ts))

            return result

        return wrapped_f
    return wrap
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

import matplotlib
# from .view import MatplotView
from matplotlib import pyplot as plt
import numpy as np
# import ipdb


class MapView(MatplotView):

    def _calculate_figure_params(self, som, which_dim, col_sz):

        # add this to avoid error when normalization is not used
        if som._normalizer:
            codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)
        else:
            codebook = som.codebook.matrix

        indtoshow, sV, sH = None, None, None

        if which_dim == 'all':
            dim = som._dim
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.arange(0, dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap

        elif type(which_dim) == int:
            dim = 1
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = 16, 16 * ratio_hitmap

        elif type(which_dim) == list:
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16*ratio_fig*ratio_hitmap

        no_row_in_plot = dim / col_sz + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        width = sH
        height = sV

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)


class View2D(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None, desnormalize=False):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        self.prepare()

        if not desnormalize:
            codebook = som.codebook.matrix
        else:
            codebook = som._normalizer.denormalize_by(som.data_raw, som.codebook.matrix)

        if which_dim == 'all':
            names = som._component_names[0]
        elif type(which_dim) == int:
            names = [som._component_names[0][which_dim]]
        elif type(which_dim) == list:
            names = som._component_names[0][which_dim]


        while axis_num < len(indtoshow):
            axis_num += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])

            min_color_scale = np.mean(codebook[:, ind].flatten()) - 1 * np.std(codebook[:, ind].flatten())
            max_color_scale = np.mean(codebook[:, ind].flatten()) + 1 * np.std(codebook[:, ind].flatten())
            min_color_scale = min_color_scale if min_color_scale >= min(codebook[:, ind].flatten()) else \
                min(codebook[:, ind].flatten())
            max_color_scale = max_color_scale if max_color_scale <= max(codebook[:, ind].flatten()) else \
                max(codebook[:, ind].flatten())
            norm = matplotlib.colors.Normalize(vmin=min_color_scale, vmax=max_color_scale, clip=True)

            mp = codebook[:, ind].reshape(som.codebook.mapsize[0],
                                          som.codebook.mapsize[1])
            pl = plt.pcolor(mp[::-1], norm=norm)
            plt.axis([0, som.codebook.mapsize[1], 0, som.codebook.mapsize[0]])
            plt.title(names[axis_num - 1])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.colorbar(pl)

        #plt.show()


class View2DPacked(MapView):

    def _set_axis(self, ax, msz0, msz1):
        plt.axis([0, msz0, 0, msz1])
        plt.axis('off')
        ax.axis('off')

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        if col_sz is None:
            col_sz = 6
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        codebook = som.codebook.matrix

        cmap = cmap or plt.cm.get_cmap('RdYlBu_r')
        msz0, msz1 = som.codebook.mapsize
        compname = som.component_names
        if what == 'codebook':
            h = .1
            w = .1
            self.width = no_col_in_plot*2.5*(1+w)
            self.height = no_row_in_plot*2.5*(1+h)
            self.prepare()

            while axis_num < len(indtoshow):
                axis_num += 1
                ax = self._fig.add_subplot(no_row_in_plot, no_col_in_plot,
                                           axis_num)
                ax.axis('off')
                ind = int(indtoshow[axis_num-1])
                mp = codebook[:, ind].reshape(msz0, msz1)
                plt.imshow(mp[::-1], norm=None, cmap=cmap)
                self._set_axis(ax, msz0, msz1)

                if self.show_text is True:
                    plt.title(compname[0][ind])
                    font = {'size': self.text_size}
                    plt.rc('font', **font)
        if what == 'cluster':
            try:
                codebook = getattr(som, 'cluster_labels')
            except:
                codebook = som.cluster()

            h = .2
            w = .001
            self.width = msz0/2
            self.height = msz1/2
            self.prepare()

            ax = self._fig.add_subplot(1, 1, 1)
            mp = codebook[:].reshape(msz0, msz1)
            plt.imshow(mp[::-1], cmap=cmap)

            self._set_axis(ax, msz0, msz1)

        plt.subplots_adjust(hspace=h, wspace=w)

        plt.show()


class View1D(MapView):

    def show(self, som, what='codebook', which_dim='all', cmap=None,
             col_sz=None):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, which_dim, col_sz)
        self.prepare()

        codebook = som.codebook.matrix

        while axis_num < len(indtoshow):
            axis_num += 1
            plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])
            mp = codebook[:, ind]
            plt.plot(mp, '-k', linewidth=0.8)

        #plt.show()
"""."""

# -*- coding: utf-8 -*-

# Author: Vahid Moosavi (sevamoo@gmail.com)
#         Chair For Computer Aided Architectural Design, ETH  Zurich
#         Future Cities Lab
#         www.vahidmoosavi.com

# Contributor: Sebastian Packmann (sebastian.packmann@gmail.com)


import tempfile
import os
import itertools
import logging
import numpy as np
from time import time
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count
from scipy.sparse import csr_matrix
from sklearn import neighbors
from sklearn.externals.joblib import Parallel, delayed, load, dump
import sys


class ComponentNamesError(Exception):
    """."""

    pass


class LabelsError(Exception):
    """."""

    pass


class SOMFactory(object):
    """."""

    @staticmethod
    def build(data, mapsize=None, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy', component_names=None):
        """Fabrica de SOM.

        :param data: dados que precisam ser clusterizados, representados por uma matriz n x m
        :param mapsize: lista que define o numero de n linhas e m colunas da parametro data
        :param mask: mask
        :param mapshape: formato planar do lattice
        :param lattice: formato retangular do lattice
        :param normalization: normalizacaoo a partir da variancia dos objetos de input
        :param initialization: metodo utilizado para inicializar o SOM. Opcoes sao: 1 - "pca" ; 2 - "random"
        :param neighborhood: funcao para calculo de vizinhanca do lattice. Opcoes sao: 1 - "gaussian"; 2 - "bubble"
        :param training: modo de treinamento. Opcoes sao: 1 - "seq"; 2 - "batch"
        :param name: nome usado para identificar a rede SOM
        """
        normalizer = NormalizatorFactory.build(normalization)
        neighborhood_calculator = NeighborhoodFactory.build(neighborhood)
        return SOM(data, neighborhood_calculator, normalizer, mapsize, mask, mapshape, lattice, initialization, training, name, component_names)


class SOM(object):
    """."""

    def __init__(self, data, neighborhood, normalizer=None, mapsize=None, mask=None, mapshape='planar', lattice='rect', initialization='pca', training='batch', name='sompy', component_names=None):
        """Self Organizing Map."""
        if normalizer:
            me, st = np.mean(data, axis=0), np.std(data, axis=0)
            st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
            self._data = (data-me)/st
        else:
            self._data = data

        # Objeto responsavel por normalizar os dados
        self._normalizer = normalizer

        # Dimensao é igual ao #colunas do dataset
        self._dim = data.shape[1]

        # Length é igual ao #linhas do dataset
        self._dlen = data.shape[0]

        # Defnimos como None o data_label e o neuronio BMU
        self._dlabel = None
        self._bmu = None

        # Guardamos os dados puros
        self.data_raw = data

        # Guarda objeto responsável por fazer a função de vizinhança
        self.neighborhood = neighborhood
        self.mapshape = mapshape
        self.initialization = initialization
        self.mask = mask or np.ones([1, self._dim])
        self.codebook = Codebook(mapsize, lattice)
        self.training = training
        self._component_names = self.build_component_names() if component_names is None else [component_names]
        self._distance_matrix = self.calculate_map_dist()
        self.name = name

        mapsize = self.calculate_map_size(lattice) if not mapsize else mapsize

    @property
    def component_names(self):
        return self._component_names

    @component_names.setter
    def component_names(self, compnames):
        if self._dim == len(compnames):
            self._component_names = np.asarray(compnames)[np.newaxis, :]
        else:
            raise ComponentNamesError('Component names should have the same '
                                      'size as the data dimension/features')

    def build_component_names(self):
        cc = ['Variable-' + str(i+1) for i in range(0, self._dim)]
        return np.asarray(cc)[np.newaxis, :]

    @property
    def data_labels(self):
        return self._dlabel

    @data_labels.setter
    def data_labels(self, labels):
        """
        Set labels of the training data, it should be in the format of a list
        of strings
        """
        if labels.shape == (1, self._dlen):
            label = labels.T
        elif labels.shape == (self._dlen, 1):
            label = labels
        elif labels.shape == (self._dlen,):
            label = labels[:, np.newaxis]
        else:
            raise LabelsError('wrong label format')

        self._dlabel = label

    def build_data_labels(self):
        cc = ['dlabel-' + str(i) for i in range(0, self._dlen)]
        return np.asarray(cc)[:, np.newaxis]

    def calculate_map_dist(self):
        """
        Calculates the grid distance, which will be used during the training
        steps. It supports only planar grids for the moment
        """
        nnodes = self.codebook.nnodes

        distance_matrix = np.zeros((nnodes, nnodes))
        for i in range(nnodes):
            distance_matrix[i] = self.codebook.grid_dist(i).reshape(1, nnodes)
        return distance_matrix

    @timeit()
    def train(self,
              n_job=1,
              shared_memory=False,
              verbose='info',
              train_rough_len=None,
              train_rough_radiusin=None,
              train_rough_radiusfin=None,
              train_finetune_len=None,
              train_finetune_radiusin=None,
              train_finetune_radiusfin=None,
              train_len_factor=1,
              maxtrainlen=np.Inf):
        """
        Trains the som

        :param n_job: number of jobs to use to parallelize the traning
        :param shared_memory: flag to active shared memory
        :param verbose: verbosity, could be 'debug', 'info' or None
        :param train_len_factor: Factor that multiply default training lenghts (similar to "training" parameter in the matlab version). (lbugnon)
        """
        logging.root.setLevel(
            getattr(logging, verbose.upper()) if verbose else logging.ERROR)

        logging.info("Treinamento em Processo...")
        logging.debug((
            "--------------------------------------------------------------\n"
            " details: \n"
            "      > data len is {data_len} and data dimension is {data_dim}\n"
            "      > map size is {mpsz0},{mpsz1}\n"
            "      > array size in log10 scale is {array_size}\n"
            "      > number of jobs in parallel: {n_job}\n"
            " -------------------------------------------------------------\n")
            .format(data_len=self._dlen,
                    data_dim=self._dim,
                    mpsz0=self.codebook.mapsize[0],
                    mpsz1=self.codebook.mapsize[1],
                    array_size=np.log10(
                        self._dlen * self.codebook.nnodes * self._dim),
                    n_job=n_job))

        if self.initialization == 'random':
            self.codebook.random_initialization(self._data)

        elif self.initialization == 'pca':
            self.codebook.pca_linear_initialization(self._data)

        self.rough_train(njob=n_job, shared_memory=shared_memory, trainlen=train_rough_len,
                         radiusin=train_rough_radiusin, radiusfin=train_rough_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        self.finetune_train(njob=n_job, shared_memory=shared_memory, trainlen=train_finetune_len,
                            radiusin=train_finetune_radiusin, radiusfin=train_finetune_radiusfin,trainlen_factor=train_len_factor,maxtrainlen=maxtrainlen)
        logging.debug(
            " --------------------------------------------------------------")
        logging.info("Erro final de quantização: %f" % np.mean(self._bmu[1]))

    def _calculate_ms_and_mpd(self):
        mn = np.min(self.codebook.mapsize)
        max_s = max(self.codebook.mapsize[0], self.codebook.mapsize[1])

        if mn == 1:
            mpd = float(self.codebook.nnodes*10)/float(self._dlen)
        else:
            mpd = float(self.codebook.nnodes)/float(self._dlen)
        ms = max_s/2.0 if mn == 1 else max_s

        return ms, mpd

    def rough_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info("[Treinamento Pesado]")

        ms, mpd = self._calculate_ms_and_mpd()
        #lbugnon: add maxtrainlen
        trainlen = min(int(np.ceil(30*mpd)),maxtrainlen) if not trainlen else trainlen
        #print("maxtrainlen %d",maxtrainlen)
        #lbugnon: add trainlen_factor
        trainlen=int(trainlen*trainlen_factor)

        if self.initialization == 'random':
            radiusin = max(1, np.ceil(ms/3.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/6.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            radiusin = max(1, np.ceil(ms/8.)) if not radiusin else radiusin
            radiusfin = max(1, radiusin/4.) if not radiusfin else radiusfin

        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def finetune_train(self, njob=1, shared_memory=False, trainlen=None, radiusin=None, radiusfin=None,trainlen_factor=1,maxtrainlen=np.Inf):
        logging.info("[Treinamento de Ajuste]")

        ms, mpd = self._calculate_ms_and_mpd()

        #lbugnon: add maxtrainlen
        if self.initialization == 'random':
            trainlen = min(int(np.ceil(50*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, ms/12.)  if not radiusin else radiusin # from radius fin in rough training
            radiusfin = max(1, radiusin/25.) if not radiusfin else radiusfin

        elif self.initialization == 'pca':
            trainlen = min(int(np.ceil(40*mpd)),maxtrainlen) if not trainlen else trainlen
            radiusin = max(1, np.ceil(ms/8.)/4) if not radiusin else radiusin
            radiusfin = 1 if not radiusfin else radiusfin # max(1, ms/128)

        #print("maxtrainlen %d",maxtrainlen)

        #lbugnon: add trainlen_factor
        trainlen=int(trainlen_factor*trainlen)


        self._batchtrain(trainlen, radiusin, radiusfin, njob, shared_memory)

    def _batchtrain(self, trainlen, radiusin, radiusfin, njob=1,
                    shared_memory=False):
        radius = np.linspace(radiusin, radiusfin, trainlen)

        if shared_memory:
            data = self._data
            data_folder = tempfile.mkdtemp()
            data_name = os.path.join(data_folder, 'data')
            dump(data, data_name)
            data = load(data_name, mmap_mode='r')

        else:
            data = self._data

        bmu = None

        # X2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use
        # for each data row in bmu finding.
        # Since it is a fixed value we can skip it during bmu finding for each
        # data point, but later we need it calculate quantification error
        fixed_euclidean_x2 = np.einsum('ij,ij->i', data, data)

        logging.info(" radius_ini: %f , radius_final: %f, trainlen: %d\n" %
                     (radiusin, radiusfin, trainlen))

        for i in range(trainlen):
            t1 = time()
            neighborhood = self.neighborhood.calculate(
                self._distance_matrix, radius[i], self.codebook.nnodes)
            bmu = self.find_bmu(data, njb=njob)
            self.codebook.matrix = self.update_codebook_voronoi(data, bmu,
                                                                neighborhood)

            qerror = (i + 1, round(time() - t1, 3), np.mean(np.sqrt(bmu[1] + fixed_euclidean_x2))) #lbugnon: ojo aca me tiró un warning, revisar (commit sinc: 965666d3d4d93bcf48e8cef6ea2c41a018c1cb83 )
            logging.info("[epoca %d] tempo:  %f, erro de quantização: %f\n" %qerror)
            if np.any(np.isnan(qerror)):
                logging.info("nan quantization error, exit train\n")

        bmu[1] = np.sqrt(bmu[1] + fixed_euclidean_x2)
        self._bmu = bmu

    @timeit(logging.DEBUG)
    def find_bmu(self, input_matrix, njb=1, nth=1):
        """
        Finds the best matching unit (bmu) for each input data from the input
        matrix. It does all at once parallelizing the calculation instead of
        going through each input and running it against the codebook.

        :param input_matrix: numpy matrix representing inputs as rows and
            features/dimension as cols
        :param njb: number of jobs to parallelize the search
        :returns: the best matching unit for each input
        """
        dlen = input_matrix.shape[0]
        y2 = np.einsum('ij,ij->i', self.codebook.matrix, self.codebook.matrix)
        if njb == -1:
            njb = cpu_count()

        pool = Pool(njb)
        chunk_bmu_finder = _chunk_based_bmu_find

        def row_chunk(part):
            return part * dlen // njb

        def col_chunk(part):
            return min((part+1)*dlen // njb, dlen)

        chunks = [input_matrix[row_chunk(i):col_chunk(i)] for i in range(njb)]
        b = pool.map(lambda chk: chunk_bmu_finder(chk, self.codebook.matrix, y2, nth=nth), chunks)
        pool.close()
        pool.join()
        bmu = np.asarray(list(itertools.chain(*b))).T
        del b
        return bmu

    @timeit(logging.DEBUG)
    def update_codebook_voronoi(self, training_data, bmu, neighborhood):
        """
        Updates the weights of each node in the codebook that belongs to the
        bmu's neighborhood.

        First finds the Voronoi set of each node. It needs to calculate a
        smaller matrix.
        Super fast comparing to classic batch training algorithm, it is based
        on the implemented algorithm in som toolbox for Matlab by Helsinky
        University.

        :param training_data: input matrix with input vectors as rows and
            vector features as cols
        :param bmu: best matching unit for each input data. Has shape of
            (2, dlen) where first row has bmu indexes
        :param neighborhood: matrix representing the neighborhood of each bmu

        :returns: An updated codebook that incorporates the learnings from the
            input data
        """
        row = bmu[0].astype(int)
        col = np.arange(self._dlen)
        val = np.tile(1, self._dlen)
        P = csr_matrix((val, (row, col)), shape=(self.codebook.nnodes,
                       self._dlen))
        S = P.dot(training_data)

        # neighborhood has nnodes*nnodes and S has nnodes*dim
        # ---> Nominator has nnodes*dim
        nom = neighborhood.T.dot(S)
        nV = P.sum(axis=1).reshape(1, self.codebook.nnodes)
        denom = nV.dot(neighborhood.T).reshape(self.codebook.nnodes, 1)
        new_codebook = np.divide(nom, denom)

        return np.around(new_codebook, decimals=6)

    def project_data(self, data):
        """
        Projects a data set to a trained SOM. It is based on nearest
        neighborhood search module of scikitlearn, but it is not that fast.
        """
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        labels = np.arange(0, self.codebook.matrix.shape[0])
        clf.fit(self.codebook.matrix, labels)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        data = self._normalizer.normalize_by(self.data_raw, data)

        return clf.predict(data)

    def predict_by(self, data, target, k=5, wt='distance'):
        # here it is assumed that target is the last column in the codebook
        # and data has dim-1 columns
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != target]
        x = self.codebook.matrix[:, indX]
        y = self.codebook.matrix[:, target]
        n_neighbors = k
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indX]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indX], data)

        predicted_values = clf.predict(data)
        predicted_values = self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)
        return predicted_values

    def predict(self, x_test, k=5, wt='distance'):
        """
        Similar to SKlearn we assume that we have X_tr, Y_tr and X_test. Here
        it is assumed that target is the last column in the codebook and data
        has dim-1 columns

        :param x_test: input vector
        :param k: number of neighbors to use
        :param wt: method to use for the weights
            (more detail in KNeighborsRegressor docs)
        :returns: predicted values for the input data
        """
        target = self.data_raw.shape[1]-1
        x_train = self.codebook.matrix[:, :target]
        y_train = self.codebook.matrix[:, target]
        clf = neighbors.KNeighborsRegressor(k, weights=wt)
        clf.fit(x_train, y_train)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        x_test = self._normalizer.normalize_by(
            self.data_raw[:, :target], x_test)
        predicted_values = clf.predict(x_test)

        return self._normalizer.denormalize_by(
            self.data_raw[:, target], predicted_values)

    def find_k_nodes(self, data, k=5):
        from sklearn.neighbors import NearestNeighbors
        # we find the k most similar nodes to the input vector
        neighbor = NearestNeighbors(n_neighbors=k)
        neighbor.fit(self.codebook.matrix)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        return neighbor.kneighbors(
            self._normalizer.normalize_by(self.data_raw, data))

    def bmu_ind_to_xy(self, bmu_ind):
        """
        Translates a best matching unit index to the corresponding
        matrix x,y coordinates.

        :param bmu_ind: node index of the best matching unit
            (number of node from top left node)
        :returns: corresponding (x,y) coordinate
        """
        rows = self.codebook.mapsize[0]
        cols = self.codebook.mapsize[1]

        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bmu_ind.shape[0], 3))
        out[:, 2] = bmu_ind
        out[:, 0] = rows-1-bmu_ind / cols
        out[:, 0] = bmu_ind / cols
        out[:, 1] = bmu_ind % cols

        return out.astype(int)

    def cluster(self, n_clusters=8):
        import sklearn.cluster as clust
        cl_labels = clust.KMeans(n_clusters=n_clusters).fit_predict(
            self._normalizer.denormalize_by(self.data_raw,
                                            self.codebook.matrix))
        self.cluster_labels = cl_labels
        return cl_labels

    def predict_probability(self, data, target, k=5):
        """
        Predicts probability of the input data to be target

        :param data: data to predict, it is assumed that 'target' is the last
            column in the codebook, so data hould have dim-1 columns
        :param target: target to predict probability
        :param k: k parameter on KNeighborsRegressor
        :returns: probability of data been target
        """
        dim = self.codebook.matrix.shape[1]
        ind = np.arange(0, dim)
        indx = ind[ind != target]
        x = self.codebook.matrix[:, indx]
        y = self.codebook.matrix[:, target]

        clf = neighbors.KNeighborsRegressor(k, weights='distance')
        clf.fit(x, y)

        # The codebook values are all normalized
        # we can normalize the input data based on mean and std of
        # original data
        dimdata = data.shape[1]

        if dimdata == dim:
            data[:, target] = 0
            data = self._normalizer.normalize_by(self.data_raw, data)
            data = data[:, indx]

        elif dimdata == dim-1:
            data = self._normalizer.normalize_by(self.data_raw[:, indx], data)

        weights, ind = clf.kneighbors(data, n_neighbors=k,
                                      return_distance=True)
        weights = 1./weights
        sum_ = np.sum(weights, axis=1)
        weights = weights/sum_[:, np.newaxis]
        labels = np.sign(self.codebook.matrix[ind, target])
        labels[labels >= 0] = 1

        # for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob *= weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]

        # for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        return np.concatenate((pos_prob, neg_prob), axis=1)

    def node_activation(self, data, target=None, wt='distance'):
        weights, ind = None, None

        if not target:
            clf = neighbors.KNeighborsClassifier(
                n_neighbors=self.codebook.nnodes)
            labels = np.arange(0, self.codebook.matrix.shape[0])
            clf.fit(self.codebook.matrix, labels)

            # The codebook values are all normalized
            # we can normalize the input data based on mean and std of
            # original data
            data = self._normalizer.normalize_by(self.data_raw, data)
            weights, ind = clf.kneighbors(data)

            # Softmax function
            weights = 1./weights

        return weights, ind

    def calculate_topographic_error(self):
        bmus1 = self.find_bmu(self.data_raw, njb=1, nth=1)
        bmus2 = self.find_bmu(self.data_raw, njb=1, nth=2)
        bmus_gap = np.abs((self.bmu_ind_to_xy(np.array(bmus1[0]))[:, 0:2] - self.bmu_ind_to_xy(np.array(bmus2[0]))[:, 0:2]).sum(axis=1))
        return np.mean(bmus_gap != 1)

    def calculate_map_size(self, lattice):
        """
        Calculates the optimal map size given a dataset using eigenvalues and eigenvectors. Matlab ported
        :lattice: 'rect' or 'hex'
        :return: map sizes
        """
        D = self.data_raw.copy()
        dlen = D.shape[0]
        dim = D.shape[1]
        munits = np.ceil(5 * (dlen ** 0.5))
        A = np.ndarray(shape=[dim, dim]) + np.Inf

        for i in range(dim):
            D[:, i] = D[:, i] - np.mean(D[np.isfinite(D[:, i]), i])

        for i in range(dim):
            for j in range(dim):
                c = D[:, i] * D[:, j]
                c = c[np.isfinite(c)]
                A[i, j] = sum(c) / len(c)
                A[j, i] = A[i, j]

        eigval = sorted(np.linalg.eig(A)[0])
        if eigval[-1] == 0 or eigval[-2] * munits < eigval[-1]:
            ratio = 1
        else:
            ratio = np.sqrt(eigval[-1] / eigval[-2])

        if lattice == "rect":
            size1 = min(munits, round(np.sqrt(munits / ratio)))
        else:
            size1 = min(munits, round(np.sqrt(munits / ratio*np.sqrt(0.75))))

        size2 = round(munits / size1)

        return [int(size1), int(size2)]


# Since joblib.delayed uses Pickle, this method needs to be a top level
# method in order to be pickled
# Joblib is working on adding support for cloudpickle or dill which will allow
# class methods to be pickled
# when that that comes out we can move this to SOM class
def _chunk_based_bmu_find(input_matrix, codebook, y2, nth=1):
    """
    Finds the corresponding bmus to the input matrix.

    :param input_matrix: a matrix of input data, representing input vector as
                         rows, and vectors features/dimention as cols
                         when parallelizing the search, the input_matrix can be
                         a sub matrix from the bigger matrix
    :param codebook: matrix of weights to be used for the bmu search
    :param y2: <not sure>
    """
    dlen = input_matrix.shape[0]
    nnodes = codebook.shape[0]
    bmu = np.empty((dlen, 2))

    # It seems that small batches for large dlen is really faster:
    # that is because of ddata in loops and n_jobs. for large data it slows
    # down due to memory needs in parallel
    blen = min(50, dlen)
    i0 = 0

    while i0+1 <= dlen:
        low = i0
        high = min(dlen, i0+blen)
        i0 = i0+blen
        ddata = input_matrix[low:high+1]
        d = np.dot(codebook, ddata.T)
        d *= -2
        d += y2.reshape(nnodes, 1)
        bmu[low:high+1, 0] = np.argpartition(d, nth, axis=0)[nth-1]
        bmu[low:high+1, 1] = np.partition(d, nth, axis=0)[nth-1]
        del ddata

    return bmu
import numpy as np
import inspect
import sys

small = .000000000001


class NeighborhoodFactory(object):

    @staticmethod
    def build(neighborhood_func):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and neighborhood_func == obj.name:
                    return obj()
        else:
            raise Exception(
                "Unsupported neighborhood function '%s'" % neighborhood_func)


class GaussianNeighborhood(object):

    name = 'gaussian'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        return np.exp(-1.0*distance_matrix/(2.0*radius**2)).reshape(dim, dim)

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)


class BubbleNeighborhood(object):

    name = 'bubble'

    @staticmethod
    def calculate(distance_matrix, radius, dim):
        def l(a, b):
            c = np.zeros(b.shape)
            c[a-b >= 0] = 1
            return c

        return l(radius,
                 np.sqrt(distance_matrix.flatten())).reshape(dim, dim) + small

    def __call__(self, *args, **kwargs):
        return self.calculate(*args)
# from .view import MatplotView
from matplotlib import pyplot as plt
import numpy as np


class DotMapView(MatplotView):

    def init_figure(self, dim, cols):
        no_row_in_plot = dim/cols + 1
        no_col_in_plot = dim if no_row_in_plot <= 1 else cols
        h = .1
        w = .1
        self.width = no_col_in_plot*2.5*(1+w)
        self.height = no_row_in_plot*2.5*(1+h)
        self.prepare()

    def plot(self, data, coords, msz0, msz1, colormap, dlen, dim, rows, cols):
        for i in range(dim):
            plt.subplot(rows, cols, i+1)

            # This uses the colors uniquely for each record, while in normal
            # views, it is based on the values within each dimensions. This is
            # important when we are dealing with time series. Where we don't
            # want to normalize colors within each time period, rather we like
            # to see the patterns of each data records in time.
            mn = np.min(data[:, :], axis=1)
            mx = np.max(data[:, :], axis=1)

            for j in range(dlen):
                plt.scatter(coords[j, 1],
                            msz0-1-coords[j, 0],
                            c=data[j, i],
                            vmax=mx[j], vmin=mn[j],
                            s=90,
                            marker='.',
                            edgecolor='None',
                            cmap=colormap,
                            alpha=1)

            eps = .0075
            plt.xlim(0-eps, msz1-1+eps)
            plt.ylim(0-eps, msz0-1+eps)
            plt.xticks([])
            plt.yticks([])

    def show(self, som, which_dim='all', colormap=None, cols=None):
        plt.cm.get_cmap(colormap) if colormap else plt.cm.get_cmap('RdYlBu_r')

        data = som.data_raw
        msz0, msz1 = som.codebook.mapsize
        coords = som.bmu_ind_to_xy(som.project_data(data))[:, :2]
        cols = cols if cols else 8  # 8 is arbitrary
        rows = data.shape[1]/cols+1

        if which_dim == 'all':
            dim = data.shape[0]
            self.init_figure(dim, cols)
            self.plot(data, coords, msz0, msz1, colormap, data.shape[0],
                      data.shape[1], rows, cols)

        else:
            dim = 1 if type(which_dim) is int else len(which_dim)
            self.init_figure(dim, cols)
            self.plot(data, coords, msz0, msz1, colormap, data.shape[0],
                      len(which_dim), rows, cols)

        plt.tight_layout()
        plt.subplots_adjust(hspace=.16, wspace=.05)

class KMeans():
    """."""

    def __init__(self, points, type_of_kmeans='default'):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.
        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
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
        #eleva-se a diferença ao quadrado
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        #soma as diferenças e faz a raiz delas, obtendo as distancias euclidianas de todos os pontos para todos os centroids
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        #identifica o centroid mais próximo de cada ponto
        centroid_mais_proximo = np.argmin(distancias, axis=0)

        return centroid_mais_proximo

    def roda_kmeans(self, k_centroids, n_iteracoes_limite = 1000, erro = 0.001, centroid_aleatorio = None):
        """."""
        if centroid_aleatorio is None:
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
# from .view import MatplotView
from matplotlib import pyplot as plt
import numpy as np


class HitMapView(MatplotView):

    def _set_labels(self, cents, ax, labels):
        for i, txt in enumerate(labels):
            ax.annotate(txt, (cents[i, 1], cents[i, 0]), size=10, va="center")

    def show(self, som, data=None):

        try:
            codebook = getattr(som, 'cluster_labels')
        except:
            codebook = som.cluster()

        # codebook = getattr(som, 'cluster_labels', som.cluster())
        msz = som.codebook.mapsize

        self.prepare()
        ax = self._fig.add_subplot(111)

        if data:
            proj = som.project_data(data)
            cents = som.bmu_ind_to_xy(proj)
            self._set_labels(cents, ax, codebook[proj])

        else:
            cents = som.bmu_ind_to_xy(np.arange(0, msz[0]*msz[1]))
            self._set_labels(cents, ax, codebook)

        plt.imshow(codebook.reshape(msz[0], msz[1])[::], alpha=.5)
        plt.show()

        return cents
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
# from .view import MatplotView
from matplotlib import pyplot as plt
from pylab import imshow, contour
from math import sqrt
import numpy as np
import scipy


class UMatrixView(MatplotView):

    def build_u_matrix(self, som, distance=1, row_normalized=False):
        UD2 = som.calculate_map_dist()
        Umatrix = np.zeros((som.codebook.nnodes, 1))
        codebook = som.codebook.matrix
        if row_normalized:
            vector = som._normalizer.normalize_by(codebook.T, codebook.T,
                                                  method='var').T
        else:
            vector = codebook

        for i in range(som.codebook.nnodes):
            codebook_i = vector[i][np.newaxis, :]
            neighborbor_ind = UD2[i][0:] <= distance
            neighborbor_codebooks = vector[neighborbor_ind]
            Umatrix[i] = scipy.spatial.distance_matrix(
                codebook_i, neighborbor_codebooks).mean()

        return Umatrix.reshape(som.codebook.mapsize)

    def show(self, som, distance2=1, row_normalized=False, show_data=True,
             contooor=True, blob=False, labels=False):
        umat = self.build_u_matrix(som, distance=distance2,
                                   row_normalized=row_normalized)
        msz = som.codebook.mapsize
        proj = som.project_data(som.data_raw)
        coord = som.bmu_ind_to_xy(proj)

        self._fig, ax = plt.subplots(1, 1)
        imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)

        if contooor:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            contour(umat, np.linspace(mn, mx, 15), linewidths=0.7,
                    cmap=plt.cm.get_cmap('Blues'))

        if show_data:
            plt.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray',
                        marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
            plt.axis('off')

        if labels:
            if labels is True:
                labels = som.build_data_labels()
            for label, x, y in zip(labels, coord[:, 1], coord[:, 0]):
                plt.annotate(str(label), xy=(x, y),
                             horizontalalignment='center',
                             verticalalignment='center')

        ratio = float(msz[0])/(msz[0]+msz[1])
        self._fig.set_size_inches((1-ratio)*15, ratio*15)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.00, wspace=.000)
        sel_points = list()

        if blob:
            from skimage.color import rgb2gray
            from skimage.feature import blob_log

            image = 1 / umat
            rgb2gray(image)

            # 'Laplacian of Gaussian'
            blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
            blobs[:, 2] = blobs[:, 2] * sqrt(2)
            imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)
            sel_points = list()

            for blob in blobs:
                row, col, r = blob
                c = plt.Circle((col, row), r, color='red', linewidth=2,
                               fill=False)
                ax.add_patch(c)
                dist = scipy.spatial.distance_matrix(
                    coord[:, :2], np.array([row, col])[np.newaxis, :])
                sel_point = dist <= r
                plt.plot(coord[:, 1][sel_point[:, 0]],
                         coord[:, 0][sel_point[:, 0]], '.r')
                sel_points.append(sel_point[:, 0])

        plt.show()
        return sel_points, umat
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
from collections import Counter

import matplotlib
import numpy as np
from matplotlib import pyplot as plt


class BmuHitsView(MapView):
    def _set_labels(self, cents, ax, labels, onlyzeros, fontsize):
        for i, txt in enumerate(labels):
            if onlyzeros == True:
                if txt > 0:
                    txt = ""
            ax.annotate(txt, (cents[i, 1] + 0.5, cents[-(i+1), 0] + 0.5), va="center", ha="center", size=fontsize)

    def show(self, som, anotate=True, onlyzeros=False, labelsize=7, cmap="jet", logaritmic = False):
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = self._calculate_figure_params(som, 1, 1)

        self.prepare()
        ax = plt.gca()
        counts = Counter(som._bmu[0])
        counts = [counts.get(x, 0) for x in range(som.codebook.mapsize[0] * som.codebook.mapsize[1])]
        mp = np.array(counts).reshape(som.codebook.mapsize[0],
                                      som.codebook.mapsize[1])

        if not logaritmic:
            norm = matplotlib.colors.Normalize(
                vmin=0,
                vmax=np.max(mp.flatten()),
                clip=True)
        else:
            norm = matplotlib.colors.LogNorm(
                vmin=1,
                vmax=np.max(mp.flatten()))

        msz = som.codebook.mapsize

        cents = som.bmu_ind_to_xy(np.arange(0, msz[0] * msz[1]))

        if anotate:
            self._set_labels(cents, ax, counts, onlyzeros, labelsize)


        pl = plt.pcolor(mp[::-1], norm=norm, cmap=cmap)

        plt.axis([0, som.codebook.mapsize[1], 0, som.codebook.mapsize[0]])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.colorbar(pl)

        plt.show()
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
from random import randint

class KMeanspp():
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

    def inicia_centroides(self, k_centroids):
        """."""
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        """."""
        centroids_redimensionado = self.centroids[:, np.newaxis, :]
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        centroid_mais_proximo = np.argmin(distancias, axis=0)
        return centroid_mais_proximo

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

    def roda_kmeans(self, k_centroids):
        """."""
        self.inicia_centroides(1)
        self.inicia_kmeanspp(k_centroids-1)

        MediaDistAnterior = 0.0
        nIteracoes = 0
        while(abs(MediaDistAnterior - self.MediaDistAtual) > self.erro):
            nIteracoes += 1
            print("quantidade de iterações igual à " + str(nIteracoes))
            if(self.lista_centroid_mais_proximos is None):
                self.labels = self.busca_centroides_mais_proximo()
                self.centroids = self.movimenta_centroides(self.labels)
            else:
                self.centroids = self.movimenta_centroides(self.lista_centroid_mais_proximos)
            MediaDistAnterior = self.MediaDistAtual
            self.MediaDistAtual = self.calculaMediaDistancias(self.lista_centroid_mais_proximos)

    def movimenta_centroides(self, closest):
        """."""
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])

    def calculaMediaDistancias(self, centroid_mais_proximo):
        """."""
        centroids_redimensionado = self.centroids[:, np.newaxis, :]
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        centroid_mais_proximo = np.argmin(distancias, axis=0)

        listaDistancias = [0.0]*len(self.centroids)
        indexlista = 0
        for centroid in centroid_mais_proximo:
            listaDistancias[centroid] += distancias[centroid][indexlista]
            indexlista += 1
        for indice in range(0, len(listaDistancias)):
            listaDistancias[indice] = listaDistancias[indice]/sum(centroid_mais_proximo == indice)
        return sum(listaDistancias)
import numpy as np
import sys
import inspect


class NormalizatorFactory(object):

    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % type_name)


class Normalizator(object):

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


class VarianceNormalizator(Normalizator):

    name = 'var'

    def _mean_and_standard_dev(self, data):
        print('mean-std dev')
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        print('normalize')
        me, st = self._mean_and_standard_dev(data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def normalize_by(self, raw_data, data):
        print('normalize-by')
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def denormalize_by(self, data_by, n_vect):
        print('denormalize-by')
        me, st = self._mean_and_standard_dev(data_by)
        return n_vect * st + me
import numpy as np

from sklearn.decomposition import PCA
# from sklearn.decomposition import RandomizedPCA# (randomizedpca is deprecated)
# from .decorators import timeit


class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass


class Codebook(object):

    def __init__(self, mapsize, lattice='rect'):
        self.lattice = lattice

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('input was considered as the numbers of nodes')
            print('map size is [{dlen},{dlen}]'.format(dlen=int(mapsize[0]/2)))
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = mapsize[0]*mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

    @timeit()
    def random_initialization(self, data):
        """
        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True

    @timeit()
    def pca_linear_initialization(self, data):
        """
        We initialize the map, just by using the first two first eigen vals and
        eigenvectors
        Further, we create a linear combination of them in the new map by
        giving values from -1 to 1 in each

        X = UsigmaWT
        XTX = Wsigma^2WT
        T = XW = Usigma

        // Transformed by W EigenVector, can be calculated by multiplication
        // PC matrix by eigenval too
        // Further, we can get lower ranks by using just few of the eigen
        // vevtors

        T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected
        eigenvectors

        (*) Note that 'X' is the covariance matrix of original data

        :param data: data to use for the initialization
        :returns: initialized matrix with same dimension as input data
        """
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        # Randomized PCA is scalable
        #pca = RandomizedPCA(n_components=pca_components) # RandomizedPCA is deprecated.
        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)
        self.initialized = True

    def grid_dist(self, node_ind):
        """
        Calculates grid distance based on the lattice type.

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        if self.lattice == 'rect':
            return self._rect_dist(node_ind)

        elif self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        raise NotImplementedError()

    def _rect_dist(self, node_ind):
        """
        Calculates the distance of the specified node to the other nodes in the
        matrix, generating a distance matrix

        Ej. The distance matrix for the node_ind=5, that corresponds to
        the_coord (1,1)
           array([[2, 1, 2, 5],
                  [1, 0, 1, 4],
                  [2, 1, 2, 5],
                  [5, 4, 5, 8]])

        :param node_ind: number between 0 and number of nodes-1. Depending on
                         the map size, starting from top left
        :returns: matrix representing the distance matrix
        """
        rows = self.mapsize[0]
        cols = self.mapsize[1]
        dist = None

        # bmu should be an integer between 0 to no_nodes
        if 0 <= node_ind <= (rows*cols):
            node_col = int(node_ind % cols)
            node_row = int(node_ind / cols)
        else:
            raise InvalidNodeIndexError(
                "Node index '%s' is invalid" % node_ind)

        if rows > 0 and cols > 0:
            r = np.arange(0, rows, 1)[:, np.newaxis]
            c = np.arange(0, cols, 1)
            dist2 = (r-node_row)**2 + (c-node_col)**2

            dist = dist2.ravel()
        else:
            raise InvalidMapsizeError(
                "One or both of the map dimensions are invalid. "
                "Cols '%s', Rows '%s'".format(cols=cols, rows=rows))

        return dist
# from .view import MatplotView
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import numpy as np


class Hist2d(MatplotView):

    def _fill_hist(self, x, y, mapsize, data_coords, what='train'):
        x = np.arange(.5, mapsize[1]+.5, 1)
        y = np.arange(.5, mapsize[0]+.5, 1)
        X, Y = np.meshgrid(x, y)

        if what == 'train':
            a = plt.hist2d(x, y, bins=(mapsize[1], mapsize[0]), alpha=.0,
                           cmap=cm.jet)
            area = a[0].T * 12
            plt.scatter(data_coords[:, 1], mapsize[0] - .5 - data_coords[:, 0],
                        s=area.flatten(), alpha=.9, c='None', marker='o',
                        cmap='jet', linewidths=3, edgecolor='r')

        else:
            a = plt.hist2d(x, y, bins=(mapsize[1], mapsize[0]), alpha=.0,
                           cmap=cm.jet, norm=LogNorm())
            area = a[0].T*50
            plt.scatter(data_coords[:, 1] + .5,
                        mapsize[0] - .5 - data_coords[:, 0],
                        s=area, alpha=0.9, c='None', marker='o', cmap='jet',
                        linewidths=3, edgecolor='r')
            plt.scatter(data_coords[:, 1]+.5, mapsize[0]-.5-data_coords[:, 0],
                        s=area, alpha=0.2, c='b', marker='o', cmap='jet',
                        linewidths=3, edgecolor='r')

        plt.xlim(0, mapsize[1])
        plt.ylim(0, mapsize[0])

    def show(self, som, data=None):
        # First Step: show the hitmap of all the training data
        coord = som.bmu_ind_to_xy(som.project_data(som.data_raw))

        self.prepare()

        ax = self._fig.add_subplot(111)
        ax.xaxis.set_ticks([i for i in range(0, som.codebook.mapsize[1])])
        ax.yaxis.set_ticks([i for i in range(0, som.codebook.mapsize[0])])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(True, linestyle='-', linewidth=.5)

        self._fill_hist(coord[:, 1], coord[:, 0], som.codebook.mapsize,
                        som.bmu_ind_to_xy(np.arange(som.codebook.nnodes)))

        if data:
            coord_d = som.bmu_ind_to_xy(som.project_data(data))
            self._fill_hist(coord[:, 1], coord[:, 0], som.codebook.mapsize,
                            coord_d, 'data')

        plt.show()
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
		print('----- Iniciando Processamento K-means -----')
		kmeans = KMeans(dados)
		kmeans.roda_kmeans(3)
		
		# ---------------------
		# SOM
		# print('----- Iniciando Processamento SOM -----')

		# mapsize = [25,25]
		# som = SOMFactory.build(dados, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='random', neighborhood='gaussian', training='batch')
		# som.train(n_job=3, verbose='info')

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
