
from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

class Xmeans():

    def __init__(self, points):
        """."""
        self.points = points
        self.num_centroids = 2
        self.labels = []

    def roda_xmeans(self, n_iteracoes = 100):
        """."""
        self.roda_kmeans(n_iteracoes = n_iteracoes)

        for iter in range(n_iteracoes):

            bool nothing_changed = True
            for i in range(self.num_centroids):
                points_per_centroid = get_centroid_points(i)
                my_labels = [i for j in range(len(points_per_centroid))]

                # BIC dos centróide pai
                bic_father = self.compute_bic(self.centroids[i], my_labels, 1, points_per_centroid)

                # Guarda o centróide pai em outra variável
                father_centroid = self.centroids[i].pop()
                
                # O número representa quanto do range dos pontos será utilizado
                new_centroids = self.get_two_new_centroids(0.9, father_centroid)
                self.roda_kmeans(kmeans_centroids = new_centroids, kmeans_points = points_per_centroid)

                # BIC dos centróides filhos
                bic_children = self.compute_bic()

                if bic_children > bic_pai:
                    nothing_changed = False
                    # Coloca centróides filhos no self.centroids
                
            if nothing_changed == True:
                return


    # roda k_means no escopo passado pelos parâmetros
    # 'start' e 'end' são os índices do centróides que serão utilizados e são, respecitvamente, inclusivo e exclusivo
    def roda_kmeans(self, kwargs**):
        kmeans_centroids = kwargs.get('kmeans_centroids', self.inicia_centroides(2))
        kmeans_points = kwargs.get('kmeans_points', self.points)
        n_iteracoes = kwargs.get('n_iteracoes', 1000)

        # ARRUMAR
        for iteration in range(n_iteracoes):
            self.labels = self.busca_centroides_mais_proximo(kmeans_points)
            self.centroids = self.movimenta_centroides(self.labels)

    def inicia_centroides(self, k_centroids):
        """."""
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        return centroids[:k_centroids]

    def busca_centroides_mais_proximo(self, kmeans_points):
        """."""
        distancias = np.sqrt(((kmeans_points - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distancias, axis=0)

    def roda_kmeans(self, k_centroids, n_iteracoes=1000, centroids):
        """."""
        self.inicia_centroides(k_centroids)
        for iteration in range(n_iteracoes):
            self.labels = self.busca_centroides_mais_proximo()
            self.centroids = self.movimenta_centroides(self.labels)

    def movimenta_centroides(self, closest):
        """."""
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])


    # Divisão de centróides em 2

    def get_centroid_points(self, i):
        points_per_centroid = []

        for k in range(len(self.labels)):
            if (self.labels[k] == i):
                points_per_centroid.append(self.points[k])

        return points_per_centroid

    def get_two_new_centroids(self, trim_percentage, father_centroid):
        one_centroid = []
        two_centroid = []

        for i in range(len(self.points)):
            range_dimension = max(self.points[i]) - min(self.points[i])
            range_dimension = range_dimension * trim_percentage
            range_divided = range_dimension / 2

            one_centroid.append(father_centroid[i] - range_divided)
            two_centroid.append(father_centroid[i] + range_divided)
        
        return [one_centroid, two_centroid]

    # Método para avaliação dos modelos de centróides

    def compute_bic(centers, labels, K, X):
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
        cl_var = (1.0 / (R - K) / d) *
                 sum([sum(distance.cdist(X[np.where(labels == i)],
                     [centers[0][i]], 'euclidean')**2) for i in range(K)])

        const_term = 0.5 * K * np.log(R) * (d + 1)

        BIC = np.sum([
                    n[i] * np.log(n[i]) -
                    n[i] * np.log(R) -
                    ((n[i] * d) / 2) * np.log(2 * np.pi * cl_var) -
                    ((n[i] - 1) * d / 2) for i in range(K)
                    ]) - const_term

        return(BIC)

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
