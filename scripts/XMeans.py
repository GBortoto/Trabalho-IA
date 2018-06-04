
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