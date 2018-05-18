class Kmeans():

    def __init__(self, type_of_kmeans, points):
        self.type_of_kmeans = type_of_kmeans
        self.points = points

    def see_points(self):
        #plt.scatter(points[:,0], points[:,1])
        ax = plt.gca()

    def inicia_centroides(self, k_centroids):
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        distancias = np.sqrt(((points - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distancias, axis=0)

    def roda_kmeans(self):
        self.inicia_centroides(4)
        self.movimenta_centroides(self.busca_centroid_mais_proximo())

    def movimenta_centroides(self, closest):
        return np.array([self.points[closest==k].mean(axis=0) for k in range(self.centroids.shape[0])])
