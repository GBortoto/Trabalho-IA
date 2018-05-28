"""Default retirado de: http://stackabuse.com/k-means-clustering-with-scikit-learn/ ."""

# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans


class KMeansDefault():
    """."""

    def __init__(self, points):
        """."""
        print('----- Rodando KMeans -----')
        plt.scatter(points[:, 0], points[:, 1], label='True Position')
        plt.show()
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(points)
        print(kmeans.cluster_centers_)
        print(kmeans.labels_)
        plt.scatter(points[:, 0], points[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.show()
        plt.scatter(points[:, 0], points[:, 1], c=kmeans.labels_, cmap='rainbow')
        plt.show()
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
        plt.show()
