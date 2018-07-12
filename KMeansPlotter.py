# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

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
