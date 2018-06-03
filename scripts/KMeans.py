
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
        return sum(listaDistancias)
