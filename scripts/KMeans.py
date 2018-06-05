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
