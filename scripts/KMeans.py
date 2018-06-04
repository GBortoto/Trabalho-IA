
class KMeans():
    """."""

    def __init__(self, points, type_of_kmeans='default'):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.
        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
        self.points = points
<<<<<<< HEAD
	## media da distancia entre os centroids e os pontos
        self.MediaDistAtual = 100000000000000000000.0
	#diferença maxima entre a distancia média entre duas iterações
        self.erro = 0.05
||||||| merged common ancestors

=======
>>>>>>> 87965afeaed30036cd8d170b8491f5608fcee9ae
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
<<<<<<< HEAD
        # É adicionado uma nova dimensão aos centroids de forma que seja possivel
        # calcular as diferenças entre as coordenadas para todos os centroids de uma vez
        # Ex:
        # Antes de ser redimensionado  :  centroids                = [[7,8,7][8,12,5][13,3,2]]
        # Depois de ser redimensionado :  centroids_redimensionado = [[[7,8,7][8,12,5][13,3,2]]]
        # dessa forma podemos obter as diferenças entre as cordenadas em uma unica operação:
        # supondo p1 seja um ponto : [3,1,2]
        # temos: centroids_redimensionado - p1 = [
        #                     [ 4, 7, 5], -> diferença das cordenadas do ponto 1 para o centroid 1
        #                     [ 5,11, 3], -> diferença das cordenadas do ponto 1 para o centroid 2
        #                     [10, 2, 0]  -> diferença das cordenadas do ponto 1 para o centroid 3
        #                   ]
        #
||||||| merged common ancestors
    
=======
>>>>>>> 87965afeaed30036cd8d170b8491f5608fcee9ae
        centroids_redimensionado = self.centroids[:, np.newaxis , :]
        #eleva-se a diferença ao quadrado
        diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
        #soma as diferenças e faz a raiz delas, obtendo as distancias euclidianas de todos os pontos para todos os centroids
        distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
        #identifica o centroid mais próximo de cada ponto
        centroid_mais_proximo = np.argmin(distancias, axis=0)

        return centroid_mais_proximo

    def roda_kmeans(self, k_centroids, n_iteracoes = 1000, erro = 0.1, centroid_aleatorio = None):
        """."""
        if centroid_aleatorio is None:
            self.inicia_centroides(k_centroids)
        else:
            self.centroids = centroid_aleatorio

        MediaDistAnterior = 0.0
        MediaDistAtual = positive_infinite

        nIteracoes = 0
        while((nIteracoes < n_iteracoes) and abs(MediaDistAnterior - MediaDistAtual) > erro):
            # Só executa se a lista de centroids não tiver sido determinada na ultima iteração
            nIteracoes += 1
            print("quantidade de iterações igual à " + str(nIteracoes))
            
            if(self.lista_centroid_mais_proximos is None):
<<<<<<< HEAD
                self.lista_centroid_mais_proximos = self.busca_centroides_mais_proximo()
                self.centroids = self.movimenta_centroides(self.lista_centroid_mais_proximos)
            else:
                #movimenta os centroids  a partir da lista adquirida na ultima iteração
                self.centroids = self.movimenta_centroides(self.lista_centroid_mais_proximos)
            MediaDistAnterior = self.MediaDistAtual
            #atualiza lista de centroids mais proximos e calcula a média da distancia entre os pontos e
            #os centroids mais proximos
            self.MediaDistAtual = self.calculaMediaDistancias()
            self.plotter.plots(self)
||||||| merged common ancestors
 
>>>>>>> 87965afeaed30036cd8d170b8491f5608fcee9ae

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
<<<<<<< HEAD
        for indice in range(0 , len(listaDistancias)):
            listaDistancias[indice] = listaDistancias[indice]/sum(self.lista_centroid_mais_proximos == indice)
||||||| merged common ancestors

=======
>>>>>>> 87965afeaed30036cd8d170b8491f5608fcee9ae
        return sum(listaDistancias)

