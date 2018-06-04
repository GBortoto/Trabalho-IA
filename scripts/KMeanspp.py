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
