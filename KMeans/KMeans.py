from numpy import dot
from numpy.linalg import norm
import numpy as np
import math

class KMeans():
    """."""

    def __init__(self, points, type_of_kmeans='default', distance_type='euclidian'):
        """Generate a KMeans model for a specific 'k' and a n-matrix of point.
        It will return a model which represents the k-means cluster function
        """
        self.type_of_kmeans = type_of_kmeans
        self.distance_type = distance_type
        self.points = points
        self.labels = []
	## uma lista contendo os centroids mais proximos de cada ponto
        self.lista_centroid_mais_proximos = None

    def inicia_centroides(self, k_centroids):
        """."""
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        """."""
        centroids_redimensionado = self.centroids[:, np.newaxis , :]
        if self.distance_type == 'euclidian':
            #eleva-se a diferença ao quadrado
            diffCordenadasAoQuadrado = (self.points - centroids_redimensionado) ** 2
            #soma as diferenças e faz a raiz delas, obtendo as distancias euclidianas de todos os pontos para todos os centroids
            distancias = np.sqrt(diffCordenadasAoQuadrado.sum(axis=2))
            #identifica o centroid mais próximo de cada ponto
            centroid_mais_proximo = np.argmin(distancias, axis=0)
            print('Shape distancia euclidiana')
            print(distancias.shape)


        if self.distance_type == 'cosine_similarity':
            cos_sim = dot(self.points, centroids_redimensionado)/(norm(self.points)*norm(centroids_redimensionado))
            centroid_mais_proximo = np.argmin(1-cos_sim, axis=0)
            print('Shape distancia cosseno')
            print(cos_sim.shape)

        return centroid_mais_proximo

    def roda_kmeans(self, k_centroids, n_iteracoes_limite = 1000, erro = 0.001, centroid_aleatorio = None):
        """."""
        if centroid_aleatorio is None:
            if self.type_of_kmeans == 'kmeans++':
                self.inicia_centroides(1)
                self.inicia_kmeanspp(k_centroids-1)
            else:
                self.inicia_centroides(k_centroids)
        else:
            self.centroids = centroid_aleatorio


        MediaDistAnterior = 0.0
        MediaDistAtual = float("inf") #positive_infinite

        nIteracoes = 0
        while((nIteracoes < n_iteracoes_limite) and abs(MediaDistAnterior - MediaDistAtual) > erro):
            # Só executa se a lista de centroids não tiver sido determinada na ultima iteração
            nIteracoes += 1
            print("quantidade de iterações igual à " + str(nIteracoes))
            print(str(abs(MediaDistAnterior - MediaDistAtual)))
            print("valor da distancia da media anterior: " + str(MediaDistAnterior))
            print("valor da distancia da media atual: " + str(MediaDistAtual))
            if(self.lista_centroid_mais_proximos is None):
                self.lista_centroid_mais_proximos = self.busca_centroides_mais_proximo()
            #movimenta os centroids  a partir da lista adquirida na ultima iteração
            self.centroids = self.movimenta_centroides(self.lista_centroid_mais_proximos)
            MediaDistAnterior = MediaDistAtual
            #atualiza lista de centroids mais proximos e calcula a média da distancia entre os pontos e
            #os centroids mais proximos
            MediaDistAtual = self.calculaMediaDistancias()

            print("valor da distancia anterior depois da atualização  : " + str(MediaDistAnterior))
            print("valor da distancia atual depois da atualização: " + str(MediaDistAtual))

    def movimenta_centroides(self, closest):
        """."""
        listaDeCentroids = []
        for centroid in closest:
            if not centroid in listaDeCentroids:
                listaDeCentroids.append(centroid)

        centroids = np.array([self.points[closest == k].mean(axis=0) for k in listaDeCentroids])

        listaResultado = [item for item in range(self.centroids.shape[0]) if item not in listaDeCentroids]

        for valor in listaResultado:
            centroids = np.insert(centroids , valor , self.centroids[valor] , axis=0 )

        return centroids

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
