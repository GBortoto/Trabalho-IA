from ListOfPoints import ListOfPoints
from typing import List
import random
import math

class KMeans():
    """Dado um conjunto de dados do tipo ListOfPoints, essa classe executa o algoritmo k-means"""
    def __init__(self, dataPoints: ListOfPoints, KGroups:int):
        """Construtor - Recebe um conjunto de dados e o número de grupos a serem descobertos"""
        # N -> Número de elementos no conjunto de dados

        # Lista de dados
        self.dataPoints = dataPoints

        # Lista de centroides
        self.centroids = ListOfPoints(KGroups, dataPoints.getNumberOfDimentions())

        # Lista de índices do centróide mais proximo de cada ponto
        # N * (int)
        self.nearest = [0 for i in range(len(dataPoints))]

        # Setar posições aleatórias para os centróides
        for i in range(len(self.centroids)):
            self.centroids.points[i] = [random.random()*1000 for i in range(self.centroids.getNumberOfDimentions())]

    def euclideanDistance(self, pointA, pointB):
        """Calcula a distância euclideana entre dois pontos"""
        # pointA e pointB DEVEM apresentar o mesmo número de dimensões
        return math.sqrt(
            sum(
                [(pointA[i] - pointB[i])**2 for i in range(len(pointA))]
                )
            )
    
    def findNearestCentroid(self):
        """Calcula, para cada elemento no conjunto de dados, o centróide mais próximo e o armazena"""
        
        for i in range(len(self.dataPoints)):       # Para cada ponto...

            # Centróide mais próximo do ponto atual
            # [Índice do centróide, distância ao ponto]
            nearestCentroid = [0, math.inf]
            
            for j in range(len(self.centroids)):    # ...e para cada centroide

                # Calcula a distância do centróide atual para o ponto atual
                distance = self.euclideanDistance(self.dataPoints.points[i], self.centroids.points[j])

                # Caso a distância calculada for menor do que a distância mínima já calculada 
                if(distance <= nearestCentroid[1]):
                    # Então este centróide passa a ser o mais próximo
                    nearestCentroid = [j, distance]

            # Após calcular qual o centróide mais próximo, atribua esse valor a lista de pontos "nearest"
            self.nearest[i] = nearestCentroid[0]
        
    def recalculateCentroids(self):
        """Recalcula a posição de cada centróide, atribuindo-a a média das posições de seus pontos"""
        
        for i in range(len(self.centroids)):        # Para cada centróide
            
            # Crie uma lista com os índicies de todos os pontos que foram atribuidos a este centróide
            indexes_pointsInCluster = []
            for j in range(len(self.nearest)):
                if(self.nearest[j] == i):
                    indexes_pointsInCluster.append(j)
            

            # Crie um ListOfPoints de tamanho igual ao tamanho da lista de índices anterior
            # e que tenha o mesmo número de dimenções que os dados
            pointsInCluster = ListOfPoints(len(indexes_pointsInCluster),
                                           self.dataPoints.getNumberOfDimentions())

            # Para cada elemento de dados que foi atribuido a este centróide
            for j in range(len(indexes_pointsInCluster)):
                # Armazene a posição deste elemento em pointsInCluster
                pointsInCluster.points[j] = self.dataPoints.points[indexes_pointsInCluster[j]]

            print(str(len(pointsInCluster)) + '\t', end='')

            # Determine uma nova posição para o centróide
            newPosition = []
            for j in range(pointsInCluster.getNumberOfDimentions()):
                mean = sum(pointsInCluster.getDimention(j)) / len(pointsInCluster)
                newPosition.append(mean)
            self.centroids.points[i] = newPosition

        print('')
        
    def run(self):
        self.findNearestCentroid()
        self.recalculateCentroids()
