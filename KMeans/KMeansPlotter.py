# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
# https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html

import matplotlib.pyplot as plt
import pylab as pl
import os
import numpy as np
from sklearn.decomposition import PCA
import time

class KMeansPlotter():

    def __init__(self ):
    #limitado a 24 markers
        self.mainDirectory = "plots"
        self.markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_"]
        self.color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        self.dirName = ''
        np.random.shuffle(self.markers)

    def salvaClusters(self, animal_names ,  clusters):
        dictionary_animal_cluster = {}
        for point in range(0 , len(clusters)):
            if clusters[point] in dictionary_animal_cluster.keys():
                lista = dictionary_animal_cluster.get(clusters[point])
                lista.append(animal_names[point])
                dictionary_animal_cluster[clusters[point]] = lista
            else:
                novaLista = []
                novaLista.append(animal_names[point])
                dictionary_animal_cluster[clusters[point]] = novaLista

        f= open(self.mainDirectory+'/'+self.dirName+'/Animal_cluster.txt',"w+")

        cluster_number = 1
        for key in sorted(dictionary_animal_cluster.keys()):
            f.write(str(cluster_number) + "º cluster: \n")
            cluster_number +=1
            for animal in dictionary_animal_cluster.get(key):
                f.write(animal + '\n')
            f.write('\n')
        f.close()
        return

    def makedir(self):

        if not os.path.exists(self.mainDirectory):
            os.makedirs(self.mainDirectory)

        dirName = 'KMeansPlot'
        number = 0
        directory = dirName
        while os.path.exists(self.mainDirectory+'/'+ directory):
            number += 1
            directory = dirName+str(number)
        os.makedirs(self.mainDirectory+'/'+directory)

        return directory

    def plots(self, kmeans,animal_names , save=True):
        self.dirName = self.makedir()
        """."""
        plt = self.createPlotPoints(kmeans , animal_names)

        if save is False:
            plt.show()
        else:
            print('Salvando resultados...')
            plt.savefig(self.mainDirectory+'/'+self.dirName+'/result_KMeans' + str(time.time()) + '.png')
        plt.clf()


    def createPlotPoints(self, kmeans , animal_names):
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

        self.salvaClusters(animal_names , clusters)

        clusters.extend(centroid_labels)

        pca = PCA(n_components=2).fit(all_points)
        dados2d = pca.transform(all_points)
        print(pca.explained_variance_ratio_.cumsum())

        for point in range(0 ,dados2d.shape[0]):
            centroid = clusters[point]
            if(centroid >= 0):
                pl.scatter(dados2d[point,0],dados2d[point,1], s = areaPoints, c= self.color[centroid%8],marker=self.markers[centroid%24])
            #centroids
            elif(centroid < 0):
                pl.scatter(dados2d[point,0],dados2d[point,1] , s = areaCentroid , c ='k' , marker = 'o')


        return pl
