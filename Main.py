
# from . import dotmap, histogram, hitmap, mapview, umatrix
# import os
# import string
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from math import inf as positive_infinite
# from scipy.spatial import distance
# from sklearn.cluster import KMeans as KMeansDefault
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.preprocessing import StandardScaler
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
import pylab as pl
from KMeans.KMeans import KMeans
from Helpers.Matrix import TransformMatrix
from Helpers.process_zoo import ProcessZoo
from Som.Som import SOM
from KMeans.KMeansPlotter import KMeansPlotter
from Indices.Silhouette import Silhouette
from Indices.SilhouettePlotter import SilhouettePlotter

def executar( opcao , dados , animal_names):
    if(opcao == 'T' or opcao == 'K'):
        print('----- Iniciando Processamento K-means -----')
        kmeans = KMeans(dados)
        kmeans.roda_kmeans(5 ,1000, 0.0001)
        plotter = KMeansPlotter()
        plotter.plots(kmeans , animal_names)

    if(opcao == 'T' or opcao == 'S' ):
        # SOM
        print('----- Iniciando Processamento SOM -----')
        som = SOM()
        som.executar(dados)
    if(opcao == 'T' or opcao == 'XM'):
        print ('código do XMeans ainda não implementado')
    if(opcao == 'T' or opcao == 'K++'):
        print('----- Iniciando Processamento K-means++  -----')
        kmeansPP = Kmeans(dados,'kmeans++')
    if(opcao == 'T' or opcao == 'SL'):
        dictionary_nroClusters_Silhouette = {}
        for nroDeClusters in range(2 , 11):
            kmeans = KMeans(dados)
            print(nroDeClusters)
            kmeans.roda_kmeans(nroDeClusters,1000, 0.0001)
            silhouette = Silhouette()
            valorSilhouette = silhouette.allGroupsSilhouette(kmeans.points,kmeans.lista_centroid_mais_proximos)
            dictionary_nroClusters_Silhouette[nroDeClusters] = valorSilhouette
        plotter = SilhouettePlotter()
        plotter.plots(dictionary_nroClusters_Silhouette)
def ExecutionOptions():
    entradaIncorreta = True
    entradasValidas = ['K','K++','XM','S', 'SL']
    while entradaIncorreta:
        print('Deseja executar todos os algoritmos?(S/N)')
        entrada = input().upper()
        if( entrada == 'N'):
            while entradaIncorreta:
                print('Deseja executar qual Algoritmo (KMeans , KMeans++ , XMeans , SOM , Silhouette )? Escreva: \n K para KMeans \n K++ para Kmeans++ \n XM para XMeans \n S para SOM \n SL para Silhouette ')
                opcaoEscolhida = input().upper()
                if opcaoEscolhida not in entradasValidas:
                    entradaIncorreta = True
                else:
                    entradaIncorreta = False
        elif(entrada == 'S'):
            opcaoEscolhida = 'T'
            entradaIncorreta = False
        else:
            entradaIncorreta = True

    return opcaoEscolhida

if __name__ == "__main__":

    opcaoExec = ExecutionOptions()

    pre_process = ProcessZoo()
    dados = pre_process.get_original_matrix()
    animal_names = pre_process.get_animals_names()

    executar(opcaoExec , dados , animal_names)
