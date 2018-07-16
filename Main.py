
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
from sklearn.decomposition import PCA
import pylab as pl
from KMeans.KMeans import KMeans
from Helpers.Matrix import TransformMatrix
from Helpers.process_zoo import ProcessZoo
from Som.Som import SOM

def executar( opcao , dados ):
    if(opcao == 'T' or opcao == 'K'):
        print('----- Iniciando Processamento K-means -----')
        kmeans = KMeans(dados)
        kmeans.roda_kmeans(3)
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


def ExecutionOptions():
    entradaIncorreta = True
    entradasValidas = ['K','K++','XM','S']
    while entradaIncorreta:
        print('Deseja executar todos os algoritmos?(S/N)')
        entrada = input().upper()
        if( entrada == 'N'):
            while entradaIncorreta:
                print('Deseja executar qual Algoritmo (KMeans , KMeans++ , XMeans , SOM )? Escreva: \n K para KMeans \n K++ para Kmeans++ \n XM para XMeans \n S para SOM ')
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

    executar(opcaoExec , dados)
