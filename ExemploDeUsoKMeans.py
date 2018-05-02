from ListOfPoints import ListOfPoints
from KMeans import KMeans
from random import random

if __name__ == '__main__':
    
    n = 10000               # Número de pontos
    dimentions = 10         # Número de dimenções
    NGroups = 5             # Número de grupos 
    iterations = 200        # Número de iterações a serem executadas

    data = ListOfPoints(n, dimentions)
    data.points = [[random()*1000 for j in range(dimentions)] for i in range(n)]
    kmeans = KMeans(data, NGroups)

    print('Inicializando execução')
    print(str(n) + ' elementos de dados')
    print(str(dimentions) + ' dimenções')
    print(str(NGroups) + ' grupos')
    print(str(iterations) + ' iterações')
    print('')
    print('Iteração\tNúmero de elementos em cada grupo')
    
    for i in range(iterations):
        print(str(i+1) + '/' + str(iterations) + '\t\t', end='')
        kmeans.run()
        
    input()
