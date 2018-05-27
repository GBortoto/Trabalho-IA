# -*- coding: utf-8 -*-
import ProcessTexts as preprocessor
import Matrix as mtx
import KMeans as kmeans
import KMeansDefault as kmeans_default

if __name__ == "__main__":
	# preprocessor = Preprocessor()
	# texts = preprocessor.readAllTextsFromDatabase()
	# #texts contêm todos os textos que serão utilizados de forma que cada index do array tem uma notícia. As notícias não estão tratadas , são o texto puro , retirado apenas os e-mails e em ordem aleatória.
	# texts = preprocessor.processTexts(texts)
	# textos = []
	# for txt in texts:
	# 	textos.append(' '.join(txt))
	# texts = [item for sublist in textos for item in sublist]
	#
	#
	# # Devemos remover o vetor de vetores e somente deixar um vetor com várias palavras por indices
	# transformador = TransformMatrix(texts)
	# mtx_binaria = transformador.matrix_binaria()
	# print(mtx_binaria)

	preprocessor = preprocessor.ProcessTexts()
	print('----- Transformando Tokens em Matriz -----')
	matrix = mtx.TransformMatrix(preprocessor.tokens)
	print('----- Resultados do bag of words -----')

	# kmeans = kmeans_default.KMeansDefault(matrix.get_matrix(type='tf-n'))
	# kmeans = kmeans.KMeans(matrix.get_matrix(type='tf-n'))
	# kmeans.plots()
	# kmeans.roda_kmeans(5)
	# kmeans.plots(type='centroids')


	"""
		[X] - Ler todos os textos
		[X] - Fazer data clean dos dados
		[] - Roda Bag of Words para transformar lista de textos em vetor bidimensional de frequencia de palavra por texto
		[] - Criar 3 outputs do Bag of Words
			[] - Matrix binaria
			[] - Matrix tf
			[] - Matrix tf_idf
		[] - Rodar K-means para cada matrix
		[] - Rodar SOM para cada matrix
		[] - Pos-processamento
	"""

	""" Teste KMeans
	n = 10000
    dimentions = 10
    NGroups = 5
    iterations = 200

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
	"""
