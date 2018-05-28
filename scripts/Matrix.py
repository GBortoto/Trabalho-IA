# -*- coding: utf-8 -*-

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# import numpy as np

"""
# Dada uma matriz pura, essa classe é responsável por retornar qualquer tipo de matrix customizável
# corpus[texto0[palavraA, palavraB, palavraC], texto1[palavraX, palavraY, palavraZ]]
# Matrix pura: corpus[n_textos][m_palavras_fixas]
# a = [["A","B","C"],["X","Y","Z"],["L","M","N"]]
# x = [["John likes to watch movies. Mary likes movies too.",
"John also likes to watch football games."],["X","Y","Z"],["L","M","N"]]
# y = ["¿Plata o plomo?  La cárcel em Estados Unidos es peor que la muerte. Bueno ... en mi opinión, una cárcel en Estados Unidos es peor que la muerte. ", "Ma tem ou não tem o celular do milhãouamm? Ma! Ao adquirir o carnê do Baú, você estará concorrendo a um prêmio de cem mil reaisam. Ma quem quer dinheiroam? Ma não existem mulher feiam, existem mulher que não conhece os produtos Jequitiamm. Qual é a musicamm? Vem pra lá, mah você vai pra cá. Agora vai, agora vem pra láamm. Patríciaaammmm... Luiz Ricardouaaammmmmm. Ma vejam só, vejam só. Ma! Ao adquirir o carnê do Baú, você estará concorrendo a um prêmio de cem mil reaisam. Ma não existem mulher feiam, existem mulher que não conhece os produtos Jequitiamm. Estamos em ritmo de festamm."]

Fonte:
http://scikit-learn.org/stable/modules/feature_extraction.html

"""

class TransformMatrix():
	"""."""
	def __init__(self, matrix):
		# Guarda matrix de lista de palavras por texto
		self.matrix = matrix

		# Cria matrix
		self._matrix_creation()

	def _matrix_creation(self):
		# Iremos criar uma "vetorizacao" baseado em frequencia (count)
		# vectorizer = CountVectorizer(max_df=0.9, min_df=0.05)
		vectorizer = CountVectorizer()

		#Retorna array TF de cada palavra
		self.bag_of_words = (vectorizer.fit_transform(self.matrix)).toarray()

		# Retorna array com as palavras (labels)
		self.feature_names = vectorizer.get_feature_names()

		# del self.matrix

	def get_matrix(self, type='tf-n'):
		# Matrix binaria será sempre a matrix TF para os casos em que a frequencia é diferente de 0
		# Método sign identifica se numero != 0
		# print(type)
		# print(type == 'tf-n')
		if type is 'binary':
			print('----- Processando Matriz Binaria -----')
			return (sp.sign(self.bag_of_words))
		# Matrix TF somente com frequencia da palavra, independente da frequencia relativa do corpus
		if type == 'tf':
			print('----- Processando Matriz TF -----')
			return self.bag_of_words
		# Matrix TF normalizada com frequencia indo de [0, 1)
		if type == 'tf-n':
			print('----- Processando Matriz TF-Normalizada -----')
			listas = [np.sum(lista, axis=0) for lista in self.bag_of_words]
			result = sum(listas)
			return self.bag_of_words / result
		# Matrix TF_IDF que utiliza inverse document
		if type == 'tf-idf':
			print('----- Processando Matriz TF-IDF -----')
			tfidf_vectorize = TfidfTransformer(smooth_idf=False)
			return tfidf_vectorize.fit_transform(self.bag_of_words).toarray()
