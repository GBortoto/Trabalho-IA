# -*- coding: utf-8 -*-
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

from sklearn.feature_extraction.text import CountVectorizer
import scipy as sp
import numpy as np
import processador

class TransformMatrix():
	def __init__(self, matrix):
		self.matrix = matrix
		self._matrix_creation()

	def _matrix_creation(self):
		vectorizer = CountVectorizer()

		#Retorna matrix com frequencia que a palavra aparece
		self.bag_of_words = (vectorizer.fit_transform(self.matrix)).toarray()
		self.feature_names = vectorizer.get_feature_names()

		#Transforma em array TF
		print(self.bag_of_words)
		#Retorna as ordem com que as palavras aparecem
		print(self.feature_names)

	def matrix_binaria(self):
		return (sp.sign(self.bag_of_words))

	# Pending
	def matrix_tf(self):
		return self.bag_of_words

	# Pending
	def matrix_tf_normalizada(self):
		tempResultado = (self.bag_of_words).sum(axis=0)
		print(tempResultado)
		print(tempResultado.sum(axis=0))
		print("Número de palavras total: " + (self.feature_names).length)

	# Pending
	def matrix_tfidf(self):
		pass
