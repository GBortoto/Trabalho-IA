# -*- coding: utf-8 -*-
"""
# Dada uma matriz pura, essa classe é responsável por retornar qualquer tipo de matrix customizável
# corpus[texto0[palavraA, palavraB, palavraC], texto1[palavraX, palavraY, palavraZ]]
# Matrix pura: corpus[n_textos][m_palavras_fixas]
# a = [["A","B","C"],["X","Y","Z"],["L","M","N"]]
# x = [["John likes to watch movies. Mary likes movies too.", 
"John also likes to watch football games."],["X","Y","Z"],["L","M","N"]]
# y = ["¿Plata o plomo?  La cárcel em Estados Unidos es peor que la muerte. Bueno ... en mi opinión, una cárcel en Estados Unidos es peor que la muerte. ", "Ma tem ou não tem o celular do milhãouamm? Ma! Ao adquirir o carnê do Baú, você estará concorrendo a um prêmio de cem mil reaisam. Ma quem quer dinheiroam? Ma não existem mulher feiam, existem mulher que não conhece os produtos Jequitiamm. Qual é a musicamm? Vem pra lá, mah você vai pra cá. Agora vai, agora vem pra láamm. Patríciaaammmm... Luiz Ricardouaaammmmmm. Ma vejam só, vejam só. Ma! Ao adquirir o carnê do Baú, você estará concorrendo a um prêmio de cem mil reaisam. Ma não existem mulher feiam, existem mulher que não conhece os produtos Jequitiamm. Estamos em ritmo de festamm."]

http://scikit-learn.org/stable/modules/feature_extraction.html

"""
from sklearn.feature_extraction.text import CountVectorizer
import processador

class TransformMatrix():
	def __init__(self, matrix, lista_de_listas):
		self.matrix = matrix

	def _matrix_normalization(self):
		vectorizer = CountVectorizer()
		self.bag_of_words = vectorizer.fit_transform(self.matrix)
		print(self.bag_of_words.toarray())
		print(vectorizer.get_feature_names())

	def matrix_binaria(self):
		pass

	def matrix_tf(self):
		pass

	def matrix_tfidf(self):
		pass