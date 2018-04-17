# -*- coding: utf-8 -*-
"""
# Dada uma matriz pura, essa classe é responsável por retornar qualquer tipo de matrix customizável
# corpus[texto0[palavraA, palavraB, palavraC], texto1[palavraX, palavraY, palavraZ]]
# Matrix pura: corpus[n_textos][m_palavras_fixas]
# a = [["A","B","C"],["X","Y","Z"],["L","M","N"]]
"""
import processador

class TransformMatrix():
	def __init__(self, matrix):
		self.matrix = matrix

	def _matrix_normalization(self):
		# HashMap
		# self.words = {self.matrix[0][i]: self.matrix[0][i+1] for i in range(0, len(self.matrix[0]), 2)}
		self.words = { for words in text}
		print(self.words)

	def matrix_binaria(self):
		pass

	def matrix_tf(self):
		pass

	def matrix_tfidf(self):
		pass