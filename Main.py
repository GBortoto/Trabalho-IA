from Preprocessor import Preprocessor
if __name__ == "__main__":
	preprocessor = Preprocessor()
	texts = preprocessor.readAllTextsFromDatabase()
	#texts contêm todos os textos que serão utilizados de forma que cada index do array tem uma notícia. As notícias não estão tratadas , são o texto puro , retirado apenas os e-mails e em ordem aleatória.
