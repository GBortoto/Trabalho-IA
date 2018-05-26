"""Class to process all texts."""

import os
import string
import Matrix as mxt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, LancasterStemmer


class ProcessTexts():
    """."""

    def __init__(self):
        """."""
        self._texts = []  # list of text samples
        for directory in sorted(os.listdir('./database/bbc_news')):
            for file in sorted(os.listdir('./database/bbc_news/' + directory)):
                path = './database/bbc_news/' + directory + "/" + file
                f = open(path, encoding='latin-1')
                t = f.read()
                self._texts.append(t)
                f.close()
        print(self._texts[0])
        print("----- Tokenizando Textos -----")
        self._processa_text()

    def _processa_text(self):
        table = str.maketrans('', '', string.punctuation)
        stop_words = set(stopwords.words('english'))
        self.tokens = []
        print("----- Tokenizando Sentencas e Palavras -----")

        # Para cada texto
        for index, text in enumerate(self._texts):
            # Tokenize por sentenca
            sentences = sent_tokenize(text)
            tokens_of_sentence = []
            # Para cada sentenca
            for sentence in sentences:
                # Tokenize por palavras, elimine stop words, pontuação e de lower
                stripped = [word.translate(table).lower() for word in word_tokenize(sentence) if not word in stop_words]
                stemmerized = self._stemmer_text(tokens=stripped)
                tokens_of_sentence = tokens_of_sentence + stemmerized
            self.tokens.append(tokens_of_sentence)
        del self._texts

    def _stemmer_text(self, tokens, type='Porter'):
        if type is 'Porter':
            porter = PorterStemmer()
            return [porter.stem(t) for t in tokens]
        if type is 'Lancaster':
            lancaster = LancasterStemmer()
            return [lancaster.stem(t) for t in tokens]


if __name__ == '__main__':
    """."""

    texto_processor = ProcessTexts()
    matriz = mtx.TransformMatrix(texto_processor.tokens)
