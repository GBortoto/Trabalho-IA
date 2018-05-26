"""Class to process all texts."""

import os
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, LancasterStemmer


class ProcessTexts():
    """."""

    def __init__(self):
        """."""
        self.texts = []  # list of text samples
        for directory in sorted(os.listdir('./database/bbc_news')):
            for file in sorted(os.listdir('./database/bbc_news/' + directory)):
                path = './database/bbc_news/' + directory + "/" + file
                f = open(path, encoding='latin-1')
                t = f.read()
                self.texts.append(t)
                f.close()
        print(self.texts[0])
        print("----- Tokenizando Textos -----")
        self._processa_text()

    def _processa_text(self):
        table = str.maketrans('', '', string.punctuation)
        tokens = []
        print("----- Tokenizando Sentencas -----")
        for text in self.texts:
            sentences = sent_tokenize(text)
            tokens_of_sentence = []
            for sentence in sentences:
                stripped = [word.translate(table).lower() for word in word_tokenize(sentence)]

            # stopWords = set(stopwords.words('english'))
            # wordsFiltered = []
            # stop_words = set(stopwords.words('english'))
            # words = [w for w in words if not w in stop_words]

            # Remove punctuation
            # words = [word for word in tokens if word.isalpha()]


            tokens.append(stripped)
            del self.texts
            print("----- Tokenizando Palavras -----")
            print(tokens[0])
            # self._stemmer_text(self.tokens)

    def _stemmer_text(self, tokens):
        porter = PorterStemmer()
        lancaster = LancasterStemmer()
        return [porter.stem(t) for t in tokens], [lancaster.stem(t) for t in tokens]


if __name__ == '__main__':
    """."""

    texto_processor = ProcessTexts()
