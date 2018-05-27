"""Class to process all texts."""

import os
import string

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer


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

        self._processa_text()

    def _processa_text(self, type='Porter'):
        print("----- Tokenizando Sentencas e Palavras -----")
        table = str.maketrans('', '', string.punctuation)
        stop_words = set(stopwords.words('english'))
        self.tokens = []

        # Para cada texto
        for index, text in enumerate(self._texts):
            # Tokenize por sentenca
            sentences = sent_tokenize(text)
            tokens_of_sentence = []
            # Para cada sentenca
            for sentence in sentences:
                # Tokenize por palavras, elimine stop words, pontuação e de lower
                stripped = [word.translate(table).lower() for word in word_tokenize(sentence) if not word in stop_words]
                stemmerized = self._normalize_text(tokens=stripped, type=type)
                tokens_of_sentence = tokens_of_sentence + stemmerized
            self.tokens.append(tokens_of_sentence)
        del self._texts
        self._join_words()

    def _normalize_text(self, tokens, type):
        if type is 'Porter':
            porter = PorterStemmer()
            return [porter.stem(t) for t in tokens]
        if type is 'Lancaster':
            lancaster = LancasterStemmer()
            return [lancaster.stem(t) for t in tokens]
        if type is 'Snowball':
            snowball = SnowballStemmer('english')
            return [snowball.stem(t) for t in tokens]
        if type is 'Lemma':
            lemma = WordNetLemmatizer()
            return [lemma.lemmatize(t) for t in tokens]

    def _join_words(self):
        new_tokens = []
        for token in self.tokens:
            # new_tokens.append((' '.join(token)).replace('  ', ' '))
            new_tokens.append(' '.join(token))
        self.tokens = new_tokens