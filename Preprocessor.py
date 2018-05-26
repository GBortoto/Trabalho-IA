# -*- coding: utf-8 -*-
import string                                                   # Lista de caracteres de pontuação
import os                                                       # Miscellaneous operating system interfaces
import re                                                       # Regular Expressions
import random                                                   # Python Random Library
import scipy as sp
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize                         # Tokenizer
from nltk.corpus import stopwords                               # Stop Words
from nltk.stem.porter import *                                  # Stemmer - Porter
from nltk.stem.snowball import SnowballStemmer                  # Stemmer - Snowball
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from typing import List                                         # Anotação de help quando uma função é escrita
from pympler import asizeof

"""
-------------------- Preprocessor --------------------
Properties:
    - translate_table_for_punctuation
    - translate_table_for_numbers
    - stopWords
    - porterStemmer
    - snowballerStemmer
    - vectorizer

Methods:
    1) readTextsFromFolder(directoryPath: str ='../input/') -> List[str]

    2-aux) aux_tokenize(self, text: str) -> List[str]
        2) tokenizeTexts(self, listOfTexts: List[str]) -> List[List[str]]

    3-aux) aux_removePunctuation(self, word: str) -> str:
        3) removePunctuationFromText(self, text: List[str]) -> List[str]

    4-aux) aux_removeNumbers(self, word: str) -> str
        4) removeNumbersFromText(self, text: List[str]) -> List[str]

        5) removeStopWords(self, listOfWords: List[str]) -> List[str]

    6-aux) aux_applyStemmer(self, word: str, stemmerOption: str ="snowball") -> str
        6) applyStemmerOnText(self, listOfWords: List[str], stemmerOption: str ="snowball") -> List[str]

    7-aux) bagOfWordsForTextInString(self, text: str)
        7) bagOfWords(self, listOfWords: List[str])


"""

class Preprocessor():
    """Classe que lida com todo o pré processamento dos textos"""
    # As variáveis criadas no corpo da classe serão compartilhadas por todas as instâncias


    # Construtor -> As variáveis criadas aqui serão validas apenas para a instância atual
    def __init__(self):
        self.translate_table_for_punctuation = dict((ord(char), None) for char in string.punctuation)
        self.translate_table_for_numbers = dict((ord(char), None) for char in string.digits)
        self.stopWords = set(stopwords.words('english'))

        # Explicar diferenca entre PorterStemmer; SnowballStemmer e CountVectorizer
        self.porterStemmer = PorterStemmer()

        # Pq só nesse temos ingles e ignore_stopwords?
        self.snowballStemmer = SnowballStemmer("english", ignore_stopwords=True)
        self.vectorizer = CountVectorizer()
        self.intervaloDeLog = 2000

    """
    2) Tokenizer para Lista de Textos __________________________________________
    - FUNÇÃO  : Transforma uma lista de textos em uma lista de listas de palavras
    - RETORNO : Lista de listas de palavras
    """
    def tokenizeTexts(self, listOfTexts: List[str]) -> List[List[str]]:
        """Transforma uma lista de textos em uma lista de listas de palavras"""
        listOfListsOfWords = []
        intervalo = 0
        totalTextos = 0
        for text in listOfTexts:
            if(intervalo >= self.intervaloDeLog):
                totalTextos += intervalo
                print(str(totalTextos) + ' textos foram tokenizados. \n')
                #print(listOfListsOfWords)
                intervalo = 0
            intervalo += 1

            listOfListsOfWords.append(word_tokenize(text, language='english'))
            #listOfListsOfWords.append([word_tokenize(phrase) for phrase in sent_tokenize(text.translate(dict.fromkeys(string.punctuation)).translate(dict.fromkeys(string.digits)))])

            #To create table
            #table = str.maketrans('', '', string.punctuation)
            #Remove pontuação de todo o texto
            #stripped = [w.translate(table) for w in words]

            #Add everything as lower
            #words = [word.lower() for word in words]

            #Word tokenize roda o Treebank tokenize, porem deveriamos considerar que o texto ja teria rodado pelo sent_tokenize
            #listOfListsOfWords.append(word_tokenize(sent_tokenize(text, language='english'), language='english'))
        print('Todos os textos foram tokenizados.')
        return listOfListsOfWords

    """
    3-aux) Removedor de Pontuação ______________________________________________
    - FUNÇÃO  : Remove o seguinte grupo de caracteres de uma palavra
                !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    - RETORNO : Palavra filtrada
    """
    # VARIÁVEL GLOBAL UTILIZADA : translate_table_for_punctuation
    def aux_removePunctuation(self, word: str) -> str:
        """Remove caracteres de pontuação"""
        return word.translate(self.translate_table_for_punctuation)

    """
    3) Removedor de Pontuação para Lista de Palavras ___________________________
    - FUNÇÃO  : Remove o seguinte grupo de caracteres de uma lista de palavras
                !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~
    - RETORNO : Lista filtrada de palavras
    """
    def removePunctuationFromText(self, text: List[str]) -> List[str]:
        """Remove caracteres de pontuação de uma lista de palavras"""
        newListOfWords = []
        for word in text:
            newWord = self.aux_removePunctuation(word)
            if(newWord != ""):
                newListOfWords.append(newWord)
        return newListOfWords

    """
    4-aux) Removedor de Números ________________________________________________
    - FUNÇÃO  : Remove números de uma palavra
    - RETORNO : Palavra filtrada
    """
    # VARIÁVEL GLOBAL UTILIZADA : translate_table_for_numbers
    def aux_removeNumbers(self, word: str) -> str:
        """Remove caracteres de pontuação"""
        return word.translate(self.translate_table_for_numbers)

    """
    4) Removedor de Números para Lista de Palavras _____________________________
    - FUNÇÃO  : Remove números de uma lista de palavras
    - RETORNO : Lista filtrada de palavras
    """
    def removeNumbersFromText(self, text: List[str]) -> List[str]:
        """Remove números de uma lista de palavras"""
        newListOfWords = []
        for word in text:
            newWord = self.aux_removeNumbers(word)
            if(newWord != ""):
                newListOfWords.append(newWord)
        return newListOfWords

    """
    5) Removedor de Stop Words _________________________________________________
    - FUNÇÃO  : Remove de um grupo de palavras as palavras presentes na lista de palavras 'stopWords'
    - RETORNO : Grupo filtrado de palavras
    """
    # VARIÁVEL GLOBAL UTILIZADA : stopWords
    def removeStopWords(self, listOfWords: List[str]) -> List[str]:
        """Remove stopwords de uma lista de palavras"""
        wordsFiltered = []
        for w in listOfWords:
            if w not in self.stopWords:
                wordsFiltered.append(w)
        return wordsFiltered

    """
    6-aux) Stemmer _____________________________________________________________
    - FUNÇÃO  : Remove afixos morfológicos de uma palavra
    - RETORNO : Radical da palavra de entrada
    """
    # VARIÁVEIS GLOBAIS UTILIZADAS : porterStemmer, snowballStemmer
    def aux_applyStemmer(self, word: str, stemmerOption: str ="snowball") -> str:
        """Remove afixos morfológicos de uma palavra"""
        if(not word):
            print('ERROR - applyStemmer(): Can\'t apply stemmer on empty word')
            return

        if(stemmerOption):
            stemmerOption = stemmerOption.lower()
            if(stemmerOption == 'porter'):
                return self.porterStemmer.stem(word)
            elif(stemmerOption == 'snowball'):
                return self.snowballStemmer.stem(word)
        print('ERROR - applyStemmer(): "' + str(stemmerOption) + '" is not a valid stemmer')
        return

    """
    6) Stemmer para lista de palavras __________________________________________
    - FUNÇÃO  : Remove afixos morfológicos das palavras de uma lista de palavras
    - RETORNO : Radical da palavra de entrada
    """
    def applyStemmerOnText(self, listOfWords: List[str], stemmerOption: str ="snowball") -> List[str]:
        newListOfWords = []
        for word in listOfWords:
            newWord = self.aux_applyStemmer(word, stemmerOption)
            if(newWord != ""):
                newListOfWords.append(newWord)
        return newListOfWords

    """
    8) Erase Emails
    - FUNÇÃO  : recebe uma lista de strings e apaga todos os emails dentro dela
    - RETORNO : retorna uma lista de string sem e-mails dentro dela
    """
    def eraseEmails(self,listOfTexts:List[str]) -> List[str]:
        regex = re.compile('([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)')
        for index, text in enumerate(listOfTexts):
            newText = re.sub(regex, '', text)
            listOfTexts[index] = newText
        return listOfTexts

    """
    9) readTexts
    - FUNÇÃO  : recebe - directory: uma string com o caminho até a pasta com os dados,
                         ignoredLines: Array de strings, não ocorrerá a leitura das
                                        linhas que forem iniciadas com um desses valores
                        suffix: só lê os documentos que terminarem com esse sufixo

                faz:    remove os headers irrelevantes pra análise do texto e
                        remove os emails dentro do texto
                        apenas lê os documentos com o sufixo determinado

    - RETORNO : retorna uma lista de string com cada um dos textos em uma string
    """

    def readTexts(self, directory:str, ignoredLines:List[str] = [] , suffix:str = '.txt') -> List[str]:
        allNews = []
        news = ''
        breakNewsBlock = False
        listOfFiles = os.listdir(directory)
        print('Lendo arquivos da pasta: ' + directory)
        for fileName in listOfFiles:
            # Apenas lê arquivos com o sufixo passado como parametro
            if (fileName.endswith(suffix)):
                with open(directory + fileName, 'r', encoding='latin-1') as textfile:
                    for fileLine in textfile:
                        readLine = True
                        line = fileLine.lstrip().replace('\n', '')
                        # verifica se é uma linha de header
                        for ignoredStrings in ignoredLines:
                            if (line.startswith(ignoredStrings)):
                                readLine = False
                                breakNewsBlock = True
                                break
                        if (readLine):
                            # Se for uma linha valida e tiver passado por um header, adiciona nova string ao vetor
                            if (breakNewsBlock and news != ""):
                                allNews.append(news)
                                news = ""
                                breakNewsBlock = False
                            news += line
                    allNews.append(news)
                    news = ""
        allNews = self.eraseEmails(allNews)
        return allNews

    """
       10) read All 20NewsGroup
       - FUNÇÃO  : recebe uma string com o caminho até a pasta com os dados do News20Group,
                           remove os headers irrelevantes pra análise do texto e
                           remove os emails dentro do texto
                           randomiza a ordem dos textos.
       - RETORNO : retorna uma lista de string com cada um dos textos em uma string
    """
    #def readAll20NewsGroup(self,directory:str = 'database/20NewsGroup/') -> List[str]:
    def readAll20NewsGroup(self,directory:str = '../input/') -> List[str]:
        ignoredLines = ['Newsgroup:', 'Document_id:', 'document_id:', 'From:']
        allNews = self.readTexts(directory, ignoredLines)
        random.shuffle(allNews)
        return allNews

    """
          11) read All Bbc
          - FUNÇÃO  : recebe uma string com o caminho até a pasta com as pastas da BBC,
                              remove os headers irrelevantes pra análise do texto e
                              remove os emails dentro do texto
                              randomiza a ordem dos textos.
          - RETORNO : retorna uma lista de string com cada um dos textos em uma string
       """

    def readAllBbc(self, directory:str='database/bbcFiles/') ->List[str]:
        listaPastas = os.listdir(directory)
        allNews = []
        for pasta in listaPastas:
            caminho = directory + pasta
            print(caminho)
            isPasta = os.path.isdir(caminho)
            if(isPasta):
                allNews.extend(self.readTexts(caminho+'/'))
        random.shuffle(allNews)
        return allNews

    """
    12) read All Texts From Database
    - FUNÇÃO  : recebe: dir20News: uma string com o caminho até a pasta com os dados do 20NewsGroup,
                               dirBbcFiles: uma string com o caminho até as pastas com as pastas do BBC

                               remove os headers irrelevantes pra análise do texto e
                               remove os emails dentro do texto
                               randomiza a ordem dos textos.
    - RETORNO : retorna uma lista de string com cada um dos textos em uma string
    """

    #def readAllTextsFromDatabase(self, dir20News:str ='database/20NewsGroup/' , dirBbcFiles:str = 'database/bbcFiles/') -> List[str]:
    def readAllTextsFromDatabase(self, dir20News:str ='../input/' , dirBbcFiles:str = '../input/') -> List[str]:
        allNews = []
        allNews += self.readAll20NewsGroup(dir20News)
        #allNews += self.readAllBbc(dirBbcFiles)
        #random.shuffle(allNews)
        print(str(len(allNews)) + ' Documentos encontrados.')
        print("Tamanho total do arquivo da lista em memória: "+ str(asizeof.asizeof(allNews)/1000000) + " MB")
        return allNews

    """
        processTexts
        - FUNÇÃO  : recebe: texts - lista de todos os textos puros (com o conteudo original)

                                   separa os textos em palavras
                                   remove pontuação , números , stop words
                                   aplica o stemmer no texto
        - RETORNO : retorna uma lista com todos os textos e cada texto separado por palavras.
                Ex: [texto0[palavra0,palavra1,palavra2], texto1[palavra0,palavra1]]
        """

    def processTexts(self, texts: List[str]) -> List[List[str]]:
        #lista de palavras ex: [texto0[palavra0,palavra1,palavra2], texto1[palavra0,palavra1]]
        print('Iniciando processo de tokenização dos textos!')
        listaPalavras = self.tokenizeTexts(texts)

        listaProcessada =[]
        intervalo = 0
        totalTextos = 0
        for texto in listaPalavras:
            if(intervalo >= self.intervaloDeLog):
                totalTextos += intervalo
                print(str(totalTextos) + ' textos foram processados \n')
                intervalo = 0
            intervalo += 1
            texto = self.removePunctuationFromText(texto)
            texto = self.removeNumbersFromText(texto)
            texto = self.removeStopWords(texto)
            texto = self.applyStemmerOnText(texto)
            listaProcessada.append(texto)
        print('todos os textos foram processados')
        print("tamanho total da lista gerada é de " + str(asizeof.asizeof(listaProcessada)/1000000) + " MB")
        return listaProcessada
