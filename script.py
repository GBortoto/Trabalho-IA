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

class TransformMatrix():
	def __init__(self, matrix):
		# Guarda matrix de lista de palavras por texto
		self.matrix = matrix

		# Cria matrix
		self._matrix_creation()

	def _matrix_creation(self):
		# Iremos criar uma "vetorizacao" baseado em frequencia (count)
		vectorizer = CountVectorizer()

		#Retorna array TF de cada palavra
		self.bag_of_words = (vectorizer.fit_transform(self.matrix)).toarray()

		# Retorna array com as palavras (labels)
		self.feature_names = vectorizer.get_feature_names()

	# Matrix binaria será sempre a matrix TF para os casos em que a frequencia é diferente de 0
	def matrix_binaria(self):
		# Método sign identifica se numero != 0
		return (sp.sign(self.bag_of_words))

	# Matrix TF somente com frequencia da palavra, independente da frequencia relativa do corpus
	def matrix_tf(self):
		return self.bag_of_words

	# Matrix TF normalizada com frequencia indo de [0, 1)
	def matrix_tf_normalizada(self):
		listas = [np.sum(lista, axis=0) for lista in self.bag_of_words]
		result = sum(listas)
		return self.bag_of_words / result

	# Matrix TF_IDF que utiliza inverse document
	def matrix_tfidf(self):
		tfidf_vectorize = TfidfTransformer(smooth_idf=False)
		return tfidf_vectorize.fit_transform(self.bag_of_words).toarray()
class Kmeans():

    def __init__(self, type_of_kmeans, points):
        self.type_of_kmeans = type_of_kmeans
        self.points = points

    def see_points(self):
        # plt.scatter(points[:,0], points[:,1])
        ax = plt.gca()

    def inicia_centroides(self, k_centroids):
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        self.centroids = centroids[:k_centroids]

    def busca_centroides_mais_proximo(self):
        distancias = np.sqrt(((self.points - self.centroids[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distancias, axis=0)

    def roda_kmeans(self):
        self.inicia_centroides(4)
        self.movimenta_centroides(self.busca_centroid_mais_proximo())

    def movimenta_centroides(self, closest):
        return np.array([self.points[closest == k].mean(axis=0) for k in range(self.centroids.shape[0])])
# -*- coding: utf-8 -*-

class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """

    #To check if the SOM has been trained
    _trained = False

    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        m x n -> dimensao do SOM
        n_interations -> #epocas que será treinado a rede
        alpha -> taxa de aprendizagem. Default 0.3
        sigma -> taxa de vizinhança. Define o raio que o BMU afeta. Default max(m, n)
        """

        #Assign required variables first
        self._m = m
        self._n = n
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))

        ##INITIALIZE GRAPH - TF Graphs é o confunto de operações que serão realizadas + operadores
        self._graph = tf.Graph()

        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():

            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE

            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
            self._weightage_vects = tf.Variable(tf.random_normal([m*n, dim]))

            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))

            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training

            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")

            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training

            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.sub(self._weightage_vects, tf.pack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)

            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])

            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.sub(1.0, tf.div(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.mul(alpha, learning_rate_op)
            _sigma_op = tf.mul(sigma, learning_rate_op)

            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.sub(
                self._location_vects, tf.pack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.neg(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.mul(_alpha_op, neighbourhood_func)

            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.pack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.mul(
                learning_rate_multiplier,
                tf.sub(tf.pack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)

            ##INITIALIZE SESSION
            self._sess = tf.Session()

            ##INITIALIZE VARIABLES
            init_op = tf.initialize_all_variables()
            self._sess.run(init_op)

    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])

    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """

        #Training iterations
        for iter_no in range(self._n_iterations):
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})

        #Store a centroid grid for easy retrieval later on
        centroid_grid = [[] for i in range(self._m)]
        self._weightages = list(self._sess.run(self._weightage_vects))
        self._locations = list(self._sess.run(self._location_vects))
        for i, loc in enumerate(self._locations):
            centroid_grid[loc[0]].append(self._weightages[i])
        self._centroid_grid = centroid_grid

        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid

    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")

        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])

        return to_return
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
    7-aux) Bag of Words (recebe um texto em string) ____________________________
    - FUNÇÃO  :
    - RETORNO :
    """
    # VARIÁVEL GLOBAL UTILIZADA : vectorizer
    def bagOfWordsForTextInString(self, text: str):
        return vectorizer.fit_transform(text).todense()

    """
    7) Bag of Words ____________________________________________________________
    - FUNÇÃO  :
    - RETORNO :
    """
    def bagOfWords(self, listOfWords: Ligedit st[str]):
        text = " ".join(listOfWords)
        return bagOfWordsForTextInString(text)

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
# -*- coding: utf-8 -*-

if __name__ == "__main__":
	preprocessor = Preprocessor()
	texts = preprocessor.readAllTextsFromDatabase()
	#texts contêm todos os textos que serão utilizados de forma que cada index do array tem uma notícia. As notícias não estão tratadas , são o texto puro , retirado apenas os e-mails e em ordem aleatória.
	texts = preprocessor.processTexts(texts)
	with open("results.txt", 'w') as output:
		for item in texts[0]:
			output.write(item)

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
