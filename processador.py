import os, math
#texto = ['a', 'b', 'a', 'c', 'd', 'b', 'a']
#texto2 = ['a', 'q', 'q', 'r', 'r', 'b', 'a']
class ProcessaVetorTexto():

    # Inicia e cria variável texto além de criar o hash_map 
    # Recebe um  subtexto. Assim se corpus = [texto0, texto1, texto2], o ProcessaVetorTexto recebe texto0, depois manualmente pode-se adicionar textos1, etc..
    def __init__(self, subtexto):
        self.textos = []
        self.hash_maps = []
        self.numero_textos = 0
        # Adiciona um subtexto dentro dos possíveis corpus
        self.textos.append(subtexto)
        self.hash_maps.append({i:subtexto.count(i) for i in set(self.textos[0])})
        self.numero_textos += 1
        print(self.textos)
        print(self.hash_maps)

    def adiciona_texto(self, subtexto):
        self.textos.append(subtexto)
        self.hash_maps.append({i:subtexto.count(i) for i in set(self.textos[0])})
        self.numero_textos += 1
        return self.numero_textos - 1;

    def get_numero_textos(self):
        return self.numero_textos

    """
        NOME: TF - Term Frequency
        FUNCAO: Identificar o quão frequente é determinada palavra dentro de um corpus | #aparicoes
        RETORNO: Retorno #aparicoes com que determinada palavra aparecede em um corpus específico
    """
    def tf(self, numero_texto, palavra):
        return self.hash_maps[numero_texto][palavra]
        
    """
        NOME: TF Normalizado - Term Frequency Normalizado
        FUNCAO: Identificar o quão frequente é determinada palavra dentro de um corpus, normalizando em relação as outras palavras  | #aparicoes/#total
        RETORNO: Retornar a % (frequencia) com que determinada palavra aparece dentro de um corpus
    """
    def tf_n(self, numero_texto, palavra):
        return (self.hash_maps[numero_texto][palavra]/sum(self.hash_maps[numero_texto].values()))
      
    #ERROR - Encontrando 20 "The" em 21 textos - muito improvável não ter palavra "the" em qql texto
    """
        NOME: Word Containing
        FUNCAO: Identificar a somatória de quantas vezes uma determinada palavras esta contida nos textos de um determinado corpus | sum(texto_contem(palavra))
        RETORNO: sum(texto_contem(palavra))
    """
    def word_containing(self, palavra):
        somatoria = 0
        for iterador in range(self.numero_textos):
            if palavra in self.hash_maps[iterador]:
                somatoria += 1
        return somatoria

    """
        NOME: IDF - Inverse Document Frequency
        FUNCAO: Identificar o log do inverso do word containing. Cria um fator pelo qual é possivel medicar a frequencia da coleção (corpus), ao invés da frequencia de um documento
        RETORNO: log(#palavras) / word_containing(palavra)
    """
    # IDF - Inverse Document Frequency
    def idf(self, palavra):
        log = math.log([len(texto) for texto in self.textos])
        #print('Log', str(log))
        dividendo = (1 + word_containing(palavra))
        #print('Dividendo', dividendo)
        resultado = log / dividendo
        #print('Resultado', resultado)
        return resultado

    """
        NOME: TF-IDF - Term Frequency Inverse Document Frequency
        FUNCAO: Calcular a representação TF (nível de documento) com o IDF (nível de coleção) e identificar a representatividade deste palavra dentro do corpus como um todo
        RETORNO: TF * IDF (repare que, para valores com baixa aparição tanto em TF quanto em IDF, a representatividade desta palavra é diretamente afetada)
    """
    # TF-IDF - TF + IDF
    def tfidf(self, palavra, tokens, hash_map):
        return tf_n(palavra, hash_map) * idf(palavra, tokens)