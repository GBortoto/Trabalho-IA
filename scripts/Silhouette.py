'''
Índice Silhouette

    Sendo i um dado,
        Axioma 1) Isil(i) = ( b(i) - a(i) ) / max{a(i), b(i)}
    representa o índice silhouette para o dado i

    onde:
        a(i) é a distância média do dado i a todos os demais dados do seu grupo
        b(i) é a distância média do dado i a todos os dados do grupo mais próximo ao seu
            (aka a menor distância média entre o dado e os dados dos outros grupos)

    Axioma 2) Isil(Grupo) = média(Isil(dado 1), Isil(dado 2), ..., Isil(dado N)  )

    Axioma 3) Isil(Agrupamento) = média(Isil(Grupo 1), Isil(Grupo 2), ..., Isil(Grupo N)  )

    O índice varia entre [-1, 1]
        1   = Ponto bem agrupado
        0   = a(i) ~= b(i) --> Não está claro se i deve pertencer ao grupo A ou ao grupo B
        -1  = Ponto mal agrupado


Referência: "Técnica de Agrupamento (Clustering)"
            Sarajane M. Peres e Clodoaldo A. M. Lima
            Slides 127-132
'''

class Silhouette():

    def __init__(self):

    def distanceBetweenPointAndPoint(pointA, pointB, typeOfDistance):
        """Retorna a distância entre os pontos A e B, dado um tipo de cálculo de distância"""

        if(len(pointA) != len(pointB)):
            raise ValueError("Silhouette - distanceBetweenPointAndPoint(): number of dimentions of PointA != number of dimentions of PointB")
        numberOfDimentions = len(pointA)

        distance = -1
        if(typeOfDistance == 'euclidean'):
            distance = sqrt(sum([(pointA[i] - pointB[i])**2 for i in range(numberOfDimentions)]))
        elif(typeOfDistance == 'cosine similarity'):
            distance = sum([pointA[i] * pointB[i] for i in range(numberOfDimentions)])
                        / (
                              sqrt(sum([pointA[i]**2 for i in range(numberOfDimentions)]))
                            * sqrt(sum([pointB[i]**2 for i in range(numberOfDimentions)]))
                        )
        # Caso o tipo de cálculo de distância seja inválido, o método jogará o erro
        if(distance == -1):
            raise ValueError('Silhouette - distanceBetweenPointAndPoint(): Invalid type of distance: "' + typeOfDistance + '"' )
        return distance

    def meanDistanceBetweenPointAndGroup(point, group, typeOfDistance):
        """Retorna a média das distâncias entre o ponto recebido e todos os pontos do grupo recebido"""
        return sum([distanceBetweenPointAndPoint(point, group[i], typeOfDistance) for i in range(len(group))]) / len(group)

    def findNearestGroup(point, groupOfPoint, groups, typeOfDistance):
        """Encontra o grupo mais próximo do ponto recebido"""

        # Cria uma lista de todos os grupos exceto o grupo atual do ponto
        otherGroups = groups.remove(groupOfPoint)

        # Calcula as médias de distância entre o ponto e todos os grupos da lista criada acima
        means = [meanDistanceBetweenPointAndGroup(point, otherGroups[i], typeOfDistance) for i in range(len(otherGroups))]

        # Retorna o grupo que possui a menor distância média em relação ao ponto
        return otherGroups[means.index(min(means))]


    # --------------------------- Public Functions -----------------------------

    def pointSilhouette(point, groupOfPoint, groups, typeOfDistance='euclidean'):
        """Calcula o índice Silhouette para um dado"""

        # Retorna o cálculo do índice Silhouette para o ponto (Axioma 1)
        Ai = meanDistanceBetweenPointAndGroup(point, groupOfPoint, typeOfDistance)
        Bi = meanDistanceBetweenPointAndGroup(point, findNearestGroup(point, groupOfPoint, groups), typeOfDistance)
        return (Bi - Ai) / max(Ai, Bi)

    def groupSilhouette(group, groups, typeOfDistance='euclidean'):
        """Calcula o índice Silhouette para um grupo de dados"""

        # Retorna a média dos silhouetes dos dados no grupo (Axioma 2)
        return sum(
            [pointSilhouette(clusterPoints[i], clusterPoints, nearestGroup, typeOfDistance) for i in range(len(group))
        )/len(group)

    def allGroupsSilhouette(groups, typeOfDistance='euclidean'):
        """Calcula o índice Silhouette para um grupo de grupos de dados"""

        # Retorna a média dos silhouetes dos grupos de dados (Axioma 3)
        return sum(
            [groupSilhouette(groups[i], groups, typeOfDistance) for i in range(len(groups))
        )/len(groups)

    # --------------------------------------------------------------------------
