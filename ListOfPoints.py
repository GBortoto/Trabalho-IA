class ListOfPoints():
    def __init__(self, size, dimentions):
        self.points = [[0.0 for j in range(dimentions)] for i in range(size)]

    def getNumberOfPoints(self):
        return len(self.points)

    def getNumberOfDimentions(self):
        if(len(self.points) == 0):
            return 0
        return len(self.points[0])

    def getDimention(self, dimention):
        """Retorna um vetor com uma dimenção específica de cada elemento"""
        return [self.points[i][dimention] for i in range(self.getNumberOfPoints())]
    
    def findAllPointsWith(self, value, dimention):
        """Retorna um ListOfPoints com todos os pontos que apresentam um valor específico em uma dimenção específica"""

        # dimention > numberOfDimentions
        if(dimention > (self.getNumberOfDimentions()-1)):
            return

        filteredPoints = []
        
        for i in range(len(self.points)):
            if(self.points[i][dimention] == value):
                filteredPoints.append(self.points[i] + [i])

        # Not found
        if(len(filteredPoints) == 0):
            return
            
        result = ListOfPoints(len(filteredPoints), len(filteredPoints[0]))
        
        for i in range(len(filteredPoints)):
            result.points[i] = filteredPoints[i]
        return result




    def __str__(self):
        return str(self.points)

    __len__ = getNumberOfPoints
    __repr__ = __str__
