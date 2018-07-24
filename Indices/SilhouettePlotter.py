import matplotlib.pyplot as plt
class SilhouettePlotter:
    def __init__(self):
        pass
    #nro de clusters x valor silhouette
    #formato dict { 2: 0.52561 , 3: 0.6521354 , 4: 079821}
    def plots(self , dictionary_nroClusters_Silhouette):
        plt.plot(dictionary_nroClusters_Silhouette.keys() , dictionary_nroClusters_Silhouette.values())
        plt.ylabel("valor silhouette")
        plt.xlabel("quantidade de clusters")
        plt.show()
