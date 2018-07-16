import numpy as np
import sompy
#import matplotlib.pylab as plt

# A toy example: two dimensional data, four clusters
# fig = plt.figure()
# plt.plot(Data1[:,0],Data1[:,1],'ob',alpha=0.2, markersize=4)
# fig.set_size_inches(7,7)
# plt.savefig('teste.jpg')


class SOM():
    """."""

    def __init__(self):
        """."""
        self.parametros = self.le_parametros()


    def train_som(self,mapsize):
        """."""
        # this will use the default parameters, but i can change the initialization and neighborhood methods
        self.som = sompy.SOMFactory.build(self.matrix, mapsize, mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='animal_som')
        self.som.train(n_job=1, verbose='info')  # verbose='debug' will print more, and verbose=None wont print anything
        # trained_som = self.som.train(n_job=1, verbose='info')
        # som.set_parameter(neighbor=0.1, learning_rate=0.2)

    def view2dpacked(self,parametro):
        """."""
        v = sompy.mapview.View2DPacked(50, 50, 'test', text_size=8)
        # could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
        # which_dim='all' default
        # 1 visualizacao
        v.show(self.som, what='codebook', which_dim=[0, 1], cmap=None, col_sz=6)
        v.save(str(parametro + 'img1'))

        # 2 visualizacao
        self.som.component_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
        v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
        v.show(self.som, what='codebook', which_dim='all', cmap='jet', col_sz=6) #which_dim='all' default
        v.save(str(parametro + 'img2'))

        # c = sompy.mapview.View2DPacked()
        v = sompy.mapview.View2DPacked(2, 2, 'test',text_size=8)
        #first you can do clustering. Currently only K-means on top of the trained som
        cl = self.som.cluster(n_clusters=7)
        # print cl
        getattr(self.som, 'cluster_labels')

        v.show(self.som, what='cluster')
        v.save(str(parametro + 'img3'))

        h = sompy.hitmap.HitMapView(10, 10, 'hitmap', text_size=8, show_text=True)
        h.show(self.som)
        v.save(str(parametro + 'img4'))

    def visualization_umatrix(self, parametro):
        """."""
        u = sompy.umatrix.UMatrixView(50, 50, 'umatrix', show_axis=True, text_size=8, show_text=True)
        #This is the Umat value
        UMAT  = u.build_u_matrix(self.som, distance=1, row_normalized=False)

        #Here you have Umatrix plus its render
        UMAT = u.show(self.som, distance2=1, row_normalized=False, show_data=True, contooor=True, blob=False)
        #Commentado por Vinícius. Motivo: execução causa erro : File "/usr/local/lib/python3.5/dist-packages/sompy/visualization/view.py", line 50, in save
        #self._fig.savefig(filename, transparent=transparent, dpi=dpi,
        #AttributeError: 'NoneType' object has no attribute 'savefig'
        #necessário maior analise
        #u.save(str(parametro + 'img5'))

    def interpolation(self):
        plt.imshow(self.som, interpolation='none')
        plt.show()

    def executar(self , dados):
        self.matrix = dados
        for parametro in self.parametros:
            mapsize = [int(parametro[0])*10 , int(parametro[0])*10 ]
            self.train_som(mapsize)
            self.view2dpacked(parametro)
            self.visualization_umatrix(parametro)
            #self.interpolation()

    def le_parametros(self):
        """Parametros para inpute no KMEANS e SOM.
        1 - tamanho do mapsize
        2 - tipo de vizinhança (gaussiana, bubble)
        3 - inicializando do SOM (PCA, aleatoria)
        4 - SOM clusters para visualizacao
        """
        parametros = None
        with open('./parametros.txt', 'r') as inp:
            parametros = inp.read().splitlines()
        return parametros
