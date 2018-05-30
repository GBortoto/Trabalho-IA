"""."""
# Ativar para rodar SOM local
# import sompy as sompy


class SomDefault():
    """."""

    def __init__(self, data):
        """."""
        som = sompy.SOMFactory.build(data, [50, 50], mask=None, mapshape='planar', lattice='rect', normalization='var', initialization='pca', neighborhood='gaussian', training='batch', name='sompy')
        som.train(n_job=3, verbose='info')
        v = sompy.mapview.View2DPacked(50, 50, 'test',text_size=8)
        # could be done in a one-liner: sompy.mapview.View2DPacked(300, 300, 'test').show(som)
        v.show(som, what='codebook', which_dim=[0,1], cmap=None, col_sz=6) #which_dim='all' default
        # v.save('2d_packed_test')
