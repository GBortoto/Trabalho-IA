import numpy as np
import sys
import inspect


class NormalizatorFactory(object):

    @staticmethod
    def build(type_name):
        for name, obj in inspect.getmembers(sys.modules[__name__]):
            if inspect.isclass(obj):
                if hasattr(obj, 'name') and type_name == obj.name:
                    return obj()
        else:
            raise Exception("Unknown normalization type '%s'" % type_name)


class Normalizator(object):

    def normalize(self, data):
        raise NotImplementedError()

    def normalize_by(self, raw_data, data):
        raise NotImplementedError()

    def denormalize_by(self, raw_data, data):
        raise NotImplementedError()


class VarianceNormalizator(Normalizator):

    name = 'var'

    def _mean_and_standard_dev(self, data):
        print('mean-std dev')
        return np.mean(data, axis=0), np.std(data, axis=0)

    def normalize(self, data):
        print('normalize')
        me, st = self._mean_and_standard_dev(data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def normalize_by(self, raw_data, data):
        print('normalize-by')
        me, st = self._mean_and_standard_dev(raw_data)
        st[st == 0] = 1  # prevent: when sd = 0, normalized result = NaN
        return (data-me)/st

    def denormalize_by(self, data_by, n_vect):
        print('denormalize-by')
        me, st = self._mean_and_standard_dev(data_by)
        return n_vect * st + me
