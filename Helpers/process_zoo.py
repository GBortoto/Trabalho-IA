"""Class to process all texts."""

import pandas as pd
import numpy as np
import os

class ProcessZoo():
    """."""

    def __init__(self):
        """."""
        self.animal_names = []
        self._read_text()

    def _read_text(self):
        """."""
        print(os.listdir())
        df = pd.read_csv('database/zoo.csv', delimiter=',', header=None)
        self.animal_names = df.ix[:,0].tolist()
        new_df = df.drop(df.columns[0], axis=1)
        new_df = new_df.drop(df.columns[-1], axis=1)
        self.animais_matrix = np.array(new_df.values)

    def get_original_matrix(self):
        """."""
        return self.animais_matrix

    def get_animals_names(self):
        return self.animal_names
