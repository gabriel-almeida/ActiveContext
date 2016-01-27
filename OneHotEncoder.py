__author__ = 'gabriel'
import numpy as np

class OneHotEncoder():
    def __init__(self, categorical_matrix, na_value=-1):
        '''
        Returns the one-hot-encoded feature matrix of a
        categorical matrix. Assumes a matrix with categorical variables,
        indicated by a number from 1 to N on
        every column, where N is the number of possible categories.
        Resulting matrix  will have the same amount of lines and
        the sum of all N's as the number of columns.
        '''
        self.na_value = na_value
        max_values = np.max(categorical_matrix, axis=0)
        self.n_contextual_condition = sum(max_values)
        self.n_contextual_factor = np.shape(categorical_matrix)[1]

        begin_index = np.cumsum(max_values)
        begin_index[-1] = 0

        #indexes indicating where each contextual factor begins
        self.contextual_factor_index = np.roll(begin_index, 1)

    def predict(self, categorical_matrix):
        n_line = np.shape(categorical_matrix)[0]
        result = np.zeros((n_line, self.n_contextual_condition))
        for i in range(n_line):
            not_na_columns = categorical_matrix[i, :] != self.na_value
            a = categorical_matrix[i, not_na_columns]
            b = self.contextual_factor_index[not_na_columns]
            c = a + b - 1
            c = c.astype(int)

            result[i, c] = 1
        return result
