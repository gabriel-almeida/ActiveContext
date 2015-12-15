__author__ = 'gabriel'
import numpy as np

class MAE():
    def __init__(self, aggregation='user'):
        self._aggregation = aggregation

    def _aggregation_map(self, dataset, key_column):
        agg_map = {}
        n_dataset = np.shape(dataset)[0]
        for i in range(n_dataset):
            key = dataset[i, key_column]
            if key not in agg_map:
                agg_map[key] = []
            agg_map[key].append(i)
        return agg_map

    def _eval_mae(self, y, y_hat):
        return np.mean(np.abs(y - y_hat))

    def calculate(self, dataset, y_hat):
        aggregation_test = None

        if self._aggregation == 'user':
            aggregation_test = self._aggregation_map(dataset, 0)
        elif self._aggregation == 'item':
            aggregation_test  = self._aggregation_map(dataset, 1)

        if aggregation_test != None:
            agg_test = np.array([[self._eval_mae(dataset[eval_set, 2], y_hat[eval_set]), len(eval_set)] for (key, eval_set) in aggregation_test.items()])
            return np.average(agg_test[:, 0], weights=agg_test[:, 1])
        else:
            return self._eval_mae(dataset[:, 2], y_hat)
