from pandas.hashtable import na_sentinel

__author__ = 'gabriel'
import random
import numpy as np
import ContextualSVD

class OneHotEncoder():
    def fit(self, categorical_matrix, na_value=-1):
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
        self.n_features = sum(max_values)

        begin_index = np.cumsum(max_values)
        begin_index[-1] = 0
        self.begin_index = np.roll(begin_index, 1)
        return begin_index

    def predict(self, categorical_matrix):
        n_line, n_col = np.shape(categorical_matrix)
        result = np.zeros((n_line, self.n_features))
        for i in range(n_line):
            not_na_columns = categorical_matrix[i, :] != self.na_value
            result[i, categorical_matrix[i, not_na_columns] + self.begin_index[not_na_columns] - 1] = 1
        return result



class ContextSelection():
    def __init__(self, n_user, n_item, contextual_indexes, train_method = ContextualSVD.ContextualSVD()):
        self.train_method = train_method
        self.n_user = n_user
        self.n_item = n_item
        self.contextual_indexes = contextual_indexes
        _, self.n_contexts = np.shape(self.contextual_indexes)

        self.train_buffer = []
        self.context_buffer = []

    def obtain_initial_train(self, train_set, context, n_context_choice):
        if n_context_choice > len(self.contextual_indexes):
            raise Exception("Number of contextual choices should not be greater than the number of contexts")

        self.n_context_choice = n_context_choice
        self.train_set = train_set
        self.train_context = context

        self.train_method.train(self, self.train_set, self.train_context, self.n_user, self.n_item)

    def obtain_contextual_train_sample(self, sample, context):
        self.train_buffer.append(sample)
        self.context_buffer.append(context)

    def obtain_contextual_test_sample(self, sample, context):
        if len(self.train_buffer) > 10:
            train_set = np.vstack((self.train_set, self.train_buffer))
            train_context = np.vstack((self.train_context , self.context_buffer))
            #Free the buffers
            self.train_buffer = []
            self.context_buffer = []
            #retrain the model
            self.obtain_initial_train(train_set, train_context, self.n_context_choice)

        return self.train_method.predict_dataset(sample, context)

    def choose_contexts(self, sample):
        options = list(range(self.n_contexts))
        return random.sample(options, self.n_context_choice)

class TestFramework():
    def __init__(self, dataset, train_ratio=0.5, candidate_ratio=0.25):
        self.train_ratio = train_ratio
        self.candidate_ratio = candidate_ratio
        self.dataset = dataset

    def __prepare_holdout(self, n_sample):
        ids = list(range(n_sample))
        random.shuffle(ids)
        train_split = round(self.train_ratio * n_sample)
        candidate_split = train_split + round(self.candidate_ratio * n_sample)

        self.ids_train = ids[0:train_split]
        self.ids_candidate = ids[train_split:candidate_split]
        self.ids_test = ids[candidate_split:]

    def test_procedure(self, dataset, contexts_columns, target_column, n_context_choice, context_selectors, n_repetitions=10):
        (nSample, nFeature) = np.shape(dataset)
        data_columns = [i for i in range(nFeature) if i != target_column and i not in contexts_columns]

        data = dataset[:, data_columns]
        target = dataset[:,target_column]
        context = dataset[:, contexts_columns]

        for repetition in range(n_repetitions):
            self.__prepare_holdout(nSample)
            for selector in context_selectors:
                selector.obtain_train(data[self.ids_train, :], target[self.ids_train], context[self.ids_train, :], n_context_choice)

                for candidate in self.ids_candidate:
                    chosen_contexts = selector.choose_contexts(data[candidate, :])
                    selector.obtain_contextual_train_sample(data[candidate, :], target[candidate], context[candidate, chosen_contexts])
                    #any more steps here?

                for test in self.ids_test:
                    chosen_contexts = selector.choose_contexts(data[test, :])
                    prediction = selector.obtain_contextual_test_sample(data[test, :], context[test, chosen_contexts])
                    current_target = target[test]
                #TODO Calculate MAE etc
                #TODO Plot statistics

if __name__ == "__main__":
    ds = np.random.random((100, 10))
    tf = TestFramework()
    context_cols = [6, 7, 8]
    target_col = 9
    n_context_choice = 2
    tf.test_procedure(ds, context_cols, target_col, n_context_choice, [ContextSelection()])