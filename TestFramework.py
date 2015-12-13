from jinja2.nodes import Concat

__author__ = 'gabriel'
import random
import numpy as np
import ContextualSVD


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


class RandomContextSelection():
    def __init__(self, train_method, encoder):
        self.train_method = train_method
        self.encoder = encoder

        self.train_buffer = []
        self.context_buffer = []

    def obtain_initial_train(self, train_set, context, n_context_choice):
        #if n_context_choice > len(self.contextual_indexes):
        #    raise Exception("Number of contextual choices should not be greater than the number of contexts")

        self.n_context_choice = n_context_choice
        self.train_set = train_set
        self.train_context = context

        self.train_method.train(self.train_set, self.encoder.predict(self.train_context))

    def obtain_contextual_train_sample(self, sample, context, chosen_contexts):
        encoded_context = np.ones(self.encoder.n_contextual_factor) * self.encoder.na_value
        encoded_context[chosen_contexts] = context

        self.train_buffer.append(sample)
        self.context_buffer.append(encoded_context)

    def obtain_contextual_test_sample(self, sample, context):
        if len(self.train_buffer) > 0:
            train_set = np.vstack((self.train_set, self.train_buffer))
            train_context = np.vstack((self.train_context, self.context_buffer))
            #Free the buffers
            self.train_buffer = []
            self.context_buffer = []
            #retrain the model
            self.obtain_initial_train(train_set, train_context, self.n_context_choice)
        context = np.reshape(context, (1, self.encoder.n_contextual_factor))
        return self.train_method.predict_rating(sample[0], sample[1], self.encoder.predict(context))

    def choose_contexts(self, sample):
        options = list(range(self.encoder.n_contextual_factor))
        return random.sample(options, self.n_context_choice)


class LargestDeviationContextSelection(RandomContextSelection):
    def __init__(self, train_method, encoder):
        RandomContextSelection.__init__(self, train_method, encoder)

    def obtain_initial_train(self, train_set, context, n_context_choice):
        RandomContextSelection.obtain_initial_train(self, train_set, context, n_context_choice)
        self.normalized_frequency = np.mean(self.encoder.predict(context), axis=0)

    def choose_contexts(self, sample):
        n_contextual_conditions = self.encoder.n_contextual_condition
        context_index = self.encoder.contextual_factor_index
        n_contextual_factors = self.encoder.n_contextual_factor

        all_contexts = np.eye(n_contextual_conditions)
        without_context = np.zeros((1, n_contextual_conditions))
        repeated_sample = np.tile(sample, (n_contextual_conditions, 1))

        prediction_with_contexts = self.train_method.predict_dataset(repeated_sample, all_contexts)
        prediction_without_context = self.train_method.predict_rating(sample[0], sample[1], without_context)
        deviation = np.abs(prediction_with_contexts - prediction_without_context)

        contextual_condition_weight = np.multiply(self.normalized_frequency, deviation)
        contextual_factor_weight = np.zeros((1, n_contextual_factors))

        for i in range(n_contextual_factors - 1):
            contextual_factor_weight[0, i] = np.mean(contextual_condition_weight[i:i+1])

        #get the last n elements
        context_choice = np.argsort(contextual_factor_weight)[0, n_contextual_factors - self.n_context_choice:]
        return context_choice

class TestFramework():
    def __init__(self, dataset, context, train_ratio=0.5, candidate_ratio=0.25, user_column = 0, item_column = 1):
        self.train_ratio = train_ratio
        self.candidate_ratio = candidate_ratio
        self.dataset = dataset
        self.context = context

    def __prepare_holdout(self, n_sample):
        ids = list(range(n_sample))
        random.shuffle(ids)
        train_split = round(self.train_ratio * n_sample)
        candidate_split = train_split + round(self.candidate_ratio * n_sample)

        self.ids_train = ids[0:train_split]
        self.ids_candidate = ids[train_split:candidate_split]
        self.ids_test = ids[candidate_split:]

    def test_procedure(self, n_context_choice, context_selectors, n_repetitions=10):
        (nSample, nFeature) = np.shape(self.dataset)

        #cast single object to list
        if type(context_selectors) is not list:
            context_selectors = [context_selectors]

        for repetition in range(n_repetitions):
            self.__prepare_holdout(nSample)
            for selector in context_selectors:
                selector.obtain_initial_train(self.dataset[self.ids_train, :], self.context[self.ids_train, :], n_context_choice)

                for candidate in self.ids_candidate:
                    chosen_contexts = selector.choose_contexts(self.dataset[candidate, :])
                    selector.obtain_contextual_train_sample(self.dataset[candidate, :], self.context[candidate, chosen_contexts], chosen_contexts)
                    #any more steps here?

                responses = []
                for test in self.ids_test:
                    #chosen_contexts = selector.choose_contexts(self.dataset[test, :])
                    #prediction = selector.obtain_contextual_test_sample(self.dataset[test, :], self.context[test, chosen_contexts])
                    prediction = selector.obtain_contextual_test_sample(self.dataset[test, :], self.context[test, :])
                    current_target = self.dataset[test, 2]
                    responses.append([prediction, current_target])

                responses = np.array(responses)
                res = mae(responses[:, 0], responses[:, 1])
                print(str(selector), res)
                #TODO Plot statistics

def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))

if __name__ == "__main__":
    import pandas as pd
    random.seed(101)

    #file = "/home/gabriel/Dropbox/Mestrado/Sistemas de Recomendação/Datasets/ldos_comoda.csv"
    file = "/home/gabriel/ldos_comoda.csv"

    m = pd.read_csv(file, header=None)
    n_context_choice = 1

    dataset = m.values[:, 0:3]
    context = m.values[:, 7: 19]

    n_user = np.max(dataset[:, 0]) + 1
    n_item = np.max(dataset[:, 1]) + 1

    svd = ContextualSVD.ContextualSVD(n_user, n_item)
    encoder = OneHotEncoder(context, na_value=-1)

    largest_deviation = LargestDeviationContextSelection(svd, encoder)
    random_choice = RandomContextSelection(svd, encoder)
    selectors = [largest_deviation, random_choice]

    tf = TestFramework(dataset, context)
    tf.test_procedure(1, selectors)