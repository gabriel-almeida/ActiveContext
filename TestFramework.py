__author__ = 'gabriel'
import random
import numpy as np


class ContextSelection():
    def __init__(self):
        pass

    def obtain_train(self, train_set, target, context, n_context_choice):
        self.n_context_choice = n_context_choice
        _ , self.n_contexts = np.shape(context)
        #TODO SVD algorithm


    def obtain_contextual_train_sample(self, sample, target, context):
        pass

    def obtain_contextual_test_sample(self, sample, context):
        return random.random()

    def choose_contexts(self, sample):
        options = list(range(self.n_contexts))
        return random.sample(options, self.n_context_choice)


class TestFramework():
    def __init__(self, train_ratio=0.5, candidate_ratio=0.25, seed=None):
        self.train_ratio = train_ratio
        self.candidate_ratio = candidate_ratio
        random.seed(seed)


    def __prepare_holdout(self, nSample):
        ids = list(range(nSample))
        random.shuffle(ids)
        train_split = round(self.train_ratio * nSample)
        candidate_split = train_split + round(self.candidate_ratio * nSample)

        self.ids_train = ids[0:train_split]
        self.ids_candidate = ids[train_split:candidate_split]
        self.ids_test = ids[candidate_split:]

    def test_procedure(self, dataset, contexts_columns, target_column, n_context_choice, context_selectors, n_repetitions=10):
        (nSample, nFeature) = np.shape(dataset)
        data_columns = [i for i in range(nFeature) if i != target_column and i not in contexts_columns]

        data = dataset[:, data_columns]
        target = dataset[:,target_column]
        context = dataset[:, contexts_columns]

        for repetion in range(n_repetitions):
            self.__prepare_holdout(nSample)
            for selector in context_selectors:
                selector.obtain_train(data[self.ids_train, :], target[self.ids_train], context[self.ids_train, :], n_context_choice)

                for candidate in self.ids_candidate:
                    choosen_contexts = selector.choose_contexts(data[candidate, :])
                    selector.obtain_contextual_train_sample(data[candidate, :], target[candidate], context[candidate, choosen_contexts])
                    #any more steps here?

                for test in self.ids_test:
                    choosen_contexts = selector.choose_contexts(data[test, :])
                    prediction = selector.obtain_contextual_test_sample(data[test, :], context[test, choosen_contexts])
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