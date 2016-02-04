import numpy as np
import ContextualSVD
import matplotlib.pyplot as plt
import mae
from OneHotEncoder import OneHotEncoder
from Selectors import LargestDeviationContextSelection, AllContextSelection,\
    CramerLargestDeviation, RandomContextSelection
import copy
import multiprocessing
import random
import pprint
from scipy.stats import t

class TestFramework():
    def __init__(self, dataset, context, train_ratio=0.5, candidate_ratio=0.25, user_column = 0, item_column = 1, seed = None):
        self.train_ratio = train_ratio
        self.candidate_ratio = candidate_ratio
        self.dataset = dataset
        self.context = context
        self.seed = seed
        random.seed(seed)

    def __prepare_holdout(self, n_sample):
        ids = list(range(n_sample))
        random.shuffle(ids)
        train_split = round(self.train_ratio * n_sample)
        candidate_split = train_split + round(self.candidate_ratio * n_sample)

        self.ids_train = ids[0:train_split]
        self.ids_candidate = ids[train_split:candidate_split]
        self.ids_test = ids[candidate_split:]

    def _test_selector(self, selector_item):
        selector = selector_item[1]
        selector_name = selector_item[0]

        selector.obtain_initial_train(self.dataset[self.ids_train, :], self.context[self.ids_train, :], self.n_context_choice)

        all_chosen_context = []
        for candidate in self.ids_candidate:
            chosen_contexts = selector.choose_contexts(self.dataset[candidate, :])
            all_chosen_context.append(chosen_contexts) # debug
            selector.obtain_contextual_train_sample(self.dataset[candidate, :], self.context[candidate, chosen_contexts], chosen_contexts)

        responses = []
        for test in self.ids_test:
            # chosen_contexts = selector.choose_contexts(self.dataset[test, :])
            prediction = selector.obtain_contextual_test_sample(self.dataset[test, :], self.context[test, :])
            responses.append(prediction)

        y_hat = np.array(responses)
        actual_mae = mae.MAE().calculate(self.dataset[self.ids_test, :], y_hat)
        return (selector_name, actual_mae, all_chosen_context)

    def test_procedure(self, n_context_choice, context_selectors, n_repetitions=20):
        (nSample, nFeature) = np.shape(self.dataset)
        self.n_context_choice = n_context_choice

        #cast single object to list
        if type(context_selectors) is not dict:
            context_selectors = {"default": context_selectors}

        #initialize statistics collection
        results_by_algorithm = {}
        for name in context_selectors.keys():
            results_by_algorithm[name] = []

        pool = multiprocessing.Pool(len(context_selectors))
        for repetition in range(n_repetitions):
            print(repetition + 1)
            self.__prepare_holdout(nSample)

            # setup RNG
            self.seed += 91
            for selector in context_selectors.values():
                selector.train_method.set_seed(self.seed)

            # parallel execution
            results = pool.map(self._test_selector, context_selectors.items())

            # result gathering
            for current_result in results:
                selector_name, actual_mae, choosen_contexts = current_result[0], current_result[1], current_result[2]
                print(selector_name, actual_mae)
                results_by_algorithm[selector_name].append(actual_mae)

        pprint.pprint(results_by_algorithm)

        # plots the iteration result
        plot(results_by_algorithm)


def plot(results_by_algorithm, y_label="MAE", title="Results"):
    if len(results_by_algorithm) == 1:
        return
    plt.clf()

    # sort the dict keys
    labels = np.array([i for i in results_by_algorithm.keys()])
    sorted_label_ids = np.argsort(labels)

    # statistics
    values = np.array([i for i in results_by_algorithm.values()])
    mean_by_algorithm = np.mean(values, axis=1)[sorted_label_ids]
    std_by_algorithm = np.std(values, axis=1)[sorted_label_ids]

    # t-student
    degrees_of_freedom = 29
    interval = t.interval(0.95, degrees_of_freedom, loc=mean_by_algorithm, scale=std_by_algorithm)
    print(interval)

    # Plot settings
    worst_case = np.min(mean_by_algorithm) - np.std(mean_by_algorithm) # for comparative purposes
    sequence = np.arange(len(labels))
    plt.bar(sequence, mean_by_algorithm - worst_case, bottom=worst_case, align='center', ecolor=(1, 0, 0),
            yerr=(mean_by_algorithm - interval[0], interval[1] - mean_by_algorithm))
    plt.ylabel(y_label)
    plt.xticks(sequence, labels[sorted_label_ids])
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    seed = 66666

    file = "MRMR_data.csv"

    m = pd.read_csv(file)
    n_context_choice = 2
    n_repetitions = 2

    dataset = m.values[:, 0:3]
    context = m.values[:, 7: 19]

    n_user = np.max(dataset[:, 0]) + 1
    n_item = np.max(dataset[:, 1]) + 1

    svd = ContextualSVD.ContextualSVD(n_user, n_item, max_steps=100, n_latent_features=20, mode='item')
    encoder = OneHotEncoder(context, na_value=-1)

    largest_deviation = LargestDeviationContextSelection(copy.deepcopy(svd), encoder)
    random_choice = RandomContextSelection(copy.deepcopy(svd), encoder)
    baseline = AllContextSelection(copy.deepcopy(svd), encoder)
    cramer = CramerLargestDeviation(copy.deepcopy(svd), encoder)

    selectors = {"Largest Deviation": largest_deviation, "Random": random_choice, "All Contexts": baseline,
                 "Cramer Deviation": cramer}

    tf = TestFramework(dataset, context, seed=seed)
    tf.test_procedure(n_context_choice, selectors, n_repetitions=n_repetitions)
