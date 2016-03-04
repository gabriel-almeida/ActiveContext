import numpy as np
import ContextualSVD
import mae
from OneHotEncoder import OneHotEncoder
from Selectors import LargestDeviationContextSelection, AllContextSelection,\
    CramerLargestDeviation, RandomContextSelection, LogCramerLargestDeviation, \
    TildeCramerLargestDeviation, LogTildeCramerLargestDeviation
import copy
import multiprocessing
import random
import json
import Plot

__author__ = 'gabriel'


class TestFramework():
    def __init__(self, dataset, context, train_ratio=0.5, candidate_ratio=0.25, user_column=0, item_column=1):
        self.train_ratio = train_ratio
        self.candidate_ratio = candidate_ratio
        self.dataset = dataset
        self.context = context

    def __prepare_holdout(self, n_sample, seed):
        random.seed(seed)
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
        #print('initial train', selector_name, selector.train_method.complete_hash())

        for candidate in self.ids_candidate:
            chosen_contexts = selector.choose_contexts(self.dataset[candidate, :])
            selector.obtain_contextual_train_sample(self.dataset[candidate, :], self.context[candidate, chosen_contexts], chosen_contexts)

        responses = []
        for test in self.ids_test:
            prediction = selector.obtain_contextual_test_sample(self.dataset[test, :], self.context[test, :])
            responses.append(prediction)

        y_hat = np.array(responses)
        actual_mae = mae.MAE().calculate(self.dataset[self.ids_test, :], y_hat)
        return (selector_name, actual_mae)

    def test_procedure(self, list_n_context_choice, context_selectors, n_repetitions=20, results_file="results-test.json", seed=None):
        (nSample, nFeature) = np.shape(self.dataset)

        #cast single object to list
        if type(context_selectors) is not dict:
            context_selectors = {"default": context_selectors}
        if type(list_n_context_choice) is not list:
            list_n_context_choice = [list_n_context_choice]

        #initialize statistics collection
        results_by_algorithm = {}
        for n_context_choice in list_n_context_choice:
            results_by_algorithm[n_context_choice] = {}
            for name in context_selectors.keys():
                results_by_algorithm[n_context_choice][name] = []

        pool = multiprocessing.Pool(len(context_selectors))
        for repetition in range(n_repetitions):
            # experiments
            for n_context_choice in list_n_context_choice:
                print((repetition + 1), "repetition with", n_context_choice, "contexts")

                # set up rng for the current run
                self.__prepare_holdout(nSample, seed)
                for selector in context_selectors.values():
                    selector.train_method.set_seed(seed)

                # parallel execution
                self.n_context_choice = n_context_choice
                results = pool.map(self._test_selector, context_selectors.items())

                # result gathering
                order = np.argsort([r[1] for r in results])
                for i in range(len(results)):
                    # performance visualization
                    selector_name, actual_mae = results[order[i]]
                    if i == 0:
                        dif = 0
                    else:
                        dif = results[order[i-1]][1] - actual_mae
                    print(selector_name, actual_mae, "(", dif, ")")

                    results_by_algorithm[n_context_choice][selector_name].append(actual_mae)

                # save current results
                with open(results_file, 'w') as f:
                    f.write(json.dumps(results_by_algorithm))

            # setup RNG for the next iteration
            if seed is not None:
                seed += 17

        Plot.plot(results_by_algorithm)

if __name__ == "__main__":
    import pandas as pd
    import time
    milis = int(round(time.time()))
    seed = milis#1457032230
    print('seed=', seed)

    file = "MRMR_data.csv"

    m = pd.read_csv(file)
    n_context_choice = [1, 2, 3, 4, 5, 6]
    n_repetitions = 30

    dataset = m.values[:, 0:3]
    context = m.values[:, 7: 19]

    n_user = np.max(dataset[:, 0]) + 1
    n_item = np.max(dataset[:, 1]) + 1

    svd = ContextualSVD.ContextualSVD(n_user, n_item, max_steps=200, n_latent_features=30, mode='item')
    encoder = OneHotEncoder(context, na_value=-1)

    largest_deviation = LargestDeviationContextSelection(copy.deepcopy(svd), encoder)
    random_choice = RandomContextSelection(copy.deepcopy(svd), encoder)
    baseline = AllContextSelection(copy.deepcopy(svd), encoder)
    cramer = CramerLargestDeviation(copy.deepcopy(svd), encoder)
    log_cramer = LogCramerLargestDeviation(copy.deepcopy(svd), encoder)
    tilde_cramer = TildeCramerLargestDeviation(copy.deepcopy(svd), encoder)
    log_tilde_cramer = LogTildeCramerLargestDeviation(copy.deepcopy(svd), encoder)

    selectors = {"Largest Deviation": largest_deviation, "Random": random_choice, "All Contexts": baseline,
                 "Cramer Deviation": cramer, "Log Cramer": log_cramer, "Tilde Cramer": tilde_cramer,
                 "Log Tilde Cramer": log_tilde_cramer}

    tf = TestFramework(dataset, context)
    tf.test_procedure(n_context_choice, selectors, n_repetitions = n_repetitions, seed = seed, results_file="results-cramer-tilde.json")
