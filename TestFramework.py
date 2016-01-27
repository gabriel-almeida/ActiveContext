__author__ = 'gabriel'
import random
import numpy as np
import ContextualSVD
import matplotlib.pyplot as plt
import mae
from OneHotEncoder import OneHotEncoder
from Selectors import LargestDeviationContextSelection, AllContextSelection,\
    CramerLargestDeviation, RandomContextSelection


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

    def test_procedure(self, n_context_choice, context_selectors, n_repetitions=20):
        (nSample, nFeature) = np.shape(self.dataset)

        arq = open("debug.csv", 'w')

        #cast single object to list
        if type(context_selectors) is not dict:
            context_selectors = {"default": context_selectors}

        #initialize statistics collection
        results_by_algorithm = {}
        for name in context_selectors.keys():
            results_by_algorithm[name] = []

        for repetition in range(n_repetitions):
            print(repetition + 1)
            self.__prepare_holdout(nSample)
            for selector_name, selector in context_selectors.items():
                selector.obtain_initial_train(self.dataset[self.ids_train, :], self.context[self.ids_train, :], n_context_choice)

                #print('inicial - ',complete_hash(selector.train_method))

                arq.write(selector_name)
                for candidate in self.ids_candidate:
                    chosen_contexts = selector.choose_contexts(self.dataset[candidate, :])
                    selector.obtain_contextual_train_sample(self.dataset[candidate, :], self.context[candidate, chosen_contexts], chosen_contexts)
                    arq.write(str(chosen_contexts))
                    arq.write(' ')
                arq.write('\n')

                responses = []
                for test in self.ids_test:
                    # chosen_contexts = selector.choose_contexts(self.dataset[test, :])
                    # prediction = selector.obtain_contextual_test_sample(self.dataset[test, :], self.context[test, chosen_contexts])
                    prediction = selector.obtain_contextual_test_sample(self.dataset[test, :], self.context[test, :])
                    responses.append(prediction)

                y_hat = np.array(responses)
                actual_mae = mae.MAE().calculate(dataset[self.ids_test, :], y_hat)

                print(selector_name, actual_mae)
                results_by_algorithm[selector_name].append(actual_mae)

        values = [i for i in results_by_algorithm.values()]
        array_values = np.array(values)

        mean = np.mean(array_values, axis=1)

        labels = [i for i in results_by_algorithm.keys()]
        labels = np.array(labels)

        sorted_args = np.argsort(labels)
        plt.plot(mean[sorted_args])

        plt.ylabel("MAE")
        plt.xticks(np.arange(len(labels)), labels[sorted_args])
        plt.title("Results")
        plt.show()

if __name__ == "__main__":
    import pandas as pd
    random.seed(100)

    #file = "/home/gabriel/Dropbox/Mestrado/Sistemas de Recomendação/Datasets/ldos_comoda.csv"
    file = "MRMR_data.csv"

    m = pd.read_csv(file)
    n_context_choice = 3
    n_repetitions = 10
    dataset = m.values[:, 0:3]
    context = m.values[:, 7: 19]

    n_user = np.max(dataset[:, 0]) + 1
    n_item = np.max(dataset[:, 1]) + 1

    svd = ContextualSVD.ContextualSVD(n_user, n_item, max_steps=100, n_latent_features=20, mode='item')
    encoder = OneHotEncoder(context, na_value=-1)

    largest_deviation = LargestDeviationContextSelection(svd, encoder)
    random_choice = RandomContextSelection(svd, encoder)
    baseline = AllContextSelection(svd, encoder)
    selectors = {"Largest Deviation": largest_deviation, "Random": random_choice, "All Contexts": baseline,
                 "Cramer Deviation": CramerLargestDeviation(svd, encoder)}

    tf = TestFramework(dataset, context)
    tf.test_procedure(n_context_choice, selectors, n_repetitions = n_repetitions)