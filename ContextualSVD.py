__author__ = 'gabriel'
import numpy as np
import random
import matplotlib.pyplot as plt


class PrintEpoch():
    def __init__(self):
        self.i = 0
    def callback(self, _):
        self.i += 1
        print(self.i)


class ContextualSVD():
    def __init__(self, n_user, n_item, max_steps=200, n_latent_features=40, learning_coeficient=0.001,
                 regularization_coeficient=0.001, k=10, mode = 'item', max_rating=5, min_rating=1):
        self.max_steps = max_steps
        self.n_user = n_user
        self.n_item = n_item
        self.max_rating = max_rating
        self.min_rating = min_rating
        self.n_latent_features = n_latent_features
        self.learning_coeficient = learning_coeficient
        self.regularization_coeficient = regularization_coeficient
        self.k = k
        self.mode = mode


    def train(self, dataset, context, callback = None, callback_interval=10):

        if np.shape(dataset)[0] != np.shape(context)[0]:
            raise Exception("Dataset and context must have the same number of lines")
        if np.shape(dataset)[1] != 3:
            raise Exception("Dataset must have only 3 columns: user_id, item_id and rating")

        self.n_context = np.shape(context)[1]


        self._initialize_variables(dataset, context)
        n_dataset = np.shape(dataset)[0]

        if callback is not None:
            callback(self)

        train_order = [i for i in range(n_dataset)]
        for step_count in range(self.max_steps):
            #random.shuffle(train_order)
            for i in train_order:
                self._train_step(dataset[i, 0], dataset[i, 1], dataset[i, 2], context[i, :])

            if step_count % callback_interval == 0 and step_count != 0:
                if callback is not None:
                    callback(self)
                else:
                    pass
                    #print(step_count)

        if callback is not None:
            callback(self)

    def _initialize_variables(self, dataset, context):
        self.global_mean = np.mean(dataset[:, 2])
        self._item_mean(dataset)
        self._user_offset(dataset)

        self.item_feature = np.ones((self.n_item, self.n_latent_features)) * 0.1
        self.user_feature = np.ones((self.n_user, self.n_latent_features)) * 0.1

        if self.mode == 'item':
            self.item_context_matrix = np.zeros((self.n_item, self.n_context))
        if self.mode == 'context':
            self.item_context_matrix = np.zeros((1, self.n_context))


    def _item_mean(self, dataset):
        n_ratings = np.shape(dataset)[0]
        items_mean = np.ones(self.n_item) * self.global_mean * self.k
        items_count = np.zeros(self.n_item)

        for i in range(n_ratings):
            item = dataset[i, 1]
            rating = dataset[i, 2]
            items_mean[item] += rating
            items_count[item] += 1

        self.item_mean = items_mean / (self.k + items_count)

    def _user_offset(self, dataset):
        n_ratings = np.shape(dataset)[0]
        users_offset = np.zeros(self.n_user)
        user_count = np.zeros(self.n_user)

        for i in range(n_ratings):
            user = dataset[i, 0]
            item = dataset[i, 1]
            rating = dataset[i, 2]
            users_offset[user] += rating - self.item_mean[item]
            user_count[user] += 1

        self.user_offset = users_offset / (self.k + user_count)

    def _train_step(self, user, item, rating, context):
        err = rating - self.predict_rating(user, item, context)

        self.user_feature[user, :] += self.learning_coeficient * \
                                      (err * self.item_feature[item, :] -
                                       self.regularization_coeficient * self.user_feature[user, :])

        self.item_feature[item, :] += self.learning_coeficient * \
                                      (err * self.user_feature[user, :] -
                                       self.regularization_coeficient * self.item_feature[item, :])


        self.user_offset[user] += self.learning_coeficient * \
                                  (err - self.regularization_coeficient * self.user_offset[user])

        nonzero_context = np.nonzero(context)
        if self.mode == 'item':
            self.item_context_matrix[item, nonzero_context] += self.learning_coeficient * \
                                                           (err - self.regularization_coeficient *
                                                            self.item_context_matrix[item, nonzero_context])
        if self.mode == 'context':
            self.item_context_matrix[0, nonzero_context] += self.learning_coeficient * \
                                                           (err - self.regularization_coeficient *
                                                            self.item_context_matrix[0, nonzero_context])

    def predict_rating(self, user_id, item_id, context):
        nonzero_context = np.nonzero(context)
        if self.mode == 'item':
            prediction = np.dot(self.user_feature[user_id, :], self.item_feature[item_id, :]) + \
                     self.item_mean[item_id] + self.user_offset[user_id] + \
                     np.sum(self.item_context_matrix[item_id, nonzero_context])

        elif self.mode == 'context':
            prediction = np.dot(self.user_feature[user_id, :], self.item_feature[item_id, :]) + \
                     self.item_mean[item_id] + self.user_offset[user_id] + \
                     np.sum(self.item_context_matrix[0, nonzero_context])

        else:
            prediction = np.dot(self.user_feature[user_id, :], self.item_feature[item_id, :]) + \
                     self.item_mean[item_id] + self.user_offset[user_id]

        if prediction > self.max_rating:
            prediction = self.max_rating
        if prediction < self.min_rating:
            prediction = self.min_rating
        return prediction

    def predict_dataset(self, dataset, context):
        n_dataset = np.shape(dataset)[0]
        # if n_columns != 2:
        #    print("ignoring columns greater than 2")

        y_hat = np.zeros(n_dataset)
        for i in range(n_dataset):
            y_hat[i] = self.predict_rating(dataset[i, 0], dataset[i, 1], context[i, :])

        return y_hat


def one_hot_encoder(categorical_matrix, na_value=-1):
    '''
    Returns the one-hot-encoded feature matrix of a
    categorical matrix. Assumes a matrix with categorical variables,
    indicated by a number from 1 to N on
    every column, where N is the number of possible categories.
    Resulting matrix  will have the same amount of lines and
    the sum of all N's as the number of columns.
    '''
    max_values = np.max(categorical_matrix, axis=0)
    n_features = sum(max_values)
    n_line, n_col = np.shape(categorical_matrix)

    begin_index = np.cumsum(max_values)
    begin_index[-1] = 0
    begin_index = np.roll(begin_index, 1)

    result = np.zeros((n_line, n_features))
    for i in range(n_line):
        not_na_columns = categorical_matrix[i, :] != na_value
        result[i, categorical_matrix[i, not_na_columns] + begin_index[not_na_columns] - 1] = 1

    return result


def mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def holdout(n_samples, train_ratio=0.7):
    ids = [i for i in range(n_samples)]
    random.shuffle(ids)
    n = round(n_samples * train_ratio)
    return ids[:n], ids[n:]



class LearningCurve():
    def __init__(self, dataset, context, train, test, eval_func=mae, aggregation = None):
        self.results = []
        self._dataset = dataset
        self._context = context
        self._train = train
        self._test = test
        self._eval_func = eval_func
        self._i = 0
        self._aggregation = aggregation
        if aggregation == 'user':
            self._aggregation_train = self._aggregation_map(dataset[train, :], 0)
            self._aggregation_test = self._aggregation_map(dataset[test, :], 0)
        elif aggregation == 'item':
            self._aggregation_train = self._aggregation_map(dataset[train, :], 1)
            self._aggregation_test  = self._aggregation_map(dataset[test, :], 1)

        elif aggregation != None:
            print("Not recognized option for aggregation:\"", aggregation, "\". Assuming None.")

    def _aggregation_map(self, dataset, key_column):
        agg_map = {}
        n_dataset = np.shape(dataset)[0]
        for i in range(n_dataset):
            key = dataset[i, key_column]
            if key not in agg_map:
                agg_map[key] = []
            agg_map[key].append(i)
        return agg_map

    def _eval(self, objSVD, eval_set):
        y_hat = objSVD.predict_dataset(self._dataset[eval_set, :], self._context[eval_set, :])
        y = self._dataset[eval_set, 2]
        return self._eval_func(y_hat, y)

    def callback_function(self, objSVD):
        self._i += 1

        if self._aggregation in ['user', 'item']:
            agg_train = np.array([[self._eval(objSVD, eval_set), len(eval_set)] for (key, eval_set) in self._aggregation_train.items()])
            self.agg_test = np.array([[self._eval(objSVD, eval_set), len(eval_set)] for (key, eval_set) in self._aggregation_test.items()])

            eval_train = np.average(agg_train[:, 0], weights=agg_train[:, 1])
            eval_test = np.average(self.agg_test[:, 0], weights=self.agg_test[:, 1])

        else:
            eval_train = self._eval(objSVD, self._train)
            eval_test = self._eval(objSVD, self._test)

        self.results.append([eval_train, eval_test])
        print(self._i, ") Train: ", eval_train, "\tTest: ", eval_test)

    def plot(self):
        results = np.array(self.results)
        plt.plot(results[:, 0], label="Train")
        plt.plot(results[:, 1], label="Test")

        plt.title("Learning curve")

        y_label = ""
        if self._aggregation in ['user', 'item']:
            y_label += self._aggregation + "-"
        eval_name = self._eval_func.__name__

        y_label += eval_name
        plt.ylabel(y_label)

        plt.xlabel("Iteration")
        plt.legend(loc='lower left')

        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    import pandas as pd
    random.seed(101)

    file = "/home/gabriel/Dropbox/Mestrado/Sistemas de Recomendação/Datasets/ldos_comoda.csv"

    m = pd.read_csv(file, header=None)

    dataset = m.values[:, 0:3]
    contexts = m.values[:, 7: 18+1]

    onehot_context = one_hot_encoder(contexts)
    svd = ContextualSVD(max_steps=700, mode='None')

    train, test = holdout(dataset.shape[0])
    learning_curve = LearningCurve(dataset, onehot_context, train, test, aggregation=None)
    svd.train(dataset[train, :], onehot_context[train, :], 268 + 1, 4381 + 1, callback=learning_curve.callback_function)
    learning_curve.plot()
