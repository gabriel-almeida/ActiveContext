__author__ = 'gabriel'
import numpy as np
import random

class ContextualSVD():
    def __init__(self, max_steps = 200, n_latent_features = 40, learning_coeficient = 0.001, regularization_coeficient = 0.001, k = 25):
        self.max_steps = max_steps
        self.n_latent_features = n_latent_features
        self.learning_coeficient = learning_coeficient
        self.regularization_coeficient = regularization_coeficient
        self.k = k

    def train(self, dataset, context, n_user, n_item, max_rating = 5, min_rating = 1):
        if np.shape(dataset)[0] != np.shape(context)[0]:
            raise Exception("Dataset and context must have the same number of lines")
        if np.shape(dataset)[1] != 3:
            raise Exception("Dataset must have only 3 columns: user_id, item_id and rating")

        self.n_user = n_user
        self.n_item = n_item
        self.n_context = np.shape(context)[1]
        self.max_rating = max_rating
        self.min_rating = min_rating

        self._initialize_variables(dataset, context)
        n_dataset = np.shape(dataset)[0]

        for step_count in range(self.max_steps):
            for i in range(n_dataset):
                self._train_step(dataset[i, 0], dataset[i, 1], dataset[i, 2], context[i, :])
            step_count % 10 == 0 and print(step_count)

    def _initialize_variables(self, dataset, context):
        self.global_mean = np.mean(dataset[:, 2])
        self._item_mean(dataset)
        self._user_offset(dataset)

        self.item_feature = np.ones((self.n_item, self.n_latent_features)) * 0.1
        self.user_feature = np.ones((self.n_user, self.n_latent_features)) * 0.1

        self.item_context_matrix = np.zeros((self.n_item, self.n_context))

    def _item_mean(self, dataset):
        n_ratings = np.shape(dataset)[0]
        items_mean = np.ones(self.n_item) * self.global_mean * self.k
        items_count = np.zeros(self.n_item)

        for i in range(n_ratings):
            item = dataset[i, 1]
            rating = dataset[i, 2]
            items_mean[item] += rating
            items_count[item] += 1

        self.item_mean =  items_mean / (self.k + items_count)

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
        self.item_context_matrix[item, nonzero_context]  += self.learning_coeficient * \
                                      (err - self.regularization_coeficient * self.item_context_matrix[item, nonzero_context])

    def predict_rating(self, user_id, item_id, context):
        nonzero_context = np.nonzero(context)
        prediction = np.dot(self.user_feature[user_id, :], self.item_feature[item_id, :]) + \
                        self.item_mean[item_id] + self.user_offset[user_id] + \
                        np.sum(self.item_context_matrix[item_id, nonzero_context])

        if prediction > self.max_rating:
            prediction = self.max_rating
        if prediction < self.min_rating:
            prediction = self.min_rating
        return prediction

    def predict_dataset(self, dataset, context):
        n_dataset, n_columns = np.shape(dataset)
        if n_columns != 2:
            print("ignoring columns greater than 2")

        y_hat = np.zeros(n_dataset)
        for i in range(n_dataset):
            y_hat[i] = self.predict_rating(dataset[i, 0], dataset[i, 1], context[i, :])

        return y_hat

def one_hot_encoder(categorical_matrix, na_value = -1):
    '''
    Returns the one-hot-encoded feature matrix of a
    categorical matrix. Assumes a matrix with categorical variables,
    indicated by a number from 1 to N on
    every column, where N is the number of possible categories.
    Resulting matrix  will have the same amount of lines and
    the sum of all N's as the number of columns.
    '''
    max_values = np.max(categorical_matrix, axis=0)
    print(max_values)
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

def holdout(n_samples, train_ratio = 0.7):
    ids = [i for i in range(n_samples)]
    random.shuffle(ids)
    n = round(n_samples * train_ratio)
    return ids[:n], ids[n:]

if __name__ == "__main__":
    import pandas as pd
    file = ""
    m = pd.read_csv(file, header=None )

    dataset = m.values[:, 0:3]
    contexts = m.values[:, 7: 18]

    onehot_context = one_hot_encoder(contexts)
    svd = ContextualSVD()

    train, test = holdout(dataset.shape[0])

    print("training")
    svd.train(dataset[train, :], onehot_context[train, :], 268 + 1,4381 + 1)

    y_hat = svd.predict_dataset(dataset[test, :], onehot_context[test, :])
    y = dataset[test, 2]

    print(mae(y, y_hat))

    print("finished")

