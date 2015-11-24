__author__ = 'gabriel'
import numpy as np

class ContextualSVD():
    def __init__(self, max_steps = 200, n_latent_features = 40, learning_coeficient = 0.001, regularization_coeficient = 0.001, k = 25):
        self.max_steps = max_steps
        self.n_latent_features = n_latent_features
        self.learning_coeficient = learning_coeficient
        self.regularization_coeficient = regularization_coeficient
        self.k = k

    def train(self, dataset, context, n_user, n_item, n_context, max_rating = 5, min_rating = 1):
        self.n_user = n_user
        self.n_item = n_item
        self.n_context = n_context
        self.max_rating = max_rating
        self.min_rating = min_rating

        self._initialize_variables(dataset)
        n_dataset = np.shape(dataset)[0]

        for step_count in range(self.max_steps):
            for i in range(n_dataset):
                self._train_step(dataset[i, 0], dataset[i, 1], dataset[i, 2], context[i, :])

    def _initialize_variables(self, dataset, context):
        self.global_mean = np.mean(dataset[:, 2])
        self._item_mean(dataset)
        self._user_offset(dataset)

        self.item_feature = np.ones(self.n_item, self.n_latent_features) * 0.1
        self.user_feature = np.ones(self.n_user, self.n_latent_features) * 0.1

        self.item_context_matrix = np.zeros(self.n_item, self.n_context)

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
        err = rating - self.predict_rating(user, item)
        self.user_feature[user, :] += self.learning_coeficient * \
                                      (err * self.item_feature[item, :] -
                                       self.regularization_coeficient * self.user_feature[user, :])

        self.item_feature[item, :] += self.learning_coeficient * \
                                      (err * self.user_feature[user, :] -
                                       self.regularization_coeficient * self.item_feature[item, :])

        self.user_offset[user, :]  += self.learning_coeficient * \
                                      (err - self.regularization_coeficient * self.user_offset[user, :])

        nonzero_context = np.nonzero(context)
        self.item_context_matrix[item, nonzero_context]  += self.learning_coeficient * \
                                      (err - self.regularization_coeficient * self.item_context_matrix[item, nonzero_context])

    def predict_rating(self, user_id, item_id, context):
        nonzero_context = np.nonzero(context)
        prediction = self.user_feature[user_id, :] * self.item_feature[item_id, :] + \
                        self.item_mean[item_id][0] + self.user_offset[user_id][0] + \
                        np.sum(self.item_context_matrix[item_id, nonzero_context])
        if prediction > self.max_rating:
            prediction = self.max_rating
        if prediction < self.min_rating:
            prediction = self.min_rating
        return prediction

def one_hot_encoder(categorical_matrix):
    pass
    max_values = np.max(categorical_matrix, axis=1)

if __name__ == "__main__":
    context_columns = [i for i in range(6, 17)]
    dataset = np.loadtxt("", delimiter=",", dtype=int)
