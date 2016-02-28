__author__ = 'gabriel'
import numpy as np
import CramersV
import abc

class AbstractContextSelection():
    __metaclass__ = abc.ABCMeta

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

            #print('train_context - ',hashlib.sha1(train_context).hexdigest())

            #Free the buffers
            self.train_buffer = []
            self.context_buffer = []
            #retrain the model
            self.obtain_initial_train(train_set, train_context, self.n_context_choice)

        context = np.reshape(context, (1, self.encoder.n_contextual_factor))
        return self.train_method.predict_rating(sample[0], sample[1], self.encoder.predict(context))

    @abc.abstractmethod
    def choose_contexts(self, sample):
        pass


class RandomContextSelection(AbstractContextSelection):
    def __init__(self, train_method, encoder):
        AbstractContextSelection.__init__(self, train_method, encoder)

    def choose_contexts(self, sample):
        return self.train_method.random_state.randint(0, self.encoder.n_contextual_factor)


class AllContextSelection(AbstractContextSelection):
    def __init__(self, train_method, encoder):
        AbstractContextSelection.__init__(self, train_method, encoder)

    def choose_contexts(self, sample):
        return np.array([i for i in range(self.encoder.n_contextual_factor)])


class LargestDeviationContextSelection(AbstractContextSelection):
    def __init__(self, train_method, encoder):
        AbstractContextSelection.__init__(self, train_method, encoder)

    def obtain_initial_train(self, train_set, context, n_context_choice):
        AbstractContextSelection.obtain_initial_train(self, train_set, context, n_context_choice)
        self.normalized_frequency = np.mean(self.encoder.predict(context), axis=0)

    def choose_contexts(self, sample):
        n_contextual_conditions = self.encoder.n_contextual_condition
        n_contextual_factors = self.encoder.n_contextual_factor

        all_contexts = np.eye(n_contextual_conditions)
        without_context = np.zeros((1, n_contextual_conditions))
        repeated_sample = np.tile(sample, (n_contextual_conditions, 1))

        prediction_with_contexts = self.train_method.predict_dataset(repeated_sample, all_contexts)
        prediction_without_context = self.train_method.predict_rating(sample[0], sample[1], without_context)
        deviation = np.abs(prediction_with_contexts - prediction_without_context)

        contextual_condition_weight = np.multiply(self.normalized_frequency, deviation)
        contextual_factor_weight = np.zeros((1, n_contextual_factors))

        context_index = self.encoder.contextual_factor_index.tolist() + [self.encoder.n_contextual_condition]
        for i in range(n_contextual_factors - 1):
            contextual_factor_weight[0, i] = np.mean(contextual_condition_weight[context_index[i]:context_index[i+1]])

        #get the last n elements
        context_choice = np.argsort(contextual_factor_weight)[0, n_contextual_factors - self.n_context_choice:]
        return context_choice


class CramerLargestDeviation(LargestDeviationContextSelection):
    def __init__(self, train_method, encoder):
        LargestDeviationContextSelection.__init__(self, train_method, encoder)

    def obtain_initial_train(self, train_set, context, n_context_choice):
        LargestDeviationContextSelection.obtain_initial_train(self, train_set, context, n_context_choice)
        self.__cramer_matrix()

    def __cramer_matrix(self):
        n_context = np.shape(self.train_context)[1]
        cramer = np.zeros((n_context, n_context))
        for i in range(n_context):
            for j in range(n_context):
                cram = CramersV.cramersV(self.train_context[:, i], self.train_context[:, j])
                cramer[i, j] = cram
        self.cramer_matrix = cramer


    def choose_contexts(self, sample):
        n_contextual_conditions = self.encoder.n_contextual_condition
        n_contextual_factors = self.encoder.n_contextual_factor

        all_contexts = np.eye(n_contextual_conditions)
        without_context = np.zeros((1, n_contextual_conditions))
        repeated_sample = np.tile(sample, (n_contextual_conditions, 1))

        prediction_with_contexts = self.train_method.predict_dataset(repeated_sample, all_contexts)
        prediction_without_context = self.train_method.predict_rating(sample[0], sample[1], without_context)
        deviation = np.abs(prediction_with_contexts - prediction_without_context)

        contextual_condition_weight = np.multiply(self.normalized_frequency, deviation)

        contextual_factor_weight = np.zeros((1, n_contextual_factors)) + 0.1

        aggregated_deviation = np.zeros((1, n_contextual_factors))
        aggregated_frequency = np.zeros((1, n_contextual_factors))

        context_index = self.encoder.contextual_factor_index.tolist() + [self.encoder.n_contextual_condition]

        for i in range(n_contextual_factors - 1):
            contextual_factor_weight[0, i] = np.mean(contextual_condition_weight[context_index[i]:context_index[i+1]])

            aggregated_deviation[0, i] = np.mean(deviation[context_index[i]:context_index[i+1]])
            aggregated_frequency[0, i] = np.mean(self.normalized_frequency[context_index[i]:context_index[i+1]])

        context_choice = [np.argmax(contextual_factor_weight)]
        possible_choices = [i for i in range(n_contextual_factors) if i not in context_choice]

        for i in range(self.n_context_choice - 1):
            score = np.ones(len(possible_choices))
            for past_choice in context_choice:
                cram = self.cramer_matrix[possible_choices, past_choice]
                context = contextual_factor_weight[0,possible_choices]
                a = np.divide(context, cram)
                score *= a
            chosen_context = np.argmax(score)
            index_context = possible_choices[chosen_context]
            context_choice.append(index_context)
            possible_choices.remove(index_context)


        #get the last n elements
        context_choice = np.argsort(contextual_factor_weight)[0, n_contextual_factors - self.n_context_choice:]
        return context_choice
