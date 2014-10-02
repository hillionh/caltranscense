# coding utf-8

import numpy as np
from scipy.stats import norm


class Gaussian:

    def __init__(self, mean=0, variance=1, weight=1):
        self.mean = mean
        self.variance = variance
        self.weight = weight

    def eval_proba(self, x=None):
        #         print("weight", self.weight, "x", x, "mean",
        #               self.mean, "var", np.matrix(self.variance))
        #         print("re", self.weight * norm.pdf(x, self.mean, self.variance))
        return(self.weight * norm.pdf(x, self.mean, self.variance))

    def eval_post_proba(self, x=None, model=None):
        return(self.eval_proba(x) / model.eval_proba(x))

    def update_parameters(self, mean=None, variance=None, weight=None):
        if mean != None:
            self.mean = mean
        if variance != None:
            self.variance = variance
        if weight != None:
            self.weight = weight


class MixtureModel:

    def __init__(self):
        self.gaussians = []
        self.parameters = {'means': [], 'variances': [], 'weights': []}

    def add_Gaussian(self, gaussian):
        self.gaussians += [gaussian]

    def eval_proba(self, x):
        res = 0
        for gaussian in self.gaussians:
            res += gaussian.eval_proba(x)
        return(res)

    def set_parameters(self):
        for gaussian in self.gaussians:
            self.parameters['means'] += [gaussian.mean]
            self.parameters['variances'] += [gaussian.variance]
            self.parameters['weights'] += [gaussian.weight]

    def update_parameters(self, means=None, variances=None, weights=None):
        self.parameters['means'] = means
        self.parameters['variances'] = variances
        self.parameters['weights'] = weights

    def get_parameters(self):
        return(self.parameters)


def EM_algorithm(max_iter=100, S=None):
    '''N is the number of speakers, K is the number of features per signal, S is the set of samples'''

    # check the validity of the samples set
    if S == None or S == []:
        print("Error : set of samples empty")
        return(None)

    # Initializations
    N = len(S)  # number of speakers
    K = len(S[0])  # number of samples
    D = len(S[0][0])  # number of features, ie vectors dimension

#     print("N=" + str(N), "K=" + str(K))

    model = MixtureModel()
    for n in range(N):
        model.add_Gaussian(
            Gaussian(mean=0, variance=np.identity(D), weight=1. / N))
    model.set_parameters()
#     print(model.parameters)

    # EM loop
    counter = 0
    converged = False
    while not converged:
        #print('test 1 ok')
        proba = []
        means = []
        variances = []
        weights = []
        speaker_id = 0

        for speaker in S:

            proba += [[]]
            k = 0

            for sample in speaker:

                # compute  probability of having sample s in mixture k
                proba[-1] += [float(
                    model.gaussians[speaker_id].eval_proba(sample) / model.eval_proba(sample))]
                k += 1

            speaker_id += 1

        for k in range(N):

            # compute weight of mixture k (N-list of 1-Dim)
            weights += [sum([proba[k][i]
                             for i in range(K)]) / K]
#             print(weights[-1])

            # compute mean of mixture k (N-list of N-Dim vector)
#             print('1',proba[k][i])
#             print('2', S[k][i][0])
#             print('3',sum([np.dot(proba[k][i], S[k][i][0]) for i in range(K)]))
#             print('4', 1. / sum([proba[k][i] for i in range(K)]))
            means += [sum([np.dot(proba[k][i], S[k][i][d]) for i in range(
                K)]) * 1. / sum([proba[k][i] for i in range(K)]) for d in range(D)]
#             print(means[-1])

            # compute variance of mixture k (N-list of (N*N)-Dim matrix)
#             print(proba[k][i])
#             print(S[k][i])
#             print(means[-1])
#             print('sum', sum([proba[k][i] * np.dot((S[k][i] - means[-1]), (S[k][i] - means[-1]).transpose()) for i in range(K)]))
#             print('sum2', sum([proba[k][i] for i in range(K)]))
            variances += [sum([proba[k][i] * np.dot((S[k][i] - means[-1]), (S[k][i] - means[-1]).transpose()) for i in range(K)]
                                         ) / sum([proba[k][i] for i in range(K)])]
#             print(variances[-1])
            # unfinished business

#         print(weights)
#         print(means)
#         print(variances)

        # update parameters
        model.update_parameters(means, variances, weights)

        # Convergence check
        counter += 1
        converged = counter >= max_iter

    return(model)
