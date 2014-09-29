# coding utf-8

import numpy as np
from scipy.stats import norm


class Gaussian:

    def __init__(self, mean=0, variance=1, weight=1):
        self.mean = mean
        self.variance = variance
        self.weight = weight

    def eval_proba(self, x=None):
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
        self.parameters = {'mean': [], 'variance': [], 'weights': []}

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
            self.parameters['variance'] += [gaussian.variance]
            self.parameters['weights'] += [gaussian.weight]

    def update_parameters(self, means=None, variances=None, weights=None):
        self.parameters['means'] = means
        self.parameters['variance'] = variances
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
    K = len(S[0])  # number of features

    model = MixtureModel()
    for n in range(N):
        model.add_Gaussian(
            Gaussian(mean=0, variance=np.identity(K), weight=1 / N))

    # EM loop
    counter = 0
    converged = False
    while not converged:

        proba = []
        means = []
        variances = []
        weights = []
        for sample in S:

            proba += [[]]

            for k in range(N):

                # compute  probability of having sample s in mixture k
                proba[-1] += [
                    model.gaussians[k].eval_proba(sample) / model.eval_proba(sample)]

        for k in range(N):

            # compute weight of mixture k
            weights += [sum([proba[i][k] for i in range(len(S))] / N)]

            # compute mean of mixture k
            means += [sum([proba[i][k] * S[i] for i in range(len(S))]
                          ) / sum([proba[i][k] for i in range(len(S))])]

            # compute variance of mixture k
            variances += [sum([proba[i][k] * (S[i] - means[-1]) * (S[i] - means[-1]).transpose() for i in range(len(S))]
                              ) / sum([proba[i][k] for i in range(len(S))])]
            # unfinished business

        # update parameters
        model.update_parameters(means, variances, weights)

        # Convergence check
        counter += 1
        converged = counter >= max_iter

    return(model)
