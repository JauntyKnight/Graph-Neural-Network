###
# Graph Neural Network Model, as described in the paper The Graph Neural Network Model
# by Scarselli et al. (2009)
# https://ieeexplore.ieee.org/document/4700287
#
#
# Structure of the GNN:
# - Transition Function (Model)
# - Output Function (Model)
# - GNN which combines the two models
#
# The GNN is trained using the backpropagation algorithm.
# REQUIREMENTS:
# - the models should be able to be trained using the backpropagation algorithm
# - the models for the transition and output function should be able to be trained
# - the models for the transition and output function should be created in a functional way, i.e. they should be callable
###


from networks import Model

import numpy as np


class GNN(Model):
    def __init__(self, transition_function, output_function):
        self.transition_function = transition_function
        self.output_function = output_function

    def __call__(self, state, action):
        return self.output_function(self.transition_function(state, action))

    def fit(self, state, target):
        pass

    def predict(self, state):
        pass

    def save(self, path):
        pass
