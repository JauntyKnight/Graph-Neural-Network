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




import numpy as np

class Model:
    def __init__(self):
        pass

    def __call__(self, state):
        pass

    def fit(self, state, target):
        pass

    def predict(self, state):
        pass

class Layer:
    def __init__(self):
        pass

    def __call__(self, state):
        pass

    def fit(self, state, target):
        pass

    def predict(self, state):
        pass

class Input(Layer):
    def __init__()

class TransitionFunction(Model):
    def __init__(self):
        pass

    def __call__(self, state, action):
        pass


class GNN:
    def __init__(self):
        
