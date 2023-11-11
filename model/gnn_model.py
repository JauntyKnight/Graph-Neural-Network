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


from model import Model

import numpy as np


class TransitionFunction(Model):
    def __init__(
        self, inputs, outputs, max_iter=50, tol=1e-5, verbose=False, name="transition_function"
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.name = name
        super().__init__(inputs, outputs, name)

    def verbose_print(self, message):
        if self.verbose:
            print(message)

    def __forward(self, state):
        return super().predict(state)

    # iteratively updates the state until convergence or max_iter is reached
    def predict(self, inputs):
        state = inputs
        for step in range(self.max_iter):
            new_state = self.__forward(state)
            if np.linalg.norm(new_state - state) < self.tol:
                break
            state = new_state

        if step == self.max_iter - 1:
            self.verbose_print("Maximum number of iterations reached")
        else:
            self.verbose_print(f"Converged to a fixed point after {step} steps")

        return state


class OutputFunction(Model):
    def __init__(self):
        pass

    def __call__(self, state):
        pass


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
