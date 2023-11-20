import numpy as np
from typing import Iterable, List, Tuple, Union

from layers import *


class Model:
    def __init__(self, inputs, outputs, name: str = "model", verbose: bool = False) -> None:
        if inputs.output_shape != outputs.output_shape:
            raise ValueError(
                f"Input and output shapes do not match: {inputs.output_shape} != {outputs.output_shape}"
            )

        self.inputs = inputs
        self.outputs = outputs
        self.id_to_layer = {}
        self.layers_count = 0
        self.__topological_sort(inputs)
        self.name = name
        self.verbose = verbose
        self.outputs = None

    def verbose_print(self, message):
        if self.verbose:
            print(message)

    # sorts the layers in topological order
    def __topological_sort(self, inputs):
        visited = set()

        def dfs(layer):
            visited.add(layer)

            self.id_to_layer[self.layers_count] = layer
            self.layers_count += 1

            for child in layer.child_layers:
                if child not in visited:
                    dfs(child)

        dfs(inputs)

    def __repr__(self) -> str:
        r: str = f"{self.name}:\n"

        for i in range(self.layers_count):
            r += f"{self.id_to_layer[i]}\n"

        return r

    def __forward(self, inputs):
        # TODO: move away from a feed-forward approach
        # and use the topological order to compute the outputs
        for i in range(self.layers_count):
            inputs = self.id_to_layer[i].predict(inputs)

        return inputs

    def predict(self, inputs):
        self.outputs = self.__forward(inputs)
        return self.outputs

    def _apply_gradients(self, gradients):
        for i in range(self.layers_count, -1, -1):
            gradients = self.id_to_layer[i].apply_gradients(gradients)

        return gradients


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
        super().__init__(inputs, outputs, name, verbose)

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
    def __init__(self, inputs, outputs, name="output_function", verbose=False):
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self.verbose = verbose
        super().__init__(inputs, outputs, name, verbose)

    def predict(self, inputs):
        return super().predict(inputs)

    def _apply_gradients(self, gradients):
        super()._apply_gradients(gradients)
