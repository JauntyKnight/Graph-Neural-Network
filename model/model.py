import numpy as np
from typing import Iterable, List, Tuple, Union

from layers import *
from activations import *


class Model:
    def __init__(self, inputs, outputs, name: str = "model") -> None:
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
        for i in range(self.layers_count):
            inputs = self.id_to_layer[i].predict(inputs)

        return inputs

    def predict(self, inputs):
        return self.__forward(inputs)
