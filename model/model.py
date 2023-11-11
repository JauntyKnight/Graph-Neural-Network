import numpy as np
from typing import Iterable, List, Tuple, Union

from layers import *
from activations import *


class Model:
    def __init__(self, inputs, outputs, name: str = "model") -> None:
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
