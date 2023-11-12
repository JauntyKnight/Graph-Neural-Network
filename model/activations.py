import numpy as np
from typing import Iterable, List, Tuple, Union, Callable

from layers import Layer


class Activation(Layer):
    def __init__(self, activation: str, name: str = None) -> None:
        super().__init__(trainable=False, name=name)
        self.activation = activation
        self.activation_fn = self._get_activation_fn(activation)
        self.output_shape = None
        self.output = None

    def __call__(self, *input_layers):
        if len(input_layers) != 1:
            raise ValueError("Activation layer can only have one input layer")

        super().__call__(*input_layers)
        input_layer = input_layers[0]
        self.output_shape = input_layer.output_shape
        return self

    def _get_activation_fn(self, activation: Union[str, Callable]) -> Callable:
        if activation == "relu":
            return lambda x: np.maximum(x, 0)
        if activation == "sigmoid":
            return lambda x: 1 / (1 + np.exp(-x))
        if activation == "tanh":
            return lambda x: np.tanh(x)
        if activation == "softmax":
            return lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
        if activation is None:
            return lambda x: x
        if callable(activation):
            return activation

        raise ValueError(f"Unknown activation: {activation}")

    def predict(self, inputs):
        self.output = self.activation_fn(inputs)
        return self.output

    def _apply_gradients(self, gradients):
        if self.activation == "relu":
            return gradients * (self.output > 0)
        # TODO: check if this is correct
        if self.activation == "sigmoid":
            return gradients * self.output * (1 - self.output)
        if self.activation == "tanh":
            return gradients * (1 - self.output**2)
        if self.activation == "softmax":
            return gradients * self.output * (1 - self.output)
        if self.activation is None:
            return gradients
