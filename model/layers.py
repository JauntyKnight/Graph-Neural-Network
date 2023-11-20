import numpy as np
from typing import Iterable, List, Tuple, Union, Callable


class Layer:
    def __init__(
        self, trainable: bool = True, name: str = None, dtype: np.number = np.float32
    ) -> None:
        self.trainable = trainable
        self.name = name
        self.dtype = dtype
        self.parent_layers = []
        self.child_layers = []
        self.output_shape = None

    def __call__(self, *input_layers):
        if len(input_layers) == 0:
            raise ValueError("No input layers provided")

        self.parent_layers.extend(input_layers)
        for layer in input_layers:
            layer.child_layers.append(self)

        return self

    def predict(self, inputs):
        raise NotImplementedError

    def _compute_output_shape(self, input_shape):
        raise NotImplementedError

    def apply_gradients(self, gradients, learning_rate=0.01):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name} ({self.__class__.__name__})    {self.output_shape}"


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


class Input(Layer):
    def __init__(
        self,
        shape: Union[int, Iterable[int]],
        trainable: bool = True,
        name: str = None,
        dtype: np.number = np.float32,
    ) -> None:
        super().__init__(trainable, name, dtype)
        self.shape = shape
        self._compute_output_shape(shape)
        self.outputs = None

    def predict(self, inputs):
        self.outputs = inputs
        return inputs

    def _apply_gradients(self, gradients):
        return gradients

    def _compute_output_shape(self, input_shape):
        self.output_shape = input_shape


class Dense(Layer):
    def __init__(
        self,
        units: int,
        trainable: bool = True,
        use_bias: bool = True,
        name: str = None,
        dtype: np.number = np.float32,
    ) -> None:
        super().__init__(trainable, name, dtype)
        self.units = units
        self.use_bias = use_bias
        self.w = None
        self.b = None
        self.outputs = None

    def build(self, input_shape):  # initialize weights and biases
        # use He initialization
        self.w = np.random.randn(input_shape[-1], self.units).astype(self.dtype) * np.sqrt(
            2 / input_shape[-1]
        )

        if self.use_bias:
            self.b = np.random.randn(self.units).astype(self.dtype)
        else:
            self.b = np.zeros(self.units).astype(self.dtype)

    def call(self, inputs):
        return inputs @ self.w + self.b

    # used for Functional construction of the model
    def __call__(self, *input_layers):
        self.build(input_layers[0].output_shape)  # TODO: check input_shape
        self._compute_output_shape(input_layers[0].output_shape)
        return super().__call__(*input_layers)

    def _compute_output_shape(self, input_shape):
        self.output_shape = (input_shape[0], self.units)

    def predict(self, inputs):
        self.outputs = self.call(inputs)
        return self.outputs

    def apply_gradients(self, gradients, learning_rate=0.01):
        if self.trainable:
            to_return = gradients @ self.w.T
            if self.use_bias:
                self.b -= learning_rate * np.sum(gradients, axis=0)
            self.w -= learning_rate * np.outer(self.parent_layers[0].outputs, gradients)

            return to_return

        return gradients
