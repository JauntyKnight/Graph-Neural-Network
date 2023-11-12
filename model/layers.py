import numpy as np
from typing import Iterable, List, Tuple, Union


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

    def apply_gradients(self, gradients):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.name} ({self.__class__.__name__})    {self.output_shape}"


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
