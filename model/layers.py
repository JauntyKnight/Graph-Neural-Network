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

    def predict(self, inputs):
        return inputs

    def apply_gradients(self, gradients):
        pass

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

    def build(self, input_shape):  # initialize weights and biases
        self.w = np.random.randn(input_shape[-1], self.units).astype(self.dtype)

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
        return self.call(inputs)
