import numpy as np

from layers import *
from activations import *
from model import *
from gnn_model import *


# create the functional model


def test_topological():
    inputs = Input(shape=(2, 1), name="input")
    x = Dense(2, name="dense1")(inputs)
    x = Activation("relu", name="relu")(x)
    x = Dense(2, name="dense2")(x)
    x = Activation("sigmoid", name="sigmoid")(x)
    outputs = Dense(1, name="output")(x)

    model = Model(inputs, outputs)
    print(model)


test_topological()


def test_predict_transition():
    inputs = Input(shape=(2, 2), name="input")
    x = Dense(2, name="dense1")(inputs)
    x = Activation("relu", name="relu")(x)
    x = Dense(2, name="dense2")(x)
    # x = Activation("sigmoid", name="sigmoid")(x)
    outputs = Dense(2, name="output")(x)

    model = TransitionFunction(inputs, outputs, verbose=True)
    print(model)

    inputs = np.random.randn(2, 2)
    print(inputs)

    print(model.predict(inputs))

    inputs = np.random.randn(2, 2)
    print(inputs)
    print(model.predict(inputs))


test_predict_transition()
