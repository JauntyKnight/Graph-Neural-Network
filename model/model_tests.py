import numpy as np

from layers import *
from activations import *
from model import *


# create the functional model


def test1():
    inputs = Input(shape=(2,), name="input")
    x = Dense(2, name="dense1")(inputs)
    x = Activation("relu", name="relu")(x)
    x = Dense(2, name="dense2")(x)
    x = Activation("sigmoid", name="sigmoid")(x)
    outputs = Dense(1, name="output")(x)

    model = Model(inputs, outputs)
    print(model)


test1()
