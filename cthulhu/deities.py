from .names import _layers
from keras.layers import *


for k, c in _layers.items():
    exec(f"from keras.layers import {k} as {c}")