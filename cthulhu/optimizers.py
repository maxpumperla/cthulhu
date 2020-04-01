from .names import _optimizers
from keras.optimizers import *


for k, c in _optimizers.items():
    exec(f"from keras.optimizers import {k} as {c}")