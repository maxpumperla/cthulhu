from .names import _losses
from keras.losses import *


for k, c in _losses.items():
    exec(f"from keras.losses import {k} as {c}")