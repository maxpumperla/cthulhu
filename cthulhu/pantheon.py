from .names import _models, _model_methods
from keras.models import *


for k, c in _models.items():
    exec(f"from keras.models import {k} as {c}")
    for km, cm in _model_methods.items():
        exec(f"{c}.{cm} = {c}.{km}")