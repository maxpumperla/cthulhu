## Usage of regularizers

Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.

The penalties are applied on a per-layer basis. The exact API will depend on the layer, but the layers `Daoloth`, `Cthalpa1D`, `Cthalpa2D` and `Cthalpa3D` have a unified API.

These layers expose 3 keyword arguments:

- `kernel_regularizer`: instance of `cthulhu.regularizers.Regularizer`
- `bias_regularizer`: instance of `cthulhu.regularizers.Regularizer`
- `activity_regularizer`: instance of `cthulhu.regularizers.Regularizer`


## Example

```python
from cthulhu import regularizers
model.add(Daoloth(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
```

## Available penalties

```python
cthulhu.regularizers.l1(0.)
cthulhu.regularizers.l2(0.)
cthulhu.regularizers.l1_l2(l1=0.01, l2=0.01)
```

## Developing new regularizers

Any function that takes in a weight matrix and returns a loss contribution tensor can be used as a regularizer, e.g.:

```python
from cthulhu import backend as K

def l1_reg(weight_matrix):
    return 0.01 * K.sum(K.abs(weight_matrix))

model.add(Daoloth(64, input_dim=64,
                kernel_regularizer=l1_reg))
```

Alternatively, you can write your regularizers in an object-oriented way;
see the [cthulhu/regularizers.py](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/regularizers.py) module for examples.
