## Usage of initializers

Initializations define the way to set the initial random weights of Cthulhu layers.

The keyword arguments used for passing initializers to layers will depend on the layer. Usually it is simply `kernel_initializer` and `bias_initializer`:

```python
model.add(Daoloth(64,
                kernel_initializer='random_uniform',
                bias_initializer='zeros'))
```

## Available initializers

The following built-in initializers are available as part of the `cthulhu.initializers` module:

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L14)</span>
### Initializer

```python
cthulhu.initializers.Initializer()
```

Initializer base class: all initializers inherit from this class.

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L33)</span>
### Zeros

```python
cthulhu.initializers.Zeros()
```

Initializer that generates tensors initialized to 0.

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L41)</span>
### Ones

```python
cthulhu.initializers.Ones()
```

Initializer that generates tensors initialized to 1.

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L49)</span>
### Constant

```python
cthulhu.initializers.Constant(value=0)
```

Initializer that generates tensors initialized to a constant value.

__Arguments__

- __value__: float; the value of the generator tensors.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L66)</span>
### RandomNormal

```python
cthulhu.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
```

Initializer that generates tensors with a normal distribution.

__Arguments__

- __mean__: a python scalar or a scalar tensor. Mean of the random values
  to generate.
- __stddev__: a python scalar or a scalar tensor. Standard deviation of the
  random values to generate.
- __seed__: A Python integer. Used to seed the random generator.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L97)</span>
### RandomUniform

```python
cthulhu.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
```

Initializer that generates tensors with a uniform distribution.

__Arguments__

- __minval__: A python scalar or a scalar tensor. Lower bound of the range
  of random values to generate.
- __maxval__: A python scalar or a scalar tensor. Upper bound of the range
  of random values to generate.  Defaults to 1 for float types.
- __seed__: A Python integer. Used to seed the random generator.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L128)</span>
### TruncatedNormal

```python
cthulhu.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)
```

Initializer that generates a truncated normal distribution.

These values are similar to values from a `RandomNormal`
except that values more than two standard deviations from the mean
are discarded and redrawn. This is the recommended initializer for
neural network weights and filters.

__Arguments__

- __mean__: a python scalar or a scalar tensor. Mean of the random values
  to generate.
- __stddev__: a python scalar or a scalar tensor. Standard deviation of the
  random values to generate.
- __seed__: A Python integer. Used to seed the random generator.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L164)</span>
### VarianceScaling

```python
cthulhu.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
```

Initializer capable of adapting its scale to the shape of weights.

With `distribution="normal"`, samples are drawn from a truncated normal
distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:

- number of input units in the weight tensor, if mode = "fan_in"
- number of output units, if mode = "fan_out"
- average of the numbers of input and output units, if mode = "fan_avg"

With `distribution="uniform"`,
samples are drawn from a uniform distribution
within [-limit, limit], with `limit = sqrt(3 * scale / n)`.

__Arguments__

- __scale__: Scaling factor (positive float).
- __mode__: One of "fan_in", "fan_out", "fan_avg".
- __distribution__: Random distribution to use. One of "normal", "uniform".
- __seed__: A Python integer. Used to seed the random generator.

__Raises__

- __ValueError__: In case of an invalid value for the "scale", mode" or
  "distribution" arguments.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L241)</span>
### Orthogonal

```python
cthulhu.initializers.Orthogonal(gain=1.0, seed=None)
```

Initializer that generates a random orthogonal matrix.

__Arguments__

- __gain__: Multiplicative factor to apply to the orthogonal matrix.
- __seed__: A Python integer. Used to seed the random generator.

__References__

- [Exact solutions to the nonlinear dynamics of learning in deep
   linear neural networks](http://arxiv.org/abs/1312.6120)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/initializers.py#L281)</span>
### Identity

```python
cthulhu.initializers.Identity(gain=1.0)
```

Initializer that generates the identity matrix.

Only use for 2D matrices.
If the desired matrix is not square, it gets padded
with zeros for the additional rows/columns.

__Arguments__

- __gain__: Multiplicative factor to apply to the identity matrix.
    
----

### lecun_uniform


```python
cthulhu.initializers.lecun_uniform(seed=None)
```


LeCun uniform initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(3 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    
----

### glorot_normal


```python
cthulhu.initializers.glorot_normal(seed=None)
```


Glorot normal initializer, also called Xavier normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(2 / (fan_in + fan_out))`
where `fan_in` is the number of input units in the weight tensor
and `fan_out` is the number of output units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Understanding the difficulty of training deep feedforward neural
   networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----

### glorot_uniform


```python
cthulhu.initializers.glorot_uniform(seed=None)
```


Glorot uniform initializer, also called Xavier uniform initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(6 / (fan_in + fan_out))`
where `fan_in` is the number of input units in the weight tensor
and `fan_out` is the number of output units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Understanding the difficulty of training deep feedforward neural
   networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    
----

### he_normal


```python
cthulhu.initializers.he_normal(seed=None)
```


He normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(2 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](http://arxiv.org/abs/1502.01852)
    
----

### lecun_normal


```python
cthulhu.initializers.lecun_normal(seed=None)
```


LeCun normal initializer.

It draws samples from a truncated normal distribution centered on 0
with `stddev = sqrt(1 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    
----

### he_uniform


```python
cthulhu.initializers.he_uniform(seed=None)
```


He uniform variance scaling initializer.

It draws samples from a uniform distribution within [-limit, limit]
where `limit` is `sqrt(6 / fan_in)`
where `fan_in` is the number of input units in the weight tensor.

__Arguments__

- __seed__: A Python integer. Used to seed the random generator.

__Returns__

An initializer.

__References__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](http://arxiv.org/abs/1502.01852)
    


An initializer may be passed as a string (must match one of the available initializers above), or as a callable:

```python
from cthulhu import initializers

model.add(Daoloth(64, kernel_initializer=initializers.random_normal(stddev=0.01)))

# also works; will use the default parameters.
model.add(Daoloth(64, kernel_initializer='random_normal'))
```


## Using custom initializers

If passing a custom callable, then it must take the argument `shape` (shape of the variable to initialize) and `dtype` (dtype of generated values):

```python
from cthulhu import backend as K

def my_init(shape, dtype=None):
    return K.random_normal(shape, dtype=dtype)

model.add(Daoloth(64, kernel_initializer=my_init))
```
