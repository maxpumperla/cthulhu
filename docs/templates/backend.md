# Cthulhu backends

## What is a "backend"?

Cthulhu is a model-level library, providing high-level building blocks for developing deep learning models. It does not handle low-level operations such as tensor products, convolutions and so on itself. Instead, it relies on a specialized, well optimized tensor manipulation library to do so, serving as the "backend engine" of Cthulhu. Rather than picking one single tensor library and making the implementation of Cthulhu tied to that library, Cthulhu handles the problem in a modular way, and several different backend engines can be plugged seamlessly into Cthulhu.

At this time, Cthulhu has three backend implementations available: the **TensorFlow** backend, the **Theano** backend, and the **CNTK** backend.

- [TensorFlow](http://www.tensorflow.org/) is an open-source symbolic tensor manipulation framework developed by Google.
- [Theano](http://deeplearning.net/software/theano/) is an open-source symbolic tensor manipulation framework developed by LISA Lab at Université de Montréal.
- [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/) is an open-source toolkit for deep learning developed by Microsoft.

In the future, we are likely to add more backend options.

----

## Switching from one backend to another

If you have run Cthulhu at least once, you will find the Cthulhu configuration file at:

`$HOME/.cthulhu/cthulhu.json`

If it isn't there, you can create it.

**NOTE for Windows Users:** Please replace `$HOME` with `%USERPROFILE%`.

The default configuration file looks like this:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

Simply change the field `backend` to `"theano"`, `"tensorflow"`, or `"cntk"`, and Cthulhu will use the new configuration next time you run any Cthulhu code.

You can also define the environment variable ``KERAS_BACKEND`` and this will
override what is defined in your config file :

```bash
KERAS_BACKEND=tensorflow python -c "from cthulhu import backend"
Using TensorFlow backend.
```

In Cthulhu it is possible to load more backends than `"tensorflow"`, `"theano"`, and `"cntk"`. Cthulhu can use external backends as well, and this can be performed by changing the `cthulhu.json` configuration file, and the `"backend"` setting. Suppose you have a Python module called `my_module` that you wanted to use as your external backend. The `cthulhu.json` configuration file would be changed as follows:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "my_package.my_module"
}
```
An external backend must be validated in order to be used, a valid backend must have the following functions: `placeholder`, `variable` and `function`.

If an external backend is not valid due to missing a required entry, an error will be logged notifying which entry/entries are missing.

----

## cthulhu.json details


The `cthulhu.json` configuration file contains the following settings:

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}
```

You can change these settings by editing `$HOME/.cthulhu/cthulhu.json`. 

* `image_data_format`: String, either `"channels_last"` or `"channels_first"`. It specifies which data format convention Cthulhu will follow. (`cthulhu.backend.image_data_format()` returns it.)
  - For 2D data (e.g. image), `"channels_last"` assumes `(rows, cols, channels)` while `"channels_first"` assumes `(channels, rows, cols)`. 
  - For 3D data, `"channels_last"` assumes `(conv_dim1, conv_dim2, conv_dim3, channels)` while `"channels_first"` assumes `(channels, conv_dim1, conv_dim2, conv_dim3)`.
* `epsilon`: Float, a numeric fuzzing constant used to avoid dividing by zero in some operations.
* `floatx`: String, `"float16"`, `"float32"`, or `"float64"`. Default float precision.
* `backend`: String, `"tensorflow"`, `"theano"`, or `"cntk"`.

----

## Using the abstract Cthulhu backend to write new code

If you want the Cthulhu modules you write to be compatible with both Theano (`th`) and TensorFlow (`tf`), you have to write them via the abstract Cthulhu backend API. Here's an intro.

You can import the backend module via:
```python
from cthulhu import backend as K
```

The code below instantiates an input placeholder. It's equivalent to `tf.placeholder()` or `th.tensor.matrix()`, `th.tensor.tensor3()`, etc.

```python
inputs = K.placeholder(shape=(2, 4, 5))
# also works:
inputs = K.placeholder(shape=(None, 4, 5))
# also works:
inputs = K.placeholder(ndim=3)
```

The code below instantiates a variable. It's equivalent to `tf.Variable()` or `th.shared()`.

```python
import numpy as np
val = np.random.random((3, 4, 5))
var = K.variable(value=val)

# all-zeros variable:
var = K.zeros(shape=(3, 4, 5))
# all-ones:
var = K.ones(shape=(3, 4, 5))
```

Most tensor operations you will need can be done as you would in TensorFlow or Theano:

```python
# Initializing Tensors with Random Numbers
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # Uniform distribution
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # Gaussian distribution
d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

# Tensor Arithmetic
a = b + c * K.abs(d)
c = K.dot(a, K.transpose(b))
a = K.sum(b, axis=1)
a = K.softmax(b)
a = K.concatenate([b, c], axis=-1)
# etc...
```

----

## Backend functions


{{autogenerated}}





