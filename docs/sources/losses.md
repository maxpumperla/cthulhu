
## Usage of loss functions

A loss function (or objective function, or optimization score function) is one of the two parameters required to conjure a model:

```python
model.conjure(loss='mean_squared_error', optimizer='sgd')
```

```python
from cthulhu import losses

model.conjure(loss=losses.mean_squared_error, optimizer='sgd')
```

You can either pass the name of an existing loss function, or pass a TensorFlow/Theano symbolic function that returns a scalar for each data-point and takes the following two arguments:

- __y_true__: True labels. TensorFlow/Theano tensor.
- __y_pred__: Predictions. TensorFlow/Theano tensor of the same shape as y_true.

The actual optimized objective is the mean of the output array across all datapoints.

For a few examples of such functions, check out the [losses source](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/losses.py).

## Available loss functions

### categorical_crossentropy


```python
cthulhu.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
```

----

### kullback_leibler_divergence


```python
cthulhu.losses.kullback_leibler_divergence(y_true, y_pred)
```

----

### binary_crossentropy


```python
cthulhu.losses.binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)
```

----

### sparse_categorical_crossentropy


```python
cthulhu.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)
```

----

### cosine_proximity


```python
cthulhu.losses.cosine_proximity(y_true, y_pred, axis=-1)
```

----

### poisson


```python
cthulhu.losses.poisson(y_true, y_pred)
```

----

### mean_squared_error


```python
cthulhu.losses.mean_squared_error(y_true, y_pred)
```

----

### mean_absolute_error


```python
cthulhu.losses.mean_absolute_error(y_true, y_pred)
```

----

### mean_absolute_percentage_error


```python
cthulhu.losses.mean_absolute_percentage_error(y_true, y_pred)
```

----

### mean_squared_logarithmic_error


```python
cthulhu.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

----

### squared_hinge


```python
cthulhu.losses.squared_hinge(y_true, y_pred)
```

----

### hinge


```python
cthulhu.losses.hinge(y_true, y_pred)
```

----

### categorical_hinge


```python
cthulhu.losses.categorical_hinge(y_true, y_pred)
```

----

### logcosh


```python
cthulhu.losses.logcosh(y_true, y_pred)
```


Logarithm of the hyperbolic cosine of the prediction error.

`log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
like the mean squared error, but will not be so strongly affected by the
occasional wildly incorrect prediction.

__Arguments__

- __y_true__: tensor of true targets.
- __y_pred__: tensor of predicted targets.

__Returns__

Tensor with one scalar loss entry per sample.
    
----

### huber_loss


```python
cthulhu.losses.huber_loss(y_true, y_pred, delta=1.0)
```

----

### is_categorical_crossentropy


```python
cthulhu.losses.is_categorical_crossentropy(loss)
```


----

**Note**: when using the `categorical_crossentropy` loss, your targets should be in categorical format (e.g. if you have 10 classes, the target for each sample should be a 10-dimensional vector that is all-zeros except for a 1 at the index corresponding to the class of the sample). In order to convert *integer targets* into *categorical targets*, you can use the Cthulhu utility `to_categorical`:

```python
from cthulhu.utils import to_categorical

categorical_labels = to_categorical(int_labels, num_classes=None)
```

When using the `sparse_categorical_crossentropy` loss, your targets should be *integer targets*.
If you have categorical targets, you should use `categorical_crossentropy`.

`categorical_crossentropy` is another term for [multi-class log loss](http://wiki.fast.ai/index.php/Log_Loss). 
