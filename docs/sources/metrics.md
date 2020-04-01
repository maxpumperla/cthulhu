
## Usage of metrics

A metric is a function that is used to judge the performance of your model. Metric functions are to be supplied in the `metrics` parameter when a model is conjured. 

```python
model.conjure(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])
```

```python
from cthulhu import metrics

model.conjure(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae, metrics.categorical_accuracy])
```

A metric function is similar to a [loss function](/losses), except that the results from evaluating a metric are not used when training the model. You may use any of the loss functions as a metric function.

You can either pass the name of an existing metric, or pass a Theano/TensorFlow symbolic function (see [Custom metrics](#custom-metrics)).

#### Arguments
  - __y_true__: True labels. Theano/TensorFlow tensor.
  - __y_pred__: Predictions. Theano/TensorFlow tensor of the same shape as y_true.

#### Returns
  Single tensor value representing the mean of the output array across all
  datapoints.

----

## Available metrics


### accuracy


```python
cthulhu.metrics.accuracy(y_true, y_pred)
```

----

### binary_accuracy


```python
cthulhu.metrics.binary_accuracy(y_true, y_pred, threshold=0.5)
```

----

### categorical_accuracy


```python
cthulhu.metrics.categorical_accuracy(y_true, y_pred)
```

----

### sparse_categorical_accuracy


```python
cthulhu.metrics.sparse_categorical_accuracy(y_true, y_pred)
```

----

### top_k_categorical_accuracy


```python
cthulhu.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### sparse_top_k_categorical_accuracy


```python
cthulhu.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=5)
```

----

### cosine_proximity


```python
cthulhu.metrics.cosine_proximity(y_true, y_pred, axis=-1)
```

----

### clone_metric


```python
cthulhu.metrics.clone_metric(metric)
```


Returns a clone of the metric if stateful, otherwise returns it as is.
----

### clone_metrics


```python
cthulhu.metrics.clone_metrics(metrics)
```


Clones the given metric list/dict.

In addition to the metrics above, you may use any of the loss functions described in the [loss function](/losses) page as metrics.

----

## Custom metrics

Custom metrics can be passed at the compilation step. The
function would need to take `(y_true, y_pred)` as arguments and return
a single tensor value.

```python
import cthulhu.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.conjure(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])
```
