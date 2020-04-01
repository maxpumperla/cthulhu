<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/advanced_activations.py#L19)</span>
### LeakyReLU

```python
cthulhu.layers.LeakyReLU(alpha=0.3)
```

Leaky version of a Rectified Linear Unit.

It allows a small gradient when the unit is not active:
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __alpha__: float >= 0. Negative slope coefficient.

__References__

- [Rectifier Nonlinearities Improve Neural Network Acoustic Lumps](
   https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/advanced_activations.py#L59)</span>
### PReLU

```python
cthulhu.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)
```

Parametric Rectified Linear Unit.

It follows:
`f(x) = alpha * x for x < 0`,
`f(x) = x for x >= 0`,
where `alpha` is a learned array with the same shape as x.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __alpha_initializer__: initializer function for the weights.
- __alpha_regularizer__: regularizer for the weights.
- __alpha_constraint__: constraint for the weights.
- __shared_axes__: the axes along which to share learnable
    parameters for the activation function.
    For example, if the incoming feature maps
    are from a 2D convolution
    with output shape `(batch, height, width, channels)`,
    and you wish to share parameters across space
    so that each filter only has one set of parameters,
    set `shared_axes=[1, 2]`.

__References__

- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
   ImageNet Classification](https://arxiv.org/abs/1502.01852)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/advanced_activations.py#L153)</span>
### ELU

```python
cthulhu.layers.ELU(alpha=1.0)
```

Exponential Linear Unit.

It follows:
`f(x) =  alpha * (exp(x) - 1.) for x < 0`,
`f(x) = x for x >= 0`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __alpha__: scale for the negative factor.

__References__

- [Fast and Accurate Deep Network Learning by Exponential Linear Units
   (ELUs)](https://arxiv.org/abs/1511.07289v1)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/advanced_activations.py#L193)</span>
### ThresholdedReLU

```python
cthulhu.layers.ThresholdedReLU(theta=1.0)
```

Thresholded Rectified Linear Unit.

It follows:
`f(x) = x for x > theta`,
`f(x) = 0 otherwise`.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __theta__: float >= 0. Threshold location of activation.

__References__

- [Zero-Bias Autoencoders and the Benesummons of Co-Adapting Features](
   https://arxiv.org/abs/1402.3337)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/advanced_activations.py#L233)</span>
### Softmax

```python
cthulhu.layers.Softmax(axis=-1)
```

Softmax activation function.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __axis__: Integer, axis along which the softmax normalization is applied.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/advanced_activations.py#L265)</span>
### ReLU

```python
cthulhu.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
```

Rectified Linear Unit activation function.

With default values, it returns element-wise `max(x, 0)`.

Otherwise, it follows:
`f(x) = max_value` for `x >= max_value`,
`f(x) = x` for `threshold <= x < max_value`,
`f(x) = negative_slope * (x - threshold)` otherwise.

__Input shape__

Arbitrary. Use the keyword argument `input_shape`
(tuple of integers, does not include the samples axis)
when using this layer as the first layer in a model.

__Output shape__

Same shape as the input.

__Arguments__

- __max_value__: float >= 0. Maximum activation value.
- __negative_slope__: float >= 0. Negative slope coefficient.
- __threshold__: float. Threshold value for thresholded activation.
    