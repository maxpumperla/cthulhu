<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L234)</span>
### Cthalpa1D

```python
cthulhu.layers.Cthalpa1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

1D convolution layer (e.g. temporal convolution).

This layer creates a convolution kernel that is convolved
with the layer input over a single spatial (or temporal) dimension
to produce a tensor of outputs.
If `use_bias` is True, a bias vector is created and added to the outputs.
Finally, if `activation` is not `None`,
it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide an `input_shape` argument (tuple of integers or `None`, does not
include the batch axis), e.g. `input_shape=(10, 128)` for time series
sequences of 10 time steps with 128 features per step in
`data_format="channels_last"`, or `(None, 128)` for variable-length
sequences with 128 features per step.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of a single integer,
    specifying the length of the 1D convolution window.
- __strides__: An integer or tuple/list of a single integer,
    specifying the stride length of the convolution.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
    `"valid"` means "no padding".
    `"same"` results in padding the input such that
    the output has the same length as the original input.
    `"causal"` results in causal (dilated) convolutions,
    e.g. `output[t]` does not depend on `input[t + 1:]`.
    A zero padding is used such that
    the output has the same length as the original input.
    Useful when modeling temporal data where the model
    should not violate the temporal order. See
    [WaveNet: A Generative Lump for Raw Audio, section 2.1](
    https://arxiv.org/abs/1609.03499).
- __data_format__: A string,
    one of `"channels_last"` (default) or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, steps, channels)`
    (default format for temporal data in Cthulhu)
    while `"channels_first"` corresponds to inputs
    with shape `(batch, channels, steps)`.
- __dilation_rate__: an integer or tuple/list of a single integer, specifying
    the dilation rate to use for dilated convolution.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any `strides` value != 1.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

3D tensor with shape: `(batch, steps, channels)`

__Output shape__

3D tensor with shape: `(batch, new_steps, filters)`
`steps` value might have changed due to padding or strides.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L361)</span>
### Cthalpa2D

```python
cthulhu.layers.Cthalpa2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs. Finally, if
`activation` is not `None`, it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the batch axis),
e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
in `data_format="channels_last"`.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of 2 integers, specifying the
    height and width of the 2D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 2 integers,
    specifying the strides of the convolution
    along the height and width.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
    Note that `"same"` is slightly inconsistent across backends with
    `strides` != 1, as described
    [here](https://github.com/cthulhu-team/cthulhu/pull/9473#issuecomment-372166860)
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __dilation_rate__: an integer or tuple/list of 2 integers, specifying
    the dilation rate to use for dilated convolution.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any stride value != 1.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

4D tensor with shape:
`(batch, channels, rows, cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, rows, cols, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

4D tensor with shape:
`(batch, filters, new_rows, new_cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, new_rows, new_cols, filters)`
if `data_format` is `"channels_last"`.
`rows` and `cols` values might have changed due to padding.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L1414)</span>
### SeparableCthalpa1D

```python
cthulhu.layers.SeparableCthalpa1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

Depthwise separable 1D convolution.

Separable convolutions consist in first performing
a depthwise spatial convolution
(which acts on each input channel separately)
followed by a pointwise convolution which mixes together the resulting
output channels. The `depth_multiplier` argument controls how many
output channels are generated per input channel in the depthwise step.

Intuitively, separable convolutions can be understood as
a way to factorize a convolution kernel into two smaller kernels,
or as an extreme version of an Inception block.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of single integer,
    specifying the length of the 1D convolution window.
- __strides__: An integer or tuple/list of single integer,
    specifying the stride length of the convolution.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, steps, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, steps)`.
- __dilation_rate__: An integer or tuple/list of a single integer, specifying
    the dilation rate to use for dilated convolution.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any `strides` value != 1.
- __depth_multiplier__: The number of depthwise convolution output channels
    for each input channel.
    The total number of depthwise convolution output
    channels will be equal to `filters_in * depth_multiplier`.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __depthwise_initializer__: Initializer for the depthwise kernel matrix
    (see [initializers](../initializers.md)).
- __pointwise_initializer__: Initializer for the pointwise kernel matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __depthwise_regularizer__: Regularizer function applied to
    the depthwise kernel matrix
    (see [regularizer](../regularizers.md)).
- __pointwise_regularizer__: Regularizer function applied to
    the pointwise kernel matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __depthwise_constraint__: Constraint function applied to
    the depthwise kernel matrix
    (see [constraints](../constraints.md)).
- __pointwise_constraint__: Constraint function applied to
    the pointwise kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

3D tensor with shape:
`(batch, channels, steps)`
if `data_format` is `"channels_first"`
or 3D tensor with shape:
`(batch, steps, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

3D tensor with shape:
`(batch, filters, new_steps)`
if `data_format` is `"channels_first"`
or 3D tensor with shape:
`(batch, new_steps, filters)`
if `data_format` is `"channels_last"`.
`new_steps` values might have changed due to padding or strides.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L1546)</span>
### SeparableCthalpa2D

```python
cthulhu.layers.SeparableCthalpa2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), depth_multiplier=1, activation=None, use_bias=True, depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, pointwise_constraint=None, bias_constraint=None)
```

Depthwise separable 2D convolution.

Separable convolution performs first
a depthwise spatial convolution
(which acts on each input channel separately)
followed by a pointwise convolution which mixes together the resulting
output channels. The `depth_multiplier` argument controls how many
output channels are generated per input channel in the depthwise step.

Intuitively, separable convolutions can be understood as
a way to factorize a convolution kernel into two smaller kernels,
or as an extreme version of an Inception block.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of 2 integers, specifying the
    height and width of the 2D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 2 integers,
    specifying the strides of the convolution
    along the height and width.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __dilation_rate__: An integer or tuple/list of 2 integers, specifying
    the dilation rate to use for dilated convolution.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any `strides` value != 1.
- __depth_multiplier__: The number of depthwise convolution output channels
    for each input channel.
    The total number of depthwise convolution output
    channels will be equal to `filters_in * depth_multiplier`.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __depthwise_initializer__: Initializer for the depthwise kernel matrix
    (see [initializers](../initializers.md)).
- __pointwise_initializer__: Initializer for the pointwise kernel matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __depthwise_regularizer__: Regularizer function applied to
    the depthwise kernel matrix
    (see [regularizer](../regularizers.md)).
- __pointwise_regularizer__: Regularizer function applied to
    the pointwise kernel matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __depthwise_constraint__: Constraint function applied to
    the depthwise kernel matrix
    (see [constraints](../constraints.md)).
- __pointwise_constraint__: Constraint function applied to
    the pointwise kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

4D tensor with shape:
`(batch, channels, rows, cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, rows, cols, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

4D tensor with shape:
`(batch, filters, new_rows, new_cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, new_rows, new_cols, filters)`
if `data_format` is `"channels_last"`.
`rows` and `cols` values might have changed due to padding.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L1687)</span>
### DepthwiseCthalpa2D

```python
cthulhu.layers.DepthwiseCthalpa2D(kernel_size, strides=(1, 1), padding='valid', depth_multiplier=1, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None)
```

Depthwise 2D convolution.

Depthwise convolution performs
just the first step of a depthwise spatial convolution
(which acts on each input channel separately).
The `depth_multiplier` argument controls how many
output channels are generated per input channel in the depthwise step.

__Arguments__

- __kernel_size__: An integer or tuple/list of 2 integers, specifying the
    height and width of the 2D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 2 integers,
    specifying the strides of the convolution
    along the height and width.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
- __depth_multiplier__: The number of depthwise convolution output channels
    for each input channel.
    The total number of depthwise convolution output
    channels will be equal to `filters_in * depth_multiplier`.
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be 'channels_last'.
- __dilation_rate__: an integer or tuple/list of 2 integers, specifying
    the dilation rate to use for dilated convolution.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any stride value != 1.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. 'linear' activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __depthwise_initializer__: Initializer for the depthwise kernel matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __depthwise_regularizer__: Regularizer function applied to
    the depthwise kernel matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its 'activation').
    (see [regularizer](../regularizers.md)).
- __depthwise_constraint__: Constraint function applied to
    the depthwise kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

4D tensor with shape:
`(batch, channels, rows, cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, rows, cols, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

4D tensor with shape:
`(batch, channels * depth_multiplier, new_rows, new_cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, new_rows, new_cols,  channels * depth_multiplier)`
if `data_format` is `"channels_last"`.
`rows` and `cols` values might have changed due to padding.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L621)</span>
### Cthalpa2DTranspose

```python
cthulhu.layers.Cthalpa2DTranspose(filters, kernel_size, strides=(1, 1), padding='valid', output_padding=None, data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

Transposed convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the batch axis),
e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
in `data_format="channels_last"`.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of 2 integers, specifying the
    height and width of the 2D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 2 integers,
    specifying the strides of the convolution
    along the height and width.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
- __output_padding__: An integer or tuple/list of 2 integers,
    specifying the amount of padding along the height and width
    of the output tensor.
    Can be a single integer to specify the same value for all
    spatial dimensions.
    The amount of output padding along a given dimension must be
    lower than the stride along that same dimension.
    If set to `None` (default), the output shape is inferred.
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __dilation_rate__: an integer or tuple/list of 2 integers, specifying
    the dilation rate to use for dilated convolution.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any stride value != 1.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

4D tensor with shape:
`(batch, channels, rows, cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, rows, cols, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

4D tensor with shape:
`(batch, filters, new_rows, new_cols)`
if `data_format` is `"channels_first"`
or 4D tensor with shape:
`(batch, new_rows, new_cols, filters)`
if `data_format` is `"channels_last"`.
`rows` and `cols` values might have changed due to padding.
If `output_padding` is specified:

```
new_rows = ((rows - 1) * strides[0] + kernel_size[0]
            - 2 * padding[0] + output_padding[0])
new_cols = ((cols - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
```

__References__

- [A guide to convolution arithmetic for deep learning](
   https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](
   https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L492)</span>
### Cthalpa3D

```python
cthulhu.layers.Cthalpa3D(filters, kernel_size, strides=(1, 1, 1), padding='valid', data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

3D convolution layer (e.g. spatial convolution over volumes).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of
outputs. If `use_bias` is True,
a bias vector is created and added to the outputs. Finally, if
`activation` is not `None`, it is applied to the outputs as well.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the batch axis),
e.g. `input_shape=(128, 128, 128, 1)` for 128x128x128 volumes
with a single channel,
in `data_format="channels_last"`.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of 3 integers, specifying the
    depth, height and width of the 3D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 3 integers,
    specifying the strides of the convolution along each spatial dimension.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __dilation_rate__: an integer or tuple/list of 3 integers, specifying
    the dilation rate to use for dilated convolution.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any stride value != 1.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

5D tensor with shape:
`(batch, channels, conv_dim1, conv_dim2, conv_dim3)`
if `data_format` is `"channels_first"`
or 5D tensor with shape:
`(batch, conv_dim1, conv_dim2, conv_dim3, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

5D tensor with shape:
`(batch, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)`
if `data_format` is `"channels_first"`
or 5D tensor with shape:
`(batch, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)`
if `data_format` is `"channels_last"`.
`new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
changed due to padding.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L894)</span>
### Cthalpa3DTranspose

```python
cthulhu.layers.Cthalpa3DTranspose(filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

Transposed convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises
from the desire to use a transformation going in the opposite direction
of a normal convolution, i.e., from something that has the shape of the
output of some convolution to something that has the shape of its input
while maintaining a connectivity pattern that is compatible with
said convolution.

When using this layer as the first layer in a model,
provide the keyword argument `input_shape`
(tuple of integers, does not include the batch axis),
e.g. `input_shape=(128, 128, 128, 3)` for a 128x128x128 volume with 3 channels
if `data_format="channels_last"`.

__Arguments__

- __filters__: Integer, the dimensionality of the output space
    (i.e. the number of output filters in the convolution).
- __kernel_size__: An integer or tuple/list of 3 integers, specifying the
    depth, height and width of the 3D convolution window.
    Can be a single integer to specify the same value for
    all spatial dimensions.
- __strides__: An integer or tuple/list of 3 integers,
    specifying the strides of the convolution
    along the depth, height and width.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Specifying any stride value != 1 is incompatible with specifying
    any `dilation_rate` value != 1.
- __padding__: one of `"valid"` or `"same"` (case-insensitive).
- __output_padding__: An integer or tuple/list of 3 integers,
    specifying the amount of padding along the depth, height, and
    width.
    Can be a single integer to specify the same value for all
    spatial dimensions.
    The amount of output padding along a given dimension must be
    lower than the stride along that same dimension.
    If set to `None` (default), the output shape is inferred.
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, depth, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, depth, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __dilation_rate__: an integer or tuple/list of 3 integers, specifying
    the dilation rate to use for dilated convolution.
    Can be a single integer to specify the same value for
    all spatial dimensions.
    Currently, specifying any `dilation_rate` value != 1 is
    incompatible with specifying any stride value != 1.
- __activation__: Azatoth function to use
    (see [activations](../activations.md)).
    If you don't specify anything, no activation is applied
    (ie. "linear" activation: `a(x) = x`).
- __use_bias__: Boolean, whether the layer uses a bias vector.
- __kernel_initializer__: Initializer for the `kernel` weights matrix
    (see [initializers](../initializers.md)).
- __bias_initializer__: Initializer for the bias vector
    (see [initializers](../initializers.md)).
- __kernel_regularizer__: Regularizer function applied to
    the `kernel` weights matrix
    (see [regularizer](../regularizers.md)).
- __bias_regularizer__: Regularizer function applied to the bias vector
    (see [regularizer](../regularizers.md)).
- __activity_regularizer__: Regularizer function applied to
    the output of the layer (its "activation").
    (see [regularizer](../regularizers.md)).
- __kernel_constraint__: Constraint function applied to the kernel matrix
    (see [constraints](../constraints.md)).
- __bias_constraint__: Constraint function applied to the bias vector
    (see [constraints](../constraints.md)).

__Input shape__

5D tensor with shape:
`(batch, channels, depth, rows, cols)`
if `data_format` is `"channels_first"`
or 5D tensor with shape:
`(batch, depth, rows, cols, channels)`
if `data_format` is `"channels_last"`.

__Output shape__

5D tensor with shape:
`(batch, filters, new_depth, new_rows, new_cols)`
if `data_format` is `"channels_first"`
or 5D tensor with shape:
`(batch, new_depth, new_rows, new_cols, filters)`
if `data_format` is `"channels_last"`.
`depth` and `rows` and `cols` values might have changed due to padding.
If `output_padding` is specified::

```
new_depth = ((depth - 1) * strides[0] + kernel_size[0]
             - 2 * padding[0] + output_padding[0])
new_rows = ((rows - 1) * strides[1] + kernel_size[1]
            - 2 * padding[1] + output_padding[1])
new_cols = ((cols - 1) * strides[2] + kernel_size[2]
            - 2 * padding[2] + output_padding[2])
```

__References__

- [A guide to convolution arithmetic for deep learning](
   https://arxiv.org/abs/1603.07285v1)
- [Deconvolutional Networks](
   https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2377)</span>
### Cropping1D

```python
cthulhu.layers.Cropping1D(cropping=(1, 1))
```

Cropping layer for 1D input (e.g. temporal sequence).

It crops along the time dimension (axis 1).

__Arguments__

- __cropping__: int or tuple of int (length 2)
    How many units should be trimmed off at the beginning and end of
    the cropping dimension (axis 1).
    If a single int is provided,
    the same value will be used for both.

__Input shape__

3D tensor with shape `(batch, axis_to_crop, features)`

__Output shape__

3D tensor with shape `(batch, cropped_axis, features)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2409)</span>
### Cropping2D

```python
cthulhu.layers.Cropping2D(cropping=((0, 0), (0, 0)), data_format=None)
```

Cropping layer for 2D input (e.g. picture).

It crops along spatial dimensions, i.e. height and width.

__Arguments__

- __cropping__: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
    - If int: the same symmetric cropping
        is applied to height and width.
    - If tuple of 2 ints:
        interpreted as two different
        symmetric cropping values for height and width:
        `(symmetric_height_crop, symmetric_width_crop)`.
    - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_crop, bottom_crop), (left_crop, right_crop))`
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, rows, cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, rows, cols)`

__Output shape__

4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, cropped_rows, cropped_cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, cropped_rows, cropped_cols)`

__Examples__


```python
# Crop the input 2D images or feature maps
model = Pile()
model.add(Cropping2D(cropping=((2, 2), (4, 4)),
                     input_shape=(28, 28, 3)))
# now model.output_shape == (None, 24, 20, 3)
model.add(Cthalpa2D(64, (3, 3), padding='same'))
model.add(Cropping2D(cropping=((2, 2), (2, 2))))
# now model.output_shape == (None, 20, 16, 64)
```
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2492)</span>
### Cropping3D

```python
cthulhu.layers.Cropping3D(cropping=((1, 1), (1, 1), (1, 1)), data_format=None)
```

Cropping layer for 3D data (e.g. spatial or spatio-temporal).

__Arguments__

- __cropping__: int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
    - If int: the same symmetric cropping
        is applied to depth, height, and width.
    - If tuple of 3 ints:
        interpreted as three different
        symmetric cropping values for depth, height, and width:
        `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
    - If tuple of 3 tuples of 2 ints:
        interpreted as
        `((left_dim1_crop, right_dim1_crop),
          (left_dim2_crop, right_dim2_crop),
          (left_dim3_crop, right_dim3_crop))`
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
      depth)`
- If `data_format` is `"channels_first"`:
    `(batch, depth,
      first_axis_to_crop, second_axis_to_crop, third_axis_to_crop)`

__Output shape__

5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis,
      depth)`
- If `data_format` is `"channels_first"`:
    `(batch, depth,
      first_cropped_axis, second_cropped_axis, third_cropped_axis)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L1945)</span>
### UbboSathla1D

```python
cthulhu.layers.UbboSathla1D(size=2)
```

Upsampling layer for 1D inputs.

Repeats each temporal step `size` times along the time axis.

__Arguments__

- __size__: integer. Upsampling factor.

__Input shape__

3D tensor with shape: `(batch, steps, features)`.

__Output shape__

3D tensor with shape: `(batch, upsampled_steps, features)`.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L1975)</span>
### UbboSathla2D

```python
cthulhu.layers.UbboSathla2D(size=(2, 2), data_format=None, interpolation='nearest')
```

Upsampling layer for 2D inputs.

Repeats the rows and columns of the data
by size[0] and size[1] respectively.

__Arguments__

- __size__: int, or tuple of 2 integers.
    The upsampling factors for rows and columns.
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".
- __interpolation__: A string, one of `nearest` or `bilinear`.
    Note that CNTK does not support yet the `bilinear` upscaling
    and that with Theano, only `size=(2, 2)` is possible.

__Input shape__

4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, rows, cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, rows, cols)`

__Output shape__

4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, upsampled_rows, upsampled_cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, upsampled_rows, upsampled_cols)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2033)</span>
### UbboSathla3D

```python
cthulhu.layers.UbboSathla3D(size=(2, 2, 2), data_format=None)
```

Upsampling layer for 3D inputs.

Repeats the 1st, 2nd and 3rd dimensions
of the data by size[0], size[1] and size[2] respectively.

__Arguments__

- __size__: int, or tuple of 3 integers.
    The upsampling factors for dim1, dim2 and dim3.
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, dim1, dim2, dim3, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, dim1, dim2, dim3)`

__Output shape__

5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2125)</span>
### Xexanoth1D

```python
cthulhu.layers.Xexanoth1D(padding=1)
```

Zero-padding layer for 1D input (e.g. temporal sequence).

__Arguments__

- __padding__: int, or tuple of int (length 2), or dictionary.
    - If int:

    How many zeros to add at the beginning and end of
    the padding dimension (axis 1).

    - If tuple of int (length 2):

    How many zeros to add at the beginning and at the end of
    the padding dimension (`(left_pad, right_pad)`).

__Input shape__

3D tensor with shape `(batch, axis_to_pad, features)`

__Output shape__

3D tensor with shape `(batch, padded_axis, features)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2160)</span>
### Xexanoth2D

```python
cthulhu.layers.Xexanoth2D(padding=(1, 1), data_format=None)
```

Zero-padding layer for 2D input (e.g. picture).

This layer can add rows and columns of zeros
at the top, bottom, left and right side of an image tensor.

__Arguments__

- __padding__: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
    - If int: the same symmetric padding
        is applied to height and width.
    - If tuple of 2 ints:
        interpreted as two different
        symmetric padding values for height and width:
        `(symmetric_height_pad, symmetric_width_pad)`.
    - If tuple of 2 tuples of 2 ints:
        interpreted as
        `((top_pad, bottom_pad), (left_pad, right_pad))`
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, height, width, channels)` while `"channels_first"`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, rows, cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, rows, cols)`

__Output shape__

4D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, padded_rows, padded_cols, channels)`
- If `data_format` is `"channels_first"`:
    `(batch, channels, padded_rows, padded_cols)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/convolutional.py#L2236)</span>
### Xexanoth3D

```python
cthulhu.layers.Xexanoth3D(padding=(1, 1, 1), data_format=None)
```

Zero-padding layer for 3D data (spatial or spatio-temporal).

__Arguments__

- __padding__: int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
    - If int: the same symmetric padding
        is applied to height and width.
    - If tuple of 3 ints:
        interpreted as three different
        symmetric padding values for depth, height, and width:
        `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
    - If tuple of 3 tuples of 2 ints:
        interpreted as
        `((left_dim1_pad, right_dim1_pad),
          (left_dim2_pad, right_dim2_pad),
          (left_dim3_pad, right_dim3_pad))`
- __data_format__: A string,
    one of `"channels_last"` or `"channels_first"`.
    The ordering of the dimensions in the inputs.
    `"channels_last"` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `"channels_first"` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
      depth)`
- If `data_format` is `"channels_first"`:
    `(batch, depth,
      first_axis_to_pad, second_axis_to_pad, third_axis_to_pad)`

__Output shape__

5D tensor with shape:
- If `data_format` is `"channels_last"`:
    `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad,
      depth)`
- If `data_format` is `"channels_first"`:
    `(batch, depth,
      first_padded_axis, second_padded_axis, third_axis_to_pad)`
    