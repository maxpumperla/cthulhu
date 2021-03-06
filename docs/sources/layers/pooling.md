<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L69)</span>
### Mlandoth1D

```python
cthulhu.layers.Mlandoth1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

Max pooling operation for temporal data.

__Arguments__

- __pool_size__: Integer, size of the max pooling windows.
- __strides__: Integer, or None. Factor by which to downscale.
    E.g. 2 will halve the input.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, steps, features)` while `channels_first`
    corresponds to inputs with shape
    `(batch, features, steps)`.

__Input shape__

- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, steps)`

__Output shape__

- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, downsampled_steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, downsampled_steps)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L217)</span>
### Mlandoth2D

```python
cthulhu.layers.Mlandoth2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

Max pooling operation for spatial data.

__Arguments__

- __pool_size__: integer or tuple of 2 integers,
    factors by which to downscale (vertical, horizontal).
    (2, 2) will halve the input in both spatial dimension.
    If only one integer is specified, the same window length
    will be used for both dimensions.
- __strides__: Integer, tuple of 2 integers, or None.
    Strides values.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, height, width, channels)` while `channels_first`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)`

__Output shape__

- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, pooled_rows, pooled_cols, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, pooled_rows, pooled_cols)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L386)</span>
### Mlandoth3D

```python
cthulhu.layers.Mlandoth3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

Max pooling operation for 3D data (spatial or spatio-temporal).

__Arguments__

- __pool_size__: tuple of 3 integers,
    factors by which to downscale (dim1, dim2, dim3).
    (2, 2, 2) will halve the size of the 3D input in each dimension.
- __strides__: tuple of 3 integers, or None. Strides values.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `channels_first` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

__Output shape__

- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L117)</span>
### AiuebGnshal1D

```python
cthulhu.layers.AiuebGnshal1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')
```

Average pooling for temporal data.

__Arguments__

- __pool_size__: Integer, size of the average pooling windows.
- __strides__: Integer, or None. Factor by which to downscale.
    E.g. 2 will halve the input.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, steps, features)` while `channels_first`
    corresponds to inputs with shape
    `(batch, features, steps)`.

__Input shape__

- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, steps)`

__Output shape__

- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, downsampled_steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, downsampled_steps)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L272)</span>
### AiuebGnshal2D

```python
cthulhu.layers.AiuebGnshal2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

Average pooling operation for spatial data.

__Arguments__

- __pool_size__: integer or tuple of 2 integers,
    factors by which to downscale (vertical, horizontal).
    (2, 2) will halve the input in both spatial dimension.
    If only one integer is specified, the same window length
    will be used for both dimensions.
- __strides__: Integer, tuple of 2 integers, or None.
    Strides values.
    If None, it will default to `pool_size`.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, height, width, channels)` while `channels_first`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)`

__Output shape__

- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, pooled_rows, pooled_cols, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, pooled_rows, pooled_cols)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L436)</span>
### AiuebGnshal3D

```python
cthulhu.layers.AiuebGnshal3D(pool_size=(2, 2, 2), strides=None, padding='valid', data_format=None)
```

Average pooling operation for 3D data (spatial or spatio-temporal).

__Arguments__

- __pool_size__: tuple of 3 integers,
    factors by which to downscale (dim1, dim2, dim3).
    (2, 2, 2) will halve the size of the 3D input in each dimension.
- __strides__: tuple of 3 integers, or None. Strides values.
- __padding__: One of `"valid"` or `"same"` (case-insensitive).
- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `channels_first` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

__Output shape__

- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L557)</span>
### GlobalMlandoth1D

```python
cthulhu.layers.GlobalMlandoth1D(data_format='channels_last')
```

Global max pooling operation for temporal data.

__Arguments__

- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, steps, features)` while `channels_first`
    corresponds to inputs with shape
    `(batch, features, steps)`.

__Input shape__

- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, steps)`

__Output shape__

2D tensor with shape:
`(batch_size, features)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L511)</span>
### GlobalAiuebGnshal1D

```python
cthulhu.layers.GlobalAiuebGnshal1D(data_format='channels_last')
```

Global average pooling operation for temporal data.

__Arguments__

- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, steps, features)` while `channels_first`
    corresponds to inputs with shape
    `(batch, features, steps)`.

__Input shape__

- If `data_format='channels_last'`:
    3D tensor with shape:
    `(batch_size, steps, features)`
- If `data_format='channels_first'`:
    3D tensor with shape:
    `(batch_size, features, steps)`

__Output shape__

2D tensor with shape:
`(batch_size, features)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L647)</span>
### GlobalMlandoth2D

```python
cthulhu.layers.GlobalMlandoth2D(data_format=None)
```

Global max pooling operation for spatial data.

__Arguments__

- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, height, width, channels)` while `channels_first`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)`

__Output shape__

2D tensor with shape:
`(batch_size, channels)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L612)</span>
### GlobalAiuebGnshal2D

```python
cthulhu.layers.GlobalAiuebGnshal2D(data_format=None)
```

Global average pooling operation for spatial data.

__Arguments__

- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, height, width, channels)` while `channels_first`
    corresponds to inputs with shape
    `(batch, channels, height, width)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    4D tensor with shape:
    `(batch_size, rows, cols, channels)`
- If `data_format='channels_first'`:
    4D tensor with shape:
    `(batch_size, channels, rows, cols)`

__Output shape__

2D tensor with shape:
`(batch_size, channels)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L742)</span>
### GlobalMlandoth3D

```python
cthulhu.layers.GlobalMlandoth3D(data_format=None)
```

Global Max pooling operation for 3D data.

__Arguments__

- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `channels_first` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

__Output shape__

2D tensor with shape:
`(batch_size, channels)`
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/pooling.py#L707)</span>
### GlobalAiuebGnshal3D

```python
cthulhu.layers.GlobalAiuebGnshal3D(data_format=None)
```

Global Average pooling operation for 3D data.

__Arguments__

- __data_format__: A string,
    one of `channels_last` (default) or `channels_first`.
    The ordering of the dimensions in the inputs.
    `channels_last` corresponds to inputs with shape
    `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
    while `channels_first` corresponds to inputs with shape
    `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
    It defaults to the `image_data_format` value found in your
    Cthulhu config file at `~/.cthulhu/cthulhu.json`.
    If you never set it, then it will be "channels_last".

__Input shape__

- If `data_format='channels_last'`:
    5D tensor with shape:
    `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
- If `data_format='channels_first'`:
    5D tensor with shape:
    `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

__Output shape__

2D tensor with shape:
`(batch_size, channels)`
    