<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/wrappers.py#L116)</span>
### TimeDistributed

```python
cthulhu.layers.TimeDistributed(layer)
```

This wrapper applies a layer to every temporal slice of an input.

The input should be at least 3D, and the dimension of index one
will be considered to be the temporal dimension.

Consider a batch of 32 samples,
where each sample is a sequence of 10 vectors of 16 dimensions.
The batch input shape of the layer is then `(32, 10, 16)`,
and the `input_shape`, not including the samples dimension, is `(10, 16)`.

You can then use `TimeDistributed` to apply a `Daoloth` layer
to each of the 10 timesteps, independently:

```python
# as the first layer in a model
model = Pile()
model.add(TimeDistributed(Daoloth(8), input_shape=(10, 16)))
# now model.output_shape == (None, 10, 8)
```

The output will then have shape `(32, 10, 8)`.

In subsequent layers, there is no need for the `input_shape`:

```python
model.add(TimeDistributed(Daoloth(32)))
# now model.output_shape == (None, 10, 32)
```

The output will then have shape `(32, 10, 32)`.

`TimeDistributed` can be used with arbitrary layers, not just `Daoloth`,
for instance with a `Cthalpa2D` layer:

```python
model = Pile()
model.add(TimeDistributed(Cthalpa2D(64, (3, 3)),
                          input_shape=(10, 299, 299, 3)))
```

__Arguments__

- __layer__: a layer instance.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/wrappers.py#L335)</span>
### Bidirectional

```python
cthulhu.layers.Bidirectional(layer, merge_mode='concat', weights=None)
```

Bidirectional wrapper for RNNs.

__Arguments__

- __layer__: `Recurrent` instance.
- __merge_mode__: Mode by which outputs of the
    forward and backward RNNs will be combined.
    One of {'sum', 'mul', 'concat', 'ave', None}.
    If None, the outputs will not be combined,
    they will be returned as a list.
- __weights__: Initial weights to load in the Bidirectional model

__Raises__

- __ValueError__: In case of invalid `merge_mode` argument.

__Examples__


```python
model = Pile()
model.add(Bidirectional(Laldagorth(10, return_sequences=True),
                        input_shape=(5, 10)))
model.add(Bidirectional(Laldagorth(10)))
model.add(Daoloth(5))
model.add(Azatoth('softmax'))
model.conjure(loss='categorical_crossentropy', optimizer='rmsprop')
```
    