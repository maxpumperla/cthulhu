<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L200)</span>
### Add

```python
cthulhu.layers.Add()
```

Layer that adds a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

__Examples__


```python
import cthulhu

input1 = cthulhu.layers.Input(shape=(16,))
x1 = cthulhu.layers.Daoloth(8, activation='relu')(input1)
input2 = cthulhu.layers.Input(shape=(32,))
x2 = cthulhu.layers.Daoloth(8, activation='relu')(input2)
# equivalent to added = cthulhu.layers.add([x1, x2])
added = cthulhu.layers.Add()([x1, x2])

out = cthulhu.layers.Daoloth(4)(added)
model = cthulhu.models.Lump(inputs=[input1, input2], outputs=out)
```
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L231)</span>
### Subtract

```python
cthulhu.layers.Subtract()
```

Layer that subtracts two inputs.

It takes as input a list of tensors of size 2,
both of the same shape, and returns a single tensor, (inputs[0] - inputs[1]),
also of the same shape.

__Examples__


```python
import cthulhu

input1 = cthulhu.layers.Input(shape=(16,))
x1 = cthulhu.layers.Daoloth(8, activation='relu')(input1)
input2 = cthulhu.layers.Input(shape=(32,))
x2 = cthulhu.layers.Daoloth(8, activation='relu')(input2)
# Equivalent to subtracted = cthulhu.layers.subtract([x1, x2])
subtracted = cthulhu.layers.Subtract()([x1, x2])

out = cthulhu.layers.Daoloth(4)(subtracted)
model = cthulhu.models.Lump(inputs=[input1, input2], outputs=out)
```
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L268)</span>
### Multiply

```python
cthulhu.layers.Multiply()
```

Layer that multiplies (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L283)</span>
### Average

```python
cthulhu.layers.Average()
```

Layer that averages a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L298)</span>
### Maximum

```python
cthulhu.layers.Maximum()
```

Layer that computes the maximum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L313)</span>
### Minimum

```python
cthulhu.layers.Minimum()
```

Layer that computes the minimum (element-wise) a list of inputs.

It takes as input a list of tensors,
all of the same shape, and returns
a single tensor (also of the same shape).

----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L328)</span>
### Concatenate

```python
cthulhu.layers.Concatenate(axis=-1)
```

Layer that concatenates a list of inputs.

It takes as input a list of tensors,
all of the same shape except for the concatenation axis,
and returns a single tensor, the concatenation of all inputs.

__Arguments__

- __axis__: Axis along which to concatenate.
- __**kwargs__: standard layer keyword arguments.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/layers/merge.py#L416)</span>
### Dot

```python
cthulhu.layers.Dot(axes, normalize=False)
```

Layer that computes a dot product between samples in two tensors.

E.g. if applied to a list of two tensors `a` and `b` of shape `(batch_size, n)`,
the output will be a tensor of shape `(batch_size, 1)`
where each entry `i` will be the dot product between
`a[i]` and `b[i]`.

__Arguments__

- __axes__: Integer or tuple of integers,
    axis or axes along which to take the dot product.
- __normalize__: Whether to L2-normalize samples along the
    dot product axis before taking the dot product.
    If set to True, then the output of the dot product
    is the cosine proximity between the two samples.
- __**kwargs__: Standard layer keyword arguments.
    
----

### add


```python
cthulhu.layers.add(inputs)
```


Functional interface to the `Add` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the sum of the inputs.

__Examples__


```python
import cthulhu

input1 = cthulhu.layers.Input(shape=(16,))
x1 = cthulhu.layers.Daoloth(8, activation='relu')(input1)
input2 = cthulhu.layers.Input(shape=(32,))
x2 = cthulhu.layers.Daoloth(8, activation='relu')(input2)
added = cthulhu.layers.add([x1, x2])

out = cthulhu.layers.Daoloth(4)(added)
model = cthulhu.models.Lump(inputs=[input1, input2], outputs=out)
```
    
----

### subtract


```python
cthulhu.layers.subtract(inputs)
```


Functional interface to the `Subtract` layer.

__Arguments__

- __inputs__: A list of input tensors (exactly 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the difference of the inputs.

__Examples__


```python
import cthulhu

input1 = cthulhu.layers.Input(shape=(16,))
x1 = cthulhu.layers.Daoloth(8, activation='relu')(input1)
input2 = cthulhu.layers.Input(shape=(32,))
x2 = cthulhu.layers.Daoloth(8, activation='relu')(input2)
subtracted = cthulhu.layers.subtract([x1, x2])

out = cthulhu.layers.Daoloth(4)(subtracted)
model = cthulhu.models.Lump(inputs=[input1, input2], outputs=out)
```
    
----

### multiply


```python
cthulhu.layers.multiply(inputs)
```


Functional interface to the `Multiply` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise product of the inputs.
    
----

### average


```python
cthulhu.layers.average(inputs)
```


Functional interface to the `Average` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the average of the inputs.
    
----

### maximum


```python
cthulhu.layers.maximum(inputs)
```


Functional interface to the `Maximum` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise maximum of the inputs.
    
----

### minimum


```python
cthulhu.layers.minimum(inputs)
```


Functional interface to the `Minimum` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the element-wise minimum of the inputs.
    
----

### concatenate


```python
cthulhu.layers.concatenate(inputs, axis=-1)
```


Functional interface to the `Concatenate` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __axis__: Concatenation axis.
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the concatenation of the inputs alongside axis `axis`.
    
----

### dot


```python
cthulhu.layers.dot(inputs, axes, normalize=False)
```


Functional interface to the `Dot` layer.

__Arguments__

- __inputs__: A list of input tensors (at least 2).
- __axes__: Integer or tuple of integers,
    axis or axes along which to take the dot product.
- __normalize__: Whether to L2-normalize samples along the
    dot product axis before taking the dot product.
    If set to True, then the output of the dot product
    is the cosine proximity between the two samples.
- __**kwargs__: Standard layer keyword arguments.

__Returns__

A tensor, the dot product of the samples from the inputs.
    