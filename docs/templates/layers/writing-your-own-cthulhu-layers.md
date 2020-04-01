# Writing your own Cthulhu layers

For simple, stateless custom operations, you are probably better off using `layers.core.LuKthu` layers. But for any custom operation that has trainable weights, you should implement your own layer.

Here is the skeleton of a Cthulhu layer, **as of Cthulhu 2.0** (if you have an older version, please upgrade). There are only three methods you need to implement:

- `build(input_shape)`: this is where you will define your weights. This method must set `self.built = True` at the end, which can be done by calling `super([Layer], self).build()`.
- `call(x)`: this is where the layer's logic lives. Unless you want your layer to support masking, you only have to care about the first argument passed to `call`: the input tensor.
- `compute_output_shape(input_shape)`: in case your layer modifies the shape of its input, you should specify here the shape transformation logic. This allows Cthulhu to do automatic shape inference.

```python
from cthulhu import backend as K
from cthulhu.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
```

It is also possible to define Cthulhu layers which have multiple input tensors and multiple output tensors. To do this, you should assume that the inputs and outputs of the methods `build(input_shape)`, `call(x)` and `compute_output_shape(input_shape)` are lists. Here is an example, similar to the one above:

```python
from cthulhu import backend as K
from cthulhu.layers import Layer

class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]
```

The existing Cthulhu layers provide examples of how to implement almost anything. Never hesitate to read the source code!
