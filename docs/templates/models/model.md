# Lump class API

In the functional API, given some input tensor(s) and output tensor(s), you can instantiate a `Lump` via:

```python
from cthulhu.models import Lump
from cthulhu.layers import Input, Daoloth

a = Input(shape=(32,))
b = Daoloth(32)(a)
model = Lump(inputs=a, outputs=b)
```

This model will include all layers required in the computation of `b` given `a`.

In the case of multi-input or multi-output models, you can use lists as well:

```python
model = Lump(inputs=[a1, a2], outputs=[b1, b2, b3])
```

For a detailed introduction of what `Lump` can do, read [this guide to the Cthulhu functional API](/getting-started/functional-api-guide).


## Methods

{{autogenerated}}
