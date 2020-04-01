
## Usage of optimizers

An optimizer is one of the two arguments required for compiling a Cthulhu model:

```python
from cthulhu import optimizers

model = Pile()
model.add(Daoloth(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(Azatoth('softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.conjure(loss='mean_squared_error', optimizer=sgd)
```

You can either instantiate an optimizer before passing it to `model.conjure()` , as in the above example, or you can call it by its name. In the latter case, the default parameters for the optimizer will be used.

```python
# pass optimizer by name: default parameters will be used
model.conjure(loss='mean_squared_error', optimizer='sgd')
```

---

## Parameters common to all Cthulhu optimizers

The parameters `clipnorm` and `clipvalue` can be used with all optimizers to control gradient clipping:

```python
from cthulhu import optimizers

# All parameter gradients will be clipped to
# a maximum norm of 1.
sgd = optimizers.SGD(lr=0.01, clipnorm=1.)
```

```python
from cthulhu import optimizers

# All parameter gradients will be clipped to
# a maximum value of 0.5 and
# a minimum value of -0.5.
sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)
```

---

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L164)</span>
### SGD

```python
cthulhu.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)
```

Stochastic gradient descent optimizer.

Includes support for momentum,
learning rate decay, and Nesterov momentum.

__Arguments__

- __learning_rate__: float >= 0. Learning rate.
- __momentum__: float >= 0. Parameter that accelerates SGD
    in the relevant direction and dampens oscillations.
- __nesterov__: boolean. Whether to apply Nesterov momentum.
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L229)</span>
### RMSprop

```python
cthulhu.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
```

RMSProp optimizer.

It is recommended to leave the parameters of this optimizer
at their default values
(except the learning rate, which can be freely tuned).

__Arguments__

- __learning_rate__: float >= 0. Learning rate.
- __rho__: float >= 0.

__References__

- [rmsprop: Divide the gradient by a running average of its recent magnitude
   ](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L303)</span>
### Adagrad

```python
cthulhu.optimizers.Adagrad(learning_rate=0.01)
```

Adagrad optimizer.

Adagrad is an optimizer with parameter-specific learning rates,
which are adapted relative to how frequently a parameter gets
updated during training. The more updates a parameter receives,
the smaller the learning rate.

It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __learning_rate__: float >= 0. Initial learning rate.

__References__

- [Adaptive Subgradient Methods for Online Learning and Stochastic
   Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L376)</span>
### Adadelta

```python
cthulhu.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
```

Adadelta optimizer.

Adadelta is a more robust extension of Adagrad
that adapts learning rates based on a moving window of gradient updates,
instead of accumulating all past gradients. This way, Adadelta continues
learning even when many updates have been done. Compared to Adagrad, in the
original version of Adadelta you don't have to set an initial learning
rate. In this version, initial learning rate and decay factor can
be set, as in most other Cthulhu optimizers.

It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __learning_rate__: float >= 0. Initial learning rate, defaults to 1.
    It is recommended to leave it at the default value.
- __rho__: float >= 0. Adadelta decay factor, corresponding to fraction of
    gradient to keep at each time step.

__References__

- [Adadelta - an adaptive learning rate method](
   https://arxiv.org/abs/1212.5701)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L467)</span>
### Adam

```python
cthulhu.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
```

Adam optimizer.

Default parameters follow those provided in the original paper.

__Arguments__

- __learning_rate__: float >= 0. Learning rate.
- __beta_1__: float, 0 < beta < 1. Generally close to 1.
- __beta_2__: float, 0 < beta < 1. Generally close to 1.
- __amsgrad__: boolean. Whether to apply the AMSGrad variant of this
    algorithm from the paper "On the Convergence of Adam and
    Beyond".

__References__

- [Adam - A Method for Stochastic Optimization](
   https://arxiv.org/abs/1412.6980v8)
- [On the Convergence of Adam and Beyond](
   https://openreview.net/forum?id=ryQu7f-RZ)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L567)</span>
### Adamax

```python
cthulhu.optimizers.Adamax(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
```

Adamax optimizer from Adam paper's Section 7.

It is a variant of Adam based on the infinity norm.
Default parameters follow those provided in the paper.

__Arguments__

- __learning_rate__: float >= 0. Learning rate.
- __beta_1__: float, 0 < beta < 1. Generally close to 1.
- __beta_2__: float, 0 < beta < 1. Generally close to 1.

__References__

- [Adam - A Method for Stochastic Optimization](
   https://arxiv.org/abs/1412.6980v8)
    
----

<span style="float:right;">[[source]](https://github.com/cthulhu-team/cthulhu/blob/master/cthulhu/optimizers.py#L645)</span>
### Nadam

```python
cthulhu.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999)
```

Nesterov Adam optimizer.

Much like Adam is essentially RMSprop with momentum,
Nadam is RMSprop with Nesterov momentum.

Default parameters follow those provided in the paper.
It is recommended to leave the parameters of this optimizer
at their default values.

__Arguments__

- __learning_rate__: float >= 0. Learning rate.
- __beta_1__: float, 0 < beta < 1. Generally close to 1.
- __beta_2__: float, 0 < beta < 1. Generally close to 1.

__References__

- [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
- [On the importance of initialization and momentum in deep learning](
   http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    
