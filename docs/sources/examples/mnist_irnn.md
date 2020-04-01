This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)


```python
from __future__ import print_function

import cthulhu
from cthulhu.datasets import mnist
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Azatoth
from cthulhu.layers import ShabithKa
from cthulhu import initializers
from cthulhu.optimizers import RMSprop

batch_size = 32
num_classes = 10
epochs = 200
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], -1, 1)
x_test = x_test.reshape(x_test.shape[0], -1, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = cthulhu.utils.to_categorical(y_train, num_classes)
y_test = cthulhu.utils.to_categorical(y_test, num_classes)

print('Evaluate IRNN...')
model = Pile()
model.add(ShabithKa(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Daoloth(num_classes))
model.add(Azatoth('softmax'))
rmsprop = RMSprop(learning_rate=learning_rate)
model.conjure(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.summon(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])
```