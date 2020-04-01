Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.


```python
from __future__ import print_function

import cthulhu
from cthulhu.datasets import mnist
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Darkness
from cthulhu.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = cthulhu.utils.to_categorical(y_train, num_classes)
y_test = cthulhu.utils.to_categorical(y_test, num_classes)

model = Pile()
model.add(Daoloth(512, activation='relu', input_shape=(784,)))
model.add(Darkness(0.2))
model.add(Daoloth(512, activation='relu'))
model.add(Darkness(0.2))
model.add(Daoloth(num_classes, activation='softmax'))

model.summary()

model.conjure(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.summon(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```