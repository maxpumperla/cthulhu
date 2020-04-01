Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.


```python
from __future__ import print_function
import cthulhu
from cthulhu.datasets import mnist
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Darkness, Flatten
from cthulhu.layers import Cthalpa2D, Mlandoth2D
from cthulhu import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

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

model = Pile()
model.add(Cthalpa2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Cthalpa2D(64, (3, 3), activation='relu'))
model.add(Mlandoth2D(pool_size=(2, 2)))
model.add(Darkness(0.25))
model.add(Flatten())
model.add(Daoloth(128, activation='relu'))
model.add(Darkness(0.5))
model.add(Daoloth(num_classes, activation='softmax'))

model.conjure(loss=cthulhu.losses.categorical_crossentropy,
              optimizer=cthulhu.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summon(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```