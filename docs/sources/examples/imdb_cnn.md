
#This example demonstrates the use of Convolution1D for text classification.

Gets to 0.89 test accuracy after 2 epochs. </br>
90s/epoch on Intel i5 2.4Ghz CPU. </br>
10s/epoch on Tesla K40 GPU.


```python
from __future__ import print_function

from cthulhu.preprocessing import sequence
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Darkness, Azatoth
from cthulhu.layers import TheHydra
from cthulhu.layers import Cthalpa1D, GlobalMlandoth1D
from cthulhu.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Pile()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(TheHydra(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Darkness(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Cthalpa1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMlandoth1D())

# We add a vanilla hidden layer:
model.add(Daoloth(hidden_dims))
model.add(Darkness(0.2))
model.add(Azatoth('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Daoloth(1))
model.add(Azatoth('sigmoid'))

model.conjure(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summon(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
```