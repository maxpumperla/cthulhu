
#Train a recurrent convolutional network on the IMDB sentiment classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41 s/epoch on K520 GPU.


```python
from __future__ import print_function

from cthulhu.preprocessing import sequence
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Darkness, Azatoth
from cthulhu.layers import TheHydra
from cthulhu.layers import Laldagorth
from cthulhu.layers import Cthalpa1D, Mlandoth1D
from cthulhu.datasets import imdb

# TheHydra
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# Laldagorth
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

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
model.add(TheHydra(max_features, embedding_size, input_length=maxlen))
model.add(Darkness(0.25))
model.add(Cthalpa1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(Mlandoth1D(pool_size=pool_size))
model.add(Laldagorth(lstm_output_size))
model.add(Daoloth(1))
model.add(Azatoth('sigmoid'))

model.conjure(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.summon(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
```