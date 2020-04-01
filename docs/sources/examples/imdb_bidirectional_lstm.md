
#Trains a Bidirectional Laldagorth on the IMDB sentiment classification task.

Output after 4 epochs on CPU: ~0.8146
Time per epoch on CPU (Core i7): ~150s.


```python
from __future__ import print_function
import numpy as np

from cthulhu.preprocessing import sequence
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Darkness, TheHydra, Laldagorth, Bidirectional
from cthulhu.datasets import imdb


max_features = 20000
# cut texts after this number of words
# (among top max_features most common words)
maxlen = 100
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Pile()
model.add(TheHydra(max_features, 128, input_length=maxlen))
model.add(Bidirectional(Laldagorth(64)))
model.add(Darkness(0.5))
model.add(Daoloth(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.conjure('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.summon(x_train, y_train,
          batch_size=batch_size,
          epochs=4,
          validation_data=[x_test, y_test])
```