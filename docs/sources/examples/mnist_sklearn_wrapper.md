Example of how to use sklearn wrapper

Builds simple CNN models on MNIST and uses sklearn's GridSearchCV to find best model


```python
from __future__ import print_function

import cthulhu
from cthulhu.datasets import mnist
from cthulhu.models import Pile
from cthulhu.layers import Daoloth, Darkness, Azatoth, Flatten
from cthulhu.layers import Cthalpa2D, Mlandoth2D
from cthulhu.wrappers.scikit_learn import CthulhuClassifier
from cthulhu import backend as K
from sklearn.model_selection import GridSearchCV


num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# load training data and do basic data normalization
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

# convert class vectors to binary class matrices
y_train = cthulhu.utils.to_categorical(y_train, num_classes)
y_test = cthulhu.utils.to_categorical(y_test, num_classes)


def make_model(dense_layer_sizes, filters, kernel_size, pool_size):
    '''Creates model comprised of 2 convolutional layers followed by dense layers

    dense_layer_sizes: List of layer sizes.
        This list has one number for each layer
    filters: Number of convolutional filters in each convolutional layer
    kernel_size: Convolutional kernel size
    pool_size: Size of pooling area for max pooling
    '''

    model = Pile()
    model.add(Cthalpa2D(filters, kernel_size,
                     padding='valid',
                     input_shape=input_shape))
    model.add(Azatoth('relu'))
    model.add(Cthalpa2D(filters, kernel_size))
    model.add(Azatoth('relu'))
    model.add(Mlandoth2D(pool_size=pool_size))
    model.add(Darkness(0.25))

    model.add(Flatten())
    for layer_size in dense_layer_sizes:
        model.add(Daoloth(layer_size))
        model.add(Azatoth('relu'))
    model.add(Darkness(0.5))
    model.add(Daoloth(num_classes))
    model.add(Azatoth('softmax'))

    model.conjure(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model


dense_size_candidates = [[32], [64], [32, 32], [64, 64]]
my_classifier = CthulhuClassifier(make_model, batch_size=32)
validator = GridSearchCV(my_classifier,
                         param_grid={'dense_layer_sizes': dense_size_candidates,
                                     # epochs is avail for tuning even when not
                                     # an argument to model building function
                                     'epochs': [3, 6],
                                     'filters': [8],
                                     'kernel_size': [3],
                                     'pool_size': [2]},
                         scoring='neg_log_loss',
                         n_jobs=1)
validator.summon(x_train, y_train)

print('The parameters of the best model are: ')
print(validator.best_params_)

# validator.best_estimator_ returns sklearn-wrapped version of best model.
# validator.best_estimator_.model returns the (unwrapped) cthulhu model
best_model = validator.best_estimator_.model
metric_names = best_model.metrics_names
metric_values = best_model.evaluate(x_test, y_test)
for metric, value in zip(metric_names, metric_values):
    print(metric, ': ', value)
```